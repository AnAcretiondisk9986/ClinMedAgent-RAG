"""结构化产物的暂存、原子替换与文件命名。

背景问题：``pdftext.build_structured`` 与 ``tools/tools_structure_v3`` 都直接往
``processed_v3/{cleaned,structured,quality.json}`` 写文件。如果这一次生成的章节
比上一次少（章节合并、章标题识别结果变化、页数变化），旧章节 Markdown 不会被
删除，重新建索引时过期内容仍会进入检索结果。

做法：生成器先把产物写进 ``processed_v3/.staging-xxxx``，全部成功后整体替换旧
内容（旧内容先移入 ``.backup-xxxx``，替换成功后删除）。中途失败时回滚，旧结果
保持可用且不会出现“半套新结果”。

回滚也失败的极端情况（产物被别的安全上下文写过、Windows 一律拒绝访问）不再
静默吞掉：备份目录会保留下来，并抛 ``OutputCommitError`` 说明哪些项没放回去。
``blocked_outputs`` 供 ``doctor`` 提前发现这种状态。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

MANAGED_OUTPUTS: tuple[str, ...] = ("cleaned", "structured", "quality.json")
STAGING_PREFIX = ".staging-"
BACKUP_PREFIX = ".backup-"

_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def staged_output_dir(
    target: Path | str,
    managed: Sequence[str] = MANAGED_OUTPUTS,
) -> "contextmanager[Path]":
    """在 ``target`` 下开一个暂存目录；with 正常退出时原子替换 ``managed`` 各项。

    ``managed`` 是相对 ``target`` 的名字列表。生成器只写暂存目录；如果某项在暂存
    目录里不存在，对应的旧产物会被删除（这正是清理陈旧章节的关键）。
    """
    return _staged_output_dir(Path(target), tuple(managed))


@contextmanager
def _staged_output_dir(target: Path, managed: tuple[str, ...]) -> Iterator[Path]:
    target.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=str(target)))
    try:
        yield staging
        _commit(target, staging, managed)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _replace_with_retry(source: Path, destination: Path, attempts: int = 8) -> None:
    """Replace a file/directory, tolerating short Windows sharing violations.

    Explorer, antivirus scanners, editors, and PDF/Markdown previewers can hold
    a handle briefly while a generated directory is being swapped.  A direct
    ``os.replace`` turns that transient condition into an opaque exit-code-1
    pipeline failure, so retry only ``PermissionError`` with a short backoff.
    Persistent failures still raise a contextual error so the UI tells the user
    which output must be closed.
    """
    attempts = max(1, int(attempts))
    last_error: PermissionError | None = None
    for attempt in range(attempts):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            last_error = exc
            if os.name != "nt" or attempt == attempts - 1:
                break
            time.sleep(min(0.25 * (attempt + 1), 1.0))
    assert last_error is not None
    raise PermissionError(
        f"无法替换处理产物：{source} → {destination}。"
        "Windows 拒绝访问；请关闭正在打开 processed_v3 的资源管理器、编辑器、"
        "PDF/Markdown 预览或杀毒扫描后重试。"
        "如果关掉这些程序仍然报同样的错，说明产物本身被别的安全上下文（其它账号、"
        "沙箱/Agent 环境）写过，ACL 异常：先试 runas 管理员 takeown /f \"<processed_v3>\" /r /d y"
        " 再 icacls \"<processed_v3>\" /reset /t /c /q；不需要旧产物的话，直接把"
        " processed_v3 改名搁置（例如 processed_v3.broken），重新生成即可。"
    ) from last_error


class OutputCommitError(RuntimeError):
    """产物替换失败且旧产物无法完整放回——备份目录需要人工处理。"""


def _commit(target: Path, staging: Path, managed: tuple[str, ...]) -> None:
    """把暂存目录里的 ``managed`` 项替换到 ``target``；失败则回滚旧内容。

    回滚必须先把新产物清掉再放回备份：反过来的话，新产物占着目的地，备份只能
    被卡在备份目录里。清理失败的项**不再硬恢复**——恢复失败会连同旧产物一起丢，
    宁可保留备份目录交给人处理。
    """
    backup = Path(tempfile.mkdtemp(prefix=BACKUP_PREFIX, dir=str(target)))
    saved: list[tuple[str, Path]] = []   # (原始名字, 备份路径)
    placed: list[Path] = []
    keep_backup = False
    try:
        for name in managed:
            destination = target / name
            if destination.exists() or destination.is_symlink():
                _replace_with_retry(destination, backup / name)
                saved.append((name, backup / name))
        for name in managed:
            source = staging / name
            if source.exists():
                _replace_with_retry(source, target / name)
                placed.append(target / name)
    except BaseException as exc:
        unrecovered = _rollback(target, placed, saved)
        if unrecovered:
            keep_backup = True
            names = "、".join(unrecovered)
            raise OutputCommitError(
                f"{exc}\n回滚失败：{names} 无法放回原位，旧产物暂时留在 {backup}，"
                "请手工处理（关掉占用程序后把备份目录里的内容搬回 processed_v3，"
                "或按上面的提示用管理员 takeown/icacls 清理后重新生成）。"
            ) from exc
        raise
    finally:
        if not keep_backup:
            shutil.rmtree(backup, ignore_errors=True)


def _rollback(target: Path, placed: list[Path], saved: list[tuple[str, Path]]) -> list[str]:
    """尽力把旧产物放回原位；返回没能恢复的项名（空列表表示回滚干净）。"""
    stuck = {path.name for path in placed if not _remove(path)}
    unrecovered: list[str] = []
    for name, backup_path in saved:
        if name in stuck:
            unrecovered.append(name)
            continue
        if not backup_path.exists():
            unrecovered.append(name)
            continue
        try:
            _replace_with_retry(backup_path, target / name, attempts=3)
        except OSError:
            unrecovered.append(name)
    return unrecovered


def _remove(path: Path) -> bool:
    """删除文件/目录；返回是否确认已删除（失败不抛异常）。"""
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass
    return not (path.exists() or path.is_symlink())


def blocked_outputs(
    target: Path | str,
    managed: Sequence[str] = MANAGED_OUTPUTS,
) -> list[str]:
    """列出 ``processed_v3`` 下“名字还在、但当前账号打不开”的产物。

    元数据（大小/时间）正常、读属性也正常，却对读、改名、删除一律返回
    ERROR_ACCESS_DENIED：典型是产物被别的安全上下文（其它账号、沙箱/Agent 环境）
    写过，带上了异常的所有者/ACL。这种目录每次生成都会卡在“把旧产物挪进备份”
    这一步，关掉资源管理器/杀毒也没用，只能 takeown + icacls /reset，或把
    processed_v3 整体改名搁置后重新生成。
    """
    root = Path(target)
    blocked: list[str] = []
    for name in managed:
        path = root / name
        try:
            if not (path.exists() or path.is_symlink()):
                continue
        except OSError:  # 连属性都读不到，更说明有问题
            blocked.append(name)
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                with os.scandir(path) as entries:
                    next(iter(entries), None)
            else:
                with open(path, "rb"):
                    pass
        except OSError:
            blocked.append(name)
    return blocked


def is_internal_output(path: Path | str) -> bool:
    """暂存/备份目录不是正式产物，遍历 Markdown 建索引时必须跳过。"""
    return any(
        part.startswith((STAGING_PREFIX, BACKUP_PREFIX))
        for part in Path(path).parts
    )


def safe_file_stem(value: str, fallback: str = "chapter", max_length: int = 80) -> str:
    """把章节标题转成安全的文件名主体（Windows 兼容，保留中文）。

    ``f"02-{title}.md"`` 在章标题含 ``/`` 或 ``:`` 时会写到错误路径甚至直接
    失败，这里统一替换非法字符并限制长度。
    """
    stem = _INVALID_NAME_CHARS.sub("_", str(value or ""))
    stem = re.sub(r"\s+", " ", stem).strip().rstrip(". ")
    if not stem:
        return fallback
    return stem[:max_length].strip().rstrip(". ") or fallback


def chapter_file_name(num: int, title: str, suffix: str = ".md") -> str:
    """章节 Markdown 文件名的统一入口：``02-关节学.md``。"""
    return f"{int(num):02d}-{safe_file_stem(title, fallback=f'第{num}章')}{suffix}"
