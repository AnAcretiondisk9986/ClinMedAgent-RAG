"""结构化产物的暂存、原子替换与文件命名。

背景问题：``pdftext.build_structured`` 与 ``tools_structure_v3`` 都直接往
``processed_v3/{cleaned,structured,quality.json}`` 写文件。如果这一次生成的章节
比上一次少（章节合并、章标题识别结果变化、页数变化），旧章节 Markdown 不会被
删除，重新建索引时过期内容仍会进入检索结果。

做法：生成器先把产物写进 ``processed_v3/.staging-xxxx``，全部成功后整体替换旧
内容（旧内容先移入 ``.backup-xxxx``，替换成功后删除）。中途失败时回滚，旧结果
保持可用且不会出现“半套新结果”。
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
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


def _commit(target: Path, staging: Path, managed: tuple[str, ...]) -> None:
    """把暂存目录里的 ``managed`` 项替换到 ``target``；失败则回滚旧内容。"""
    backup = Path(tempfile.mkdtemp(prefix=BACKUP_PREFIX, dir=str(target)))
    saved: list[tuple[str, Path]] = []   # (原始名字, 备份路径)
    placed: list[Path] = []
    try:
        for name in managed:
            destination = target / name
            if destination.exists() or destination.is_symlink():
                os.replace(destination, backup / name)
                saved.append((name, backup / name))
        for name in managed:
            source = staging / name
            if source.exists():
                os.replace(source, target / name)
                placed.append(target / name)
    except BaseException:
        for path in placed:
            _remove(path)
        for name, backup_path in saved:
            if backup_path.exists():
                try:
                    os.replace(backup_path, target / name)
                except OSError:
                    pass
        raise
    finally:
        shutil.rmtree(backup, ignore_errors=True)


def _remove(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass


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
