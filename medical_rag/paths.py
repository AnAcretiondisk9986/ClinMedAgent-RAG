"""项目路径解析（可移动性核心）。

索引库（``.medical_rag/library.sqlite3``）**不保存** ``F:\\...\\AI Agent检索知识库\\res\\...``
这种绝对路径，而是保存相对项目根目录的 POSIX 路径（如 ``res/系统解剖学/processed_v3``）。
读取时再用当前根目录还原成绝对路径，于是：

* 同一台电脑换目录、换盘、改名：索引照常认出教材，书卡、按书检索、页码正文都不丢；
* 整个项目复制一份放到别处：两份副本各自定位到自己那份数据，互不干扰；
* 老版本留下的绝对路径：打开库时自动迁移（:meth:`Library._migrate_paths`），
  即使迁移没成功，读取时也会按 ``res/`` 之后的相对部分重新挂到当前根目录上。

约定：
* ``store_path(root, p)``  ：写入前的归一化，返回相对 POSIX 路径或绝对路径字符串；
* ``absolute_path(root, v)``：读取时的还原，相对路径拼当前根目录，绝对路径尽量重挂；
* ``relative_key(root, v)``：比较用的规范化键（Windows 下不区分大小写）。
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

# 项目根目录（本文件在 <root>/medical_rag/paths.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 允许用环境变量把项目根目录固定到别处（例如把索引库放在项目外）
ROOT_ENV = "MEDICAL_RAG_ROOT"
# 教材资源目录名：绝对路径换根时以它为锚点
RES_DIR_NAME = "res"


def project_root() -> Path:
    """当前进程认为的项目根目录（优先环境变量，其次包所在目录）。"""
    value = os.environ.get(ROOT_ENV, "").strip()
    return Path(value).expanduser().resolve() if value else PROJECT_ROOT


def root_for_db(db: Path | str) -> Path:
    """由索引库位置推断项目根目录：``<root>/.medical_rag/x.sqlite3`` → ``<root>``。

    自定义 ``--db``（如 ``D:\\index\\library.sqlite3``）时退化为库所在目录，
    相对路径仍能自洽地写进去、读出来；要跨目录搬动请显式传 ``root``。
    """
    env = os.environ.get(ROOT_ENV, "").strip()
    if env:
        return Path(env).expanduser().resolve()
    path = Path(db).expanduser()
    try:
        path = path.resolve()
    except OSError:
        pass
    parent = path.parent
    if parent.name == ".medical_rag":
        return parent.parent
    return parent


def res_suffix(value: Path | str) -> str | None:
    """取路径里 ``res/`` 之后的部分（换根用的稳定锚点）；没有 res/ 返回 None。"""
    path = Path(str(value))
    parts = path.parts
    if RES_DIR_NAME not in parts:
        return None
    index = len(parts) - 1 - parts[::-1].index(RES_DIR_NAME)
    return PurePosixPath(*parts[index:]).as_posix()


def store_path(root: Path | str, path: Path | str) -> str:
    """把路径归一化成"项目内相对路径"，写入索引库时使用。

    - 已在项目内：``res/系统解剖学/processed_v3``
    - 旧根目录的绝对路径（项目被移动过）：按 ``res/`` 之后的部分重新相对
    - 项目外的路径（用户自己的 PDF、自定义 --db）：保持绝对路径
    """
    text = str(path or "")
    if not text:
        return text
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return PurePosixPath(candidate.as_posix()).as_posix()
    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate
    base = Path(root).expanduser()
    try:
        base = base.resolve()
    except OSError:
        pass
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        pass
    suffix = res_suffix(resolved)
    if suffix:
        return suffix
    return str(resolved)


def absolute_path(root: Path | str, value: Path | str | None) -> Path | None:
    """把库里存的值还原成绝对路径。

    绝对路径原样优先；原路径已不存在时，按 ``res/`` 之后的部分挂到当前根目录，
    这样"移动前写入、移动后读取"的老库也能用，不需要手动改库。
    """
    if value is None or str(value) == "":
        return None
    candidate = Path(str(value)).expanduser()
    base = Path(root).expanduser()
    if not candidate.is_absolute():
        return base / candidate
    try:
        if candidate.exists():
            return candidate
    except OSError:
        return candidate
    suffix = res_suffix(candidate)
    if suffix:
        return base / suffix
    return candidate


def relative_key(root: Path | str, value: Path | str | None) -> str:
    """路径比较键：忽略绝对/相对写法与大小写差异，只比项目内的相对位置。"""
    if value is None or str(value) == "":
        return ""
    text = store_path(root, value)
    return os.path.normcase(text)
