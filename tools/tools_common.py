"""tools/ 各流水线脚本共用的产物读写工具。

只依赖标准库：fitz / PIL 都在函数内部惰性导入——这些脚本分别跑在主环境、
.venv-ocr、.venv-ocr312 三个不同虚拟环境里，模块顶层的重依赖会让另一侧
直接 import 失败。

这里集中三件事，都是“产物写坏了要重跑数小时 GPU”的对策：

1. :func:`atomic_write_text` —— 同目录临时文件 + ``os.replace``；
2. :func:`load_page_json` / :func:`valid_page_json` —— 续跑前校验 JSON 真的完整；
3. :func:`cached_render` —— 页图缓存带来源指纹，换了一版 PDF 不会复用旧图。
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    """原子写文本：先写同目录临时文件，再 ``os.replace`` 覆盖目标。

    中途被取消/超时/断电时，目标文件要么还是旧的完整内容，要么根本没生成，
    不会留下“文件存在但内容截断”的半截 JSON——那正是续跑必然再崩的根源。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".tmp-")
    try:
        with os.fdopen(handle, "w", encoding=encoding, newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def load_page_json(path: Path | str, page: int | None = None) -> dict[str, Any] | None:
    """读取单页产物；截断/损坏/页号不符都返回 ``None``。

    把“文件存在”当“内容完整”是续跑的地雷：一次 kill 留下半截 JSON，
    ``exists()`` 判据会让这一页永远不再重做，而 manifest 汇总时必定崩。
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("boxes"), list):
        return None
    if page is not None and payload.get("page") != page:
        return None
    return payload


def valid_page_json(path: Path | str, page: int | None = None) -> bool:
    """续跑判据：``load_page_json`` 能读出完整内容才算这一页真的做完了。"""
    return load_page_json(path, page) is not None


def render_meta(pdf: Path | str, page: int, dpi: int) -> dict[str, Any]:
    """页图缓存的来源指纹：PDF 路径 + size + mtime + dpi。"""
    path = Path(pdf)
    try:
        stat = path.stat()
        size, mtime = stat.st_size, round(stat.st_mtime, 6)
    except OSError:
        size, mtime = -1, -1.0
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    return {"page": int(page), "dpi": int(dpi), "size": int(size), "mtime": mtime, "pdf": resolved}


def _meta_matches(meta_path: Path, expected: dict[str, Any]) -> bool:
    try:
        stored = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(stored, dict) and all(stored.get(key) == value for key, value in expected.items())


def cached_render(
    pdf: Path | str, page: int, dpi: int, cache_dir: Path | str
) -> tuple[Path, int, int]:
    """渲染（或复用）第 ``page`` 页的 300 DPI 页图，返回 (路径, 宽, 高)。

    命中条件不只是 ``png 存在``：还必须有一份来源 meta（PDF size/mtime/dpi）
    全等。否则“同目录换了一版 PDF”会把旧页图复用给新书，表格 HTML 对着错页
    生成、页码全错却静默入索引（映射到引用契约就是错误引用）。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"page-{int(page):04d}.png"
    meta_path = cache_dir / f"page-{int(page):04d}.meta.json"
    expected = render_meta(pdf, page, dpi)
    if not (out.exists() and _meta_matches(meta_path, expected)):
        import fitz

        with fitz.open(str(pdf)) as doc:
            target = doc[int(page) - 1]
            scale = dpi / 72
            pix = target.get_pixmap(
                matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False
            )
            pix.save(str(out))
        atomic_write_text(
            meta_path, json.dumps(expected, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    from PIL import Image

    with Image.open(out) as image:
        return out, image.width, image.height
