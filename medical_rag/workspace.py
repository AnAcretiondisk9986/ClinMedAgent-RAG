"""教材工作区：发现 res/ 下的教材目录，汇总处理状态（PDF / OCR / 版面 / 表格 / 结构化 / 索引）。

本地网站用它渲染书库列表和教材状态；处理流水线用它判断哪些阶段还没跑。
不修改任何文件，只做只读扫描。
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .library import DEFAULT_DB
from .pdftext import cached_analyze

PDF_PAGE_CACHE: dict[tuple[str, float], int] = {}
_INVALID_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Windows 保留设备名：即使带扩展名（CON.pdf、NUL.pdf）也无法创建，目录名与
# 文件名共用这一份定义，避免两处实现漂移
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def book_id_for(rel_dir: str) -> str:
    """由相对路径生成稳定的教材 ID（URL 用）。"""
    return hashlib.sha1(rel_dir.replace("\\", "/").encode("utf-8")).hexdigest()[:10]


def safe_dir_name(title: str, fallback: str = "未命名教材") -> str:
    """把书名转成安全的目录名（Windows 兼容）。"""
    name = _INVALID_DIR_CHARS.sub("", str(title)).strip().strip(".")
    name = re.sub(r"\s+", " ", name)
    if not name or name.upper() in WINDOWS_RESERVED_NAMES:
        return fallback
    return name[:60]


def find_pdf(book_dir: Path) -> Path | None:
    """查找教材主 PDF：优先 <book>/PDF/*.pdf，其次 <book>/*.pdf。"""
    pdf_dir = book_dir / "PDF"
    candidates = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.is_dir() else []
    if not candidates:
        candidates = sorted(book_dir.glob("*.pdf"))
    return candidates[0] if candidates else None


def pdf_page_count(pdf: Path) -> int | None:
    """PDF 页数（按路径 + mtime 缓存）。"""
    try:
        key = (str(pdf), pdf.stat().st_mtime)
    except OSError:
        return None
    if key in PDF_PAGE_CACHE:
        return PDF_PAGE_CACHE[key]
    try:
        import fitz

        with fitz.open(pdf) as doc:
            count = len(doc)
    except Exception:
        return None
    PDF_PAGE_CACHE[key] = count
    return count


def _count_files(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_db_books(db: Path | str = DEFAULT_DB) -> list[dict]:
    """读取索引库里的书籍记录；库不存在时返回空列表。"""
    db_path = Path(db).expanduser()
    if not db_path.exists():
        return []
    try:
        connection = sqlite3.connect(db_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute("SELECT * FROM books ORDER BY id").fetchall()
            stats = {
                row["book_id"]: (row["chunks"], row["pages"])
                for row in connection.execute(
                    "SELECT book_id, COUNT(*) AS chunks, COUNT(DISTINCT page) AS pages FROM chunks GROUP BY book_id"
                )
            }
            books = []
            for row in rows:
                item = dict(row)
                chunks, chunk_pages = stats.get(row["id"], (0, 0))
                item["chunks"] = chunks
                item["chunk_pages"] = chunk_pages
                books.append(item)
            return books
        finally:
            connection.close()
    except sqlite3.Error:
        return []


def _match_db_book(db_books: list[dict], book_dir: Path) -> dict | None:
    """找出属于该教材目录的索引记录（processed_v3 或其他子目录）。"""
    book_dir = book_dir.resolve()
    best: dict | None = None
    for item in db_books:
        try:
            path = Path(item["path"]).expanduser().resolve()
        except (OSError, ValueError):
            continue
        if path == book_dir or book_dir in path.parents:
            best = item
            break
    return best


def book_state(
    root: Path,
    book_dir: Path,
    db_books: list[dict] | None = None,
) -> dict[str, Any]:
    """扫描一本教材目录，返回状态字典（不读 PDF 像素，仅页数）。"""
    root = Path(root).resolve()
    book_dir = Path(book_dir).resolve()
    if db_books is None:
        db_books = load_db_books(root / ".medical_rag" / "library.sqlite3")

    pdf = find_pdf(book_dir)
    text_info: dict[str, Any] | None = None
    if pdf is not None:
        try:
            text_info = cached_analyze(pdf, root / ".medical_rag" / "pdf_text")
        except Exception:  # noqa: BLE001 - 检测失败不应影响书库扫描
            text_info = None
    text_dir = book_dir / "text_v3"
    boxes_dir = text_dir / "boxes"
    tables_dir = text_dir / "tables"
    layout = _read_json(text_dir / "layout.json")
    manifest = _read_json(text_dir / "manifest.json")
    quality = _read_json(book_dir / "processed_v3" / "quality.json")
    structured_dir = book_dir / "processed_v3" / "structured"

    boxes = _count_files(boxes_dir, "page-*.json")
    tables = _count_files(tables_dir, "page-*.md")
    structured = _count_files(structured_dir, "*.md")
    layout_pages = len(layout.get("table_pages", [])) if layout else 0

    pages: int | None = None
    if quality and isinstance(quality.get("pages"), int):
        pages = int(quality["pages"])
    elif manifest and isinstance(manifest.get("pages"), list):
        pages = len(manifest["pages"])
    elif pdf is not None:
        pages = pdf_page_count(pdf)

    db_book = _match_db_book(db_books, book_dir)
    try:
        rel_dir = book_dir.relative_to(root).as_posix()
    except ValueError:
        rel_dir = book_dir.as_posix()
    title = book_dir.name
    index_title = db_book["title"] if db_book else None

    pdf_info: dict[str, Any] = {"found": pdf is not None, "path": str(pdf) if pdf else None, "pages": pages}
    if pdf is not None:
        try:
            stat = pdf.stat()
            pdf_info.update({"size": stat.st_size, "mtime": stat.st_mtime, "name": pdf.name})
        except OSError:
            pass

    ocr_total = pages or boxes
    layout_done = layout is not None
    tables_complete = layout_done and tables >= layout_pages
    structure_done = quality is not None and structured > 0
    indexed = db_book is not None and db_book.get("chunks", 0) > 0
    text_kind = text_info.get("kind") if text_info else None
    workflow = "text" if text_kind == "text" else "ocr"

    status = {
        "workflow": workflow,
        "pdf": pdf_info,
        "text": {
            "kind": text_kind,
            "pages": text_info.get("pages", 0) if text_info else 0,
            "text_pages": text_info.get("text_pages", 0) if text_info else 0,
            "chars": text_info.get("chars", 0) if text_info else 0,
            "avg_chars": text_info.get("avg_chars", 0) if text_info else 0,
            "complete": text_kind == "text",
        },
        "ocr": {
            "done": boxes,
            "total": ocr_total,
            "complete": bool(pages and boxes >= pages),
        },
        "layout": {
            "found": layout_done,
            "table_pages": layout_pages,
            "complete": layout_done,
        },
        "tables": {
            "done": tables,
            "total": layout_pages,
            "complete": tables_complete,
        },
        "structure": {
            "found": structure_done,
            "chapters": len(quality.get("chapters", [])) if quality else 0,
            "files": structured,
            "page_offset": quality.get("page_offset") if quality else None,
            "complete": structure_done,
        },
        "index": {
            "found": indexed,
            "book_id": db_book["id"] if db_book else None,
            "title": index_title,
            "chunks": db_book.get("chunks", 0) if db_book else 0,
            "chunk_pages": db_book.get("chunk_pages", 0) if db_book else 0,
            "path": db_book["path"] if db_book else None,
            "complete": indexed,
        },
    }

    return {
        "id": book_id_for(rel_dir),
        "rel_dir": rel_dir,
        "dir": str(book_dir),
        "title": title,
        "index_title": index_title,
        "pdf": str(pdf) if pdf else None,
        "workflow": workflow,
        "status": status,
        "quality": {
            "pages": quality.get("pages") if quality else None,
            "page_offset": quality.get("page_offset") if quality else None,
            "table_pages": quality.get("table_pages", []) if quality else [],
            "chapters": quality.get("chapters", []) if quality else [],
        },
        "orphan": False,
    }


def scan_books(root: Path, db: Path | str | None = None) -> list[dict[str, Any]]:
    """扫描 res/ 下所有教材目录，并附带索引库中不在 res/ 的记录。"""
    root = Path(root).resolve()
    db_path = Path(db).expanduser() if db else root / ".medical_rag" / "library.sqlite3"
    db_books = load_db_books(db_path)
    books: list[dict[str, Any]] = []
    seen_rel: set[str] = set()

    res_dir = root / "res"
    if res_dir.is_dir():
        for child in sorted(res_dir.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name.startswith("."):
                continue
            state = book_state(root, child, db_books)
            books.append(state)
            seen_rel.add(state["rel_dir"])

    # 索引库里有、但 res/ 下没有目录的记录（例如索引指向别处）
    for item in db_books:
        path = Path(item["path"]).expanduser()
        try:
            rel = path.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            rel = None
        if rel is not None and rel in seen_rel:
            continue
        if rel is not None and any(rel == b["rel_dir"] or rel.startswith(b["rel_dir"] + "/") for b in books):
            continue
        books.append(
            {
                "id": book_id_for(str(path)),
                "rel_dir": rel or str(path),
                "dir": None,
                "title": path.stem,
                "index_title": item["title"],
                "pdf": None,
                "workflow": "ocr",
                "status": {
                    "workflow": "ocr",
                    "pdf": {"found": False, "path": None, "pages": None},
                    "text": {"kind": None, "pages": 0, "text_pages": 0, "chars": 0, "avg_chars": 0, "complete": False},
                    "ocr": {"done": 0, "total": 0, "complete": False},
                    "layout": {"found": False, "table_pages": 0, "complete": False},
                    "tables": {"done": 0, "total": 0, "complete": False},
                    "structure": {"found": False, "chapters": 0, "files": 0, "page_offset": None, "complete": False},
                    "index": {
                        "found": True,
                        "book_id": item["id"],
                        "title": item["title"],
                        "chunks": item.get("chunks", 0),
                        "chunk_pages": item.get("chunk_pages", 0),
                        "path": item["path"],
                        "complete": True,
                    },
                },
                "quality": {"pages": None, "page_offset": None, "table_pages": [], "chapters": []},
                "orphan": True,
            }
        )

    books.sort(key=lambda b: (b["orphan"], b["title"]))
    return books


def find_book(root: Path, book_id: str, db: Path | str | None = None) -> dict[str, Any] | None:
    for book in scan_books(root, db):
        if book["id"] == book_id:
            return book
    return None
