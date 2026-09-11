"""本地医学教材工作台网站（标准库 HTTP 服务，仅依赖 PyMuPDF）。

启动：
    python -m medical_rag.webapp                     # 默认 http://127.0.0.1:17173
    python -m medical_rag.webapp --port 18080        # 指定端口
    python -m medical_rag.webapp --no-browser        # 不自动打开浏览器

页面功能：
    - 显示教材列表与处理状态（PDF / OCR / 版面 / 表格 / 结构化 / 索引）；
    - 封面取 PDF 第一页，逐页预览支持输入页码跳转，并可查看该页识别文本；
    - 快捷导入教材（本地上传或直接填写路径）；
    - 一键处理教材，实时显示各阶段进度与日志。

API 概览：
    GET  /api/overview                 书库、任务、环境信息
    GET  /api/books                    教材列表
    GET  /api/books/<id>               教材详情（含章节、页范围）
    GET  /api/books/<id>/cover.png     封面（PDF 第一页）
    GET  /api/books/<id>/page/<n>.png  页面预览图（磁盘缓存）
    GET  /api/books/<id>/text/<n>      该页识别文本
    GET  /api/books/<id>/pdf           原版 PDF（支持 Range，可用 #page=N 跳页）
    POST /api/import                   按本地路径导入
    POST /api/import/begin             上传第一步：登记文件名，返回 task_id
    PUT  /api/import/upload/<task_id>  上传第二步：请求体即文件字节流
    POST /api/inspect                  检测 PDF 是否带文字层（导入前预览）
    POST /api/process                  处理教材（异步任务）
    POST /api/reindex                  只重建索引
    POST /api/search                   检索教材证据
    GET  /api/tasks                    任务列表
    GET  /api/tasks/<id>?since=N       任务快照 + 增量日志
    POST /api/tasks/<id>/cancel        取消任务
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .library import Library
from .pdftext import cached_analyze, kind_label, recommendation
from .pipeline import PACKAGE_ROOT, run_book_pipeline, validate_pdf, venv_python
from .tasks import Task, TaskManager
from .workspace import safe_dir_name, scan_books

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17173  # 冷门端口，避开 8000/8080/5000 等常用端口
WEBUI_DIR = Path(__file__).resolve().parent / "webui"
DEFAULT_ROOT = PACKAGE_ROOT
MAX_JSON_BODY = 8 * 1024 * 1024
# 单次上传的字节上限（可用 MEDICAL_RAG_MAX_UPLOAD 覆盖，单位字节）
MAX_UPLOAD_SIZE = int(os.environ.get("MEDICAL_RAG_MAX_UPLOAD") or 2 * 1024 * 1024 * 1024)
COPY_CHUNK = 1024 * 1024
PDF_CHUNK = 1024 * 1024


class UploadTooLarge(ValueError):
    """上传体积超过 MAX_UPLOAD_SIZE（与控制流里的普通 ValueError 区分状态码）。"""


def safe_file_name(name: str, fallback: str = "教材.pdf") -> str:
    """把上传文件名转成安全的单文件名，并保证 .pdf 后缀。"""
    name = Path(str(name)).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().strip(".")
    if not name:
        name = fallback
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return name[:120]


def book_lock_key(directory: Path | str) -> str:
    """教材目录的规范化锁键（Windows 下忽略大小写，路径分隔符统一）。

    同一本教材的处理 / 重建索引 / 导入共用一个键，避免多个任务同时写
    text_v3、processed_v3、SQLite 索引与缓存 PNG。
    """
    return os.path.normcase(str(Path(directory).resolve()))


def interpreter_info(root: Path) -> dict[str, Any]:
    ocr = os.environ.get("MEDICAL_RAG_OCR_PYTHON") or venv_python(root, "ocr") or sys.executable
    paddle = (
        os.environ.get("MEDICAL_RAG_PADDLE_PYTHON")
        or venv_python(root, "ocr312")
        or ocr
    )
    return {
        "python": os.environ.get("MEDICAL_RAG_PYTHON") or sys.executable,
        "ocr_python": ocr,
        "paddle_python": paddle,
        "venv_ocr": venv_python(root, "ocr") is not None,
        "venv_ocr312": venv_python(root, "ocr312") is not None,
    }


def render_page_png(app: "WebApp", book: dict[str, Any], page_no: int, width: int) -> Path:
    """把 PDF 某页渲染成 PNG（磁盘缓存，按 PDF mtime 失效）。"""
    pdf = Path(book["pdf"])
    if not pdf.exists():
        raise FileNotFoundError(f"PDF 不存在：{pdf}")
    page_no = max(1, int(page_no))
    width = max(320, min(int(width), 2600))
    out_dir = app.cache_dir / str(book["id"])
    out = out_dir / f"page-{page_no:04d}-w{width}.png"
    mtime = pdf.stat().st_mtime
    if out.exists() and out.stat().st_mtime >= mtime:
        return out
    with app.render_lock:
        if out.exists() and out.stat().st_mtime >= mtime:
            return out
        import fitz

        with fitz.open(pdf) as doc:
            if page_no > len(doc):
                raise IndexError(f"页码超出范围：{page_no} > {len(doc)}")
            page = doc[page_no - 1]
            scale = width / page.rect.width
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        temp = out.with_name(out.stem + ".tmp.png")
        pix.save(str(temp))
        os.replace(temp, out)
    return out


class WebApp:
    """站点状态：教材扫描缓存、任务管理器、渲染锁。"""

    def __init__(self, root: Path | str = DEFAULT_ROOT, db: Path | str | None = None):
        self.root = Path(root).expanduser().resolve()
        self.db = Path(db).expanduser().resolve() if db else self.root / ".medical_rag" / "library.sqlite3"
        self.cache_dir = self.root / ".medical_rag" / "web_cache"
        self.tasks = TaskManager()
        self.render_lock = threading.Lock()
        self._books_lock = threading.Lock()
        self._books_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])

    # ------------------------------------------------------------------ 书库
    def books(self, max_age: float = 1.5) -> list[dict[str, Any]]:
        with self._books_lock:
            now = time.time()
            if now - self._books_cache[0] < max_age:
                return self._books_cache[1]
            books = scan_books(self.root, self.db)
            self._books_cache = (now, books)
            return books

    def invalidate_books(self) -> None:
        with self._books_lock:
            self._books_cache = (0.0, [])

    def book(self, book_id: str) -> dict[str, Any] | None:
        return next((book for book in self.books() if book["id"] == book_id), None)

    def library(self) -> Library:
        return Library(self.db)

    def overview(self, port: int | None = None) -> dict[str, Any]:
        books = self.books()
        indexed = [book for book in books if book["status"]["index"]["complete"]]
        return {
            "root": str(self.root),
            "db": str(self.db),
            "port": port,
            "time": time.time(),
            "interpreters": interpreter_info(self.root),
            "books": books,
            "tasks": self.tasks.list(),
            "totals": {
                "books": len(books),
                "indexed": len(indexed),
                "chunks": sum(book["status"]["index"]["chunks"] for book in books),
            },
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "MedicalRAGWeb/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> WebApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    # ------------------------------------------------------------- 响应工具
    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store", extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_data(self, data: Any, status: int = 200) -> None:
        body = json.dumps({"ok": status < 400, "data": data}, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_conflict(self, task: Task, message: str) -> None:
        """409：这本教材已有任务在运行，同时把已有 task_id 返回给调用方。"""
        body = json.dumps(
            {
                "ok": False,
                "error": message,
                "data": {"task_id": task.id, "label": task.label, "status": task.status},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        # 请求体可能还没读（超大上传），关闭连接避免 HTTP/1.1 流水线错位
        self.close_connection = True
        self._send(409, body, "application/json; charset=utf-8")

    def _send_file(self, path: Path, content_type: str, cache: str) -> None:
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_error_json(500, f"读取文件失败：{exc}")
            return
        self._send(200, body, content_type, cache=cache)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_JSON_BODY:
            raise ValueError("请求体过大")
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    @staticmethod
    def _trim(data: Any) -> Any:
        """去掉内部字段，保持 API 输出干净。"""
        if isinstance(data, dict):
            return {key: Handler._trim(value) for key, value in data.items() if not key.startswith("_")}
        if isinstance(data, list):
            return [Handler._trim(item) for item in data]
        return data

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            query = urllib.parse.parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                return self._serve_static("index.html")
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")

            if path == "/api/overview":
                port = getattr(self.server, "server_port", None)
                return self._send_data(self._trim(self.app.overview(port)))
            if path == "/api/books":
                return self._send_data(self._trim(self.app.books()))
            if path == "/api/tasks":
                return self._send_data(self.app.tasks.list())

            parts = [segment for segment in path.split("/") if segment]
            if parts[:2] == ["api", "books"] and len(parts) >= 3:
                return self._book_get(parts[2], parts[3:], query)
            if parts[:2] == ["api", "tasks"] and len(parts) == 3:
                return self._task_get(parts[2], query)
            return self._send_error_json(404, "没有这个接口")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _serve_static(self, name: str) -> None:
        base = WEBUI_DIR.resolve()
        target = (base / name).resolve()
        if not target.is_file() or base not in target.parents:
            return self._send_error_json(404, "静态资源不存在")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix in (".html", ".css", ".js"):
            content_type = {"html": "text/html", "css": "text/css", "js": "text/javascript"}[target.suffix[1:]] + "; charset=utf-8"
        self._send_file(target, content_type, cache="no-cache")

    def _book_get(self, book_id: str, rest: list[str], query: dict[str, list[str]]) -> None:
        book = self.app.book(book_id)
        if book is None:
            return self._send_error_json(404, "没有这本教材")
        if not rest:
            data = dict(book)
            index = book["status"]["index"]
            if index.get("book_id"):
                library = self.app.library()
                try:
                    library_row = library.get_book(index["book_id"])
                    data["index_book"] = library_row
                    page_range = library.page_range(index["book_id"])
                    data["page_range"] = {"min": page_range[0], "max": page_range[1]} if page_range else None
                finally:
                    library.close()
            return self._send_data(self._trim(data))

        if not book.get("pdf"):
            return self._send_error_json(404, "这本教材没有 PDF，无法预览")
        try:
            if rest == ["cover.png"] or rest == ["cover"]:
                width = _int_arg(query, "w", 480)
                path = render_page_png(self.app, book, 1, width)
                return self._send_file(path, "image/png", cache="public, max-age=86400")
            if rest == ["pdf"]:
                return self._serve_pdf(Path(book["pdf"]))
            if len(rest) == 2 and rest[0] == "page":
                page_no = int(rest[1].split(".")[0])
                width = _int_arg(query, "w", 1500)
                path = render_page_png(self.app, book, page_no, width)
                return self._send_file(path, "image/png", cache="public, max-age=86400")
            if len(rest) == 2 and rest[0] == "text":
                page_no = int(rest[1].split(".")[0])
                return self._send_data(self._page_text(book, page_no))
        except (ValueError, IndexError) as exc:
            return self._send_error_json(400, str(exc))
        return self._send_error_json(404, "没有这个接口")

    def _page_text(self, book: dict[str, Any], page_no: int) -> dict[str, Any]:
        index = book["status"]["index"]
        if not index.get("book_id"):
            return {"page": page_no, "indexed": False, "chunks": []}
        library = self.app.library()
        try:
            chunks = library.chunks_for_page(index["book_id"], page_no)
        finally:
            library.close()
        return {
            "page": page_no,
            "indexed": True,
            "chunks": [
                {"chunk_id": item["chunk_id"], "section": item["section"], "text": item["text"]}
                for item in chunks
            ],
        }

    def _serve_pdf(self, pdf: Path) -> None:
        if not pdf.exists():
            return self._send_error_json(404, "PDF 文件不存在")
        size = pdf.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and size:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = int(match.group(2))
                if not match.group(1) and match.group(2):  # bytes=-N
                    start = max(0, size - int(match.group(2)))
                end = min(end, size - 1)
                if start > end:
                    return self._send(416, b"", "application/pdf", extra={"Content-Range": f"bytes */{size}"})
                status = 206
        self.send_response(status)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        filename = urllib.parse.quote(pdf.name)
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{filename}")
        self.end_headers()
        if self.command == "HEAD":
            return
        remaining = end - start + 1
        try:
            with pdf.open("rb") as stream:
                stream.seek(start)
                while remaining > 0:
                    chunk = stream.read(min(PDF_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _task_get(self, task_id: str, query: dict[str, list[str]]) -> None:
        since = _int_arg(query, "since", 0)
        snapshot = self.app.tasks.snapshot(task_id, since=since)
        if snapshot is None:
            return self._send_error_json(404, "没有这个任务")
        self._send_data(snapshot)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        try:
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            parts = [segment for segment in path.split("/") if segment]
            if path == "/api/import":
                return self._post_import()
            if path == "/api/import/begin":
                return self._post_import_begin()
            if path == "/api/inspect":
                return self._post_inspect()
            if path == "/api/process":
                return self._post_process()
            if path == "/api/reindex":
                return self._post_reindex()
            if path == "/api/search":
                return self._post_search()
            if parts[:2] == ["api", "tasks"] and len(parts) == 4 and parts[3] == "cancel":
                return self._post_task_cancel(parts[2])
            self._drain_body()
            return self._send_error_json(404, "没有这个接口")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _drain_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if 0 < length <= COPY_CHUNK:
            self.rfile.read(length)

    def _post_import(self) -> None:
        body = self._read_json()
        source = str(body.get("path") or "").strip()
        if not source:
            return self._send_error_json(400, "请填写 PDF 路径")
        title = str(body.get("title") or "").strip() or None
        path = Path(source).expanduser()
        if not path.exists():
            return self._send_error_json(400, f"文件不存在：{path}")
        if path.is_dir():
            return self._send_error_json(400, "请填写 PDF 文件路径，而不是目录")
        if path.suffix.lower() != ".pdf":
            return self._send_error_json(400, "目前只支持 PDF 文件")
        resolved = path.resolve()
        if self._pdf_conflict(title or resolved.stem, resolved.name):
            return self._send_error_json(400, "目标目录已存在其他 PDF，请换一个书名")

        from .pipeline import import_pdf  # 延迟导入，避免循环引用

        app = self.app
        book_name = title or resolved.stem
        task, created = app.tasks.create_unique(
            "import",
            f"导入《{book_name}》",
            lambda current: import_pdf(current, app.root, resolved, title=title),
            meta={"kind": "path", "path": str(resolved)},
            lock_key=book_lock_key(app.root / "res" / safe_dir_name(book_name)),
        )
        if not created:
            return self._send_conflict(task, f"《{book_name}》已有任务在运行，请等它结束")
        self._send_data({"task_id": task.id})

    def _pdf_conflict(self, title: str, filename: str) -> bool:
        """目标目录里已有其它 PDF 时不允许导入，避免一本教材两个主文件。"""
        book_dir = self.app.root / "res" / safe_dir_name(title)
        pdf_dir = book_dir / "PDF"
        if not pdf_dir.is_dir():
            return False
        return any(path.name != filename for path in pdf_dir.glob("*.pdf"))

    def _post_import_begin(self) -> None:
        body = self._read_json()
        filename = safe_file_name(str(body.get("filename") or ""))
        title = str(body.get("title") or "").strip() or Path(filename).stem
        if not filename:
            return self._send_error_json(400, "缺少文件名")
        # 先判运行中的任务：这是比“目录已有其它 PDF”更根本的拒绝理由
        lock_key = book_lock_key(self.app.root / "res" / safe_dir_name(title))
        running = self.app.tasks.active_for(lock_key)
        if running is not None:
            return self._send_conflict(running, f"《{title}》已有任务在运行，请等它结束")
        if self._pdf_conflict(title, filename):
            return self._send_error_json(400, "同名目录已存在其他 PDF，请换一个书名")
        task = self.app.tasks.create_manual(
            "import",
            f"导入《{title}》",
            # 注意用 book_key 而不是 lock_key：pending 状态不算已持锁，否则用户
            # 放弃上传后残留的任务会永久阻塞这本教材
            meta={"kind": "upload", "filename": filename, "title": title, "book_key": lock_key},
        )
        self._send_data(
            {
                "task_id": task.id,
                "upload_url": f"/api/import/upload/{task.id}",
                "book_dir": str(self.app.root / "res" / safe_dir_name(title)),
            }
        )

    def _post_inspect(self) -> None:
        """导入前检测 PDF 是否带文字层（供导入弹窗使用）。"""
        body = self._read_json()
        path_text = str(body.get("path") or "").strip()
        if not path_text:
            return self._send_error_json(400, "请填写 PDF 路径")
        pdf = Path(path_text).expanduser()
        if not pdf.exists():
            return self._send_error_json(400, f"文件不存在：{pdf}")
        if pdf.is_dir() or pdf.suffix.lower() != ".pdf":
            return self._send_error_json(400, "请填写 PDF 文件路径")
        info = cached_analyze(pdf, self.app.root / ".medical_rag" / "pdf_text")
        if info.get("error"):
            return self._send_error_json(400, str(info["error"]))
        text_kind = info.get("kind")
        self._send_data(
            {
                **info,
                "label": kind_label(text_kind),
                "recommendation": recommendation(info),
                "recommended_stages": ["text", "index"]
                if text_kind == "text"
                else ["ocr", "layout", "tables", "fix", "structure", "index"],
            }
        )

    def _post_process(self) -> None:
        body = self._read_json()
        book_id = str(body.get("book_id") or "")
        book = self.app.book(book_id)
        if book is None:
            return self._send_error_json(404, "没有这本教材")
        if not book.get("pdf"):
            return self._send_error_json(400, "这本教材没有 PDF，无法处理")
        stages = body.get("stages", "auto")
        if not isinstance(stages, (list, str)):
            return self._send_error_json(400, "stages 必须是数组或 auto")
        force = bool(body.get("force", False))
        dpi = int(body.get("dpi") or 300)
        device = str(body.get("device") or "gpu")
        reset_ocr = bool(body.get("reset_ocr", False))

        app = self.app
        stages_label = "自动" if stages == "auto" or stages is None else "/".join(stages)
        task, created = app.tasks.create_unique(
            "process",
            f"处理《{book['title']}》（{stages_label}）",
            lambda current: run_book_pipeline(
                current, app.root, book, stages=stages, force=force,
                dpi=dpi, device=device, reset_ocr=reset_ocr, db=app.db,
            ),
            meta={"book_id": book_id, "kind": "process", "stages": stages},
            lock_key=book_lock_key(book["dir"]),
        )
        if not created:
            return self._send_conflict(task, f"《{book['title']}》已有任务在运行，请等它结束")
        self._send_data({"task_id": task.id})

    def _post_reindex(self) -> None:
        body = self._read_json()
        book_id = str(body.get("book_id") or "")
        book = self.app.book(book_id)
        if book is None:
            return self._send_error_json(404, "没有这本教材")
        if not book["status"]["structure"]["complete"]:
            return self._send_error_json(400, "还没有结构化结果（processed_v3/structured），请先处理教材")
        app = self.app
        task, created = app.tasks.create_unique(
            "index",
            f"重建索引《{book['title']}》",
            lambda current: run_book_pipeline(current, app.root, book, stages=["index"], db=app.db),
            meta={"book_id": book_id, "kind": "reindex"},
            lock_key=book_lock_key(book["dir"]),
        )
        if not created:
            return self._send_conflict(task, f"《{book['title']}》已有任务在运行，请等它结束")
        self._send_data({"task_id": task.id})

    def _post_search(self) -> None:
        body = self._read_json()
        query = str(body.get("query") or "").strip()
        if not query:
            return self._send_error_json(400, "检索词不能为空")
        limit = max(1, min(int(body.get("limit") or 8), 20))
        book_id = body.get("book_id")
        scope = None
        library = self.app.library()
        try:
            if book_id not in (None, "", "all"):
                index_id = self._resolve_index_book_id(book_id, library)
                if index_id is None:
                    return self._send_error_json(400, "没有这本教材，或该教材尚未建立索引")
                row = library.get_book(index_id)
                scope = {"id": index_id, "title": row["title"] if row else str(book_id)}
                results = library.search(query, limit, book=index_id)
            else:
                results = library.search(query, limit)
        finally:
            library.close()
        self._send_data({"query": query, "results": results, "scope": scope})

    def _resolve_index_book_id(self, book_id: Any, library: Library) -> int | None:
        """兼容两种范围标识：网站书卡的 workspace id（hash）与索引库的整数 id。"""
        workspace = self.app.book(str(book_id))
        if workspace is not None:
            index_id = workspace["status"]["index"].get("book_id")
            if index_id is not None:
                return int(index_id)
        if str(book_id).isdigit():
            row = library.get_book(int(book_id))
            return int(row["id"]) if row else None
        return None

    def _post_task_cancel(self, task_id: str) -> None:
        self._drain_body()
        if not self.app.tasks.cancel(task_id):
            return self._send_error_json(400, "任务不存在或已经结束")
        self._send_data({"task_id": task_id, "cancelled": True})

    # ------------------------------------------------------------------- PUT
    def do_PUT(self) -> None:  # noqa: N802
        try:
            path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
            parts = [segment for segment in path.split("/") if segment]
            if parts[:3] == ["api", "import", "upload"] and len(parts) == 4:
                return self._put_upload(parts[3])
            self._drain_body()
            return self._send_error_json(404, "没有这个接口")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _put_upload(self, task_id: str) -> None:
        task = self.app.tasks.get(task_id)
        if task is None:
            return self._send_error_json(404, "没有这个上传任务")
        if task.status != "pending":
            self._drain_body()
            return self._send_error_json(409, "该上传任务已经开始或结束")
        # 先卡体积再占锁/读流：超大文件不进落盘路径，也不占用教材锁
        if self.headers.get("Transfer-Encoding", "").lower().strip() == "chunked":
            task.mark_error("不支持 chunked 传输编码")
            self.close_connection = True
            return self._send_error_json(411, "请提供 Content-Length（暂不支持 chunked 传输编码）")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._send_error_json(411, "缺少 Content-Length 或文件为空")
        if length > MAX_UPLOAD_SIZE:
            task.mark_error(f"文件 {length} 字节超过上限 {MAX_UPLOAD_SIZE} 字节")
            self.close_connection = True
            return self._send_error_json(
                413, f"文件超过上限（{MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB）"
            )
        # 真正开始写书库前才占锁，避免放弃上传后残留 pending 任务永久阻塞这本教材
        lock_key = str(task.meta.get("book_key") or "")
        if not self.app.tasks.acquire_lock(task, lock_key):
            return self._send_conflict(
                self.app.tasks.active_for(lock_key) or task,
                "这本教材已有任务在运行，请等它结束",
            )
        filename = str(task.meta.get("filename") or "教材.pdf")
        title = str(task.meta.get("title") or Path(filename).stem)
        book_dir = self.app.root / "res" / safe_dir_name(title)
        pdf_dir = book_dir / "PDF"
        other = [path for path in pdf_dir.glob("*.pdf") if path.name != filename] if pdf_dir.is_dir() else []
        if other:
            task.mark_error("目标目录已存在其他 PDF")
            return self._send_error_json(400, "同名目录已存在其他 PDF，请换一个书名")

        pdf_dir.mkdir(parents=True, exist_ok=True)
        target = pdf_dir / filename
        temp = pdf_dir / (filename + ".part")
        task.mark_running()
        task.log(f"接收上传：{filename}（{length / 1024 / 1024:.1f} MB）→ {target}")
        written = 0
        try:
            with temp.open("wb") as stream:
                while written < length:
                    if task.cancelled:
                        raise InterruptedError("已取消")
                    chunk = self.rfile.read(min(COPY_CHUNK, length - written))
                    if not chunk:
                        raise IOError("上传中断：客户端提前断开")
                    written += len(chunk)
                    # 累计上限（防御 chunked/长度头不一致的情况）
                    if written > MAX_UPLOAD_SIZE:
                        raise UploadTooLarge(
                            f"上传超过上限（{MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB）"
                        )
                    stream.write(chunk)
                    task.set_progress(written, length, f"{written / 1024 / 1024:.1f} / {length / 1024 / 1024:.1f} MB")
            # 字节收齐后先校验，再原子落盘：无效/截断的文件不能进书库
            validate_pdf(temp, label=f"上传的 {filename}")
            os.replace(temp, target)
        except InterruptedError:
            temp.unlink(missing_ok=True)
            task.mark_cancelled()
            return self._send_error_json(400, "上传已取消")
        except UploadTooLarge as exc:
            temp.unlink(missing_ok=True)
            task.mark_error(str(exc))
            self.close_connection = True
            return self._send_error_json(413, str(exc))
        except ValueError as exc:
            temp.unlink(missing_ok=True)
            task.mark_error(str(exc))
            return self._send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            temp.unlink(missing_ok=True)
            task.mark_error(str(exc))
            raise
        try:
            info = cached_analyze(target, self.app.root / ".medical_rag" / "pdf_text")
            task.log(
                f"PDF 检测：{kind_label(info.get('kind'))}，"
                f"{info.get('text_pages', 0)}/{info.get('pages', 0)} 页有文字，共 {info.get('chars', 0)} 字"
            )
            task.meta["pdf_kind"] = info.get("kind")
            task.meta["pdf_text_pages"] = info.get("text_pages")
            if info.get("kind") == "text":
                task.log("建议：直接运行「文字层解析 + 建立索引」，无需 OCR")
            else:
                task.log("建议：使用 OCR 流水线（OCR → 版面 → 表格 → 纪错 → 结构化 → 索引）")
        except Exception as exc:  # noqa: BLE001 - 已确认是有效 PDF，检测失败不影响上传结果
            task.log(f"警告：PDF 文字层检测失败（文件本身有效）：{exc}")
        task.mark_done(f"上传完成：{target}")
        self.app.invalidate_books()
        self._send_data({"task_id": task.id, "path": str(target), "book_dir": str(book_dir)})

    # ------------------------------------------------------------------ HEAD
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()


class MedicalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: WebApp, verbose: bool = False):
        super().__init__(address, Handler)
        self.app = app
        self.verbose = verbose


def _int_arg(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    root: Path | str = DEFAULT_ROOT,
    db: Path | str | None = None,
    verbose: bool = False,
    port_attempts: int = 20,
) -> MedicalHTTPServer:
    app = WebApp(root, db)
    last_error: OSError | None = None
    for offset in range(port_attempts):
        try:
            server = MedicalHTTPServer((host, port + offset), app, verbose=verbose)
            if offset:
                print(f"[信息] 端口 {port} 被占用，改用 {port + offset}", flush=True)
            return server
        except OSError as exc:
            last_error = exc
    raise OSError(f"无法绑定端口 {port}-{port + port_attempts - 1}：{last_error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="本地医学教材工作台网站")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址（默认 127.0.0.1）")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MEDICAL_RAG_WEB_PORT") or DEFAULT_PORT),
        help=f"监听端口（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument("--root", type=Path, default=None, help="项目根目录（默认自动定位）")
    parser.add_argument("--db", type=Path, default=None, help="library.sqlite3 路径")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--verbose", action="store_true", help="输出每个 HTTP 请求日志")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    server = create_server(args.host, args.port, args.root or DEFAULT_ROOT, args.db, verbose=args.verbose)
    host, port = server.server_address[0], server.server_address[1]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{display_host}:{port}/"
    info = interpreter_info(Path(args.root or DEFAULT_ROOT).resolve())

    print("=" * 62)
    print("  本地医学教材工作台")
    print(f"  地址：{url}")
    print(f"  项目：{server.app.root}")
    print(f"  索引库：{server.app.db}")
    print(f"  Python：{info['python']}")
    print(f"  OCR 环境：{info['ocr_python']}" + ("" if info["venv_ocr"] else "（未找到 .venv-ocr，使用主环境）"))
    print(f"  Paddle 环境：{info['paddle_python']}" + ("" if info["venv_ocr312"] else "（未找到 .venv-ocr312，使用主环境）"))
    print("=" * 62)
    print("  Ctrl+C 停止服务")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止…")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
