from __future__ import annotations

import contextlib
import http.client
import io
import json
import os
import socket
import sqlite3
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from unittest import mock

import fitz

from medical_rag import webapp as webapp_module
from medical_rag.library import Library
from medical_rag.webapp import MAX_FILE_NAME, create_server, safe_file_name
from medical_rag.workspace import WINDOWS_RESERVED_NAMES


def make_pdf(path: Path, pages: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Anatomy page {index + 1}", fontsize=14)
        page.insert_text((300, 800), str(index + 1), fontsize=10)
    doc.save(str(path))
    doc.close()


def make_text_pdf(path: Path, pages: int = 3) -> None:
    """带完整文字层的中文 PDF（每页约 260 字，含页码与章节标题）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for index in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 110), f"第{index}章 测试章节", fontname="china-s", fontsize=16)
        for row in range(8):
            page.insert_text(
                (72, 160 + row * 26),
                "组织学与胚胎学：上皮组织由密集排列的上皮细胞和少量细胞外基质组成。",
                fontname="china-s",
                fontsize=11,
            )
        page.insert_text((300, 800), str(index), fontsize=10)
    doc.save(str(path))
    doc.close()


class WebAppTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_web_test_"))
        self.book_dir = self.tmp / "res" / "测试教材"
        self.pdf = self.book_dir / "PDF" / "测试教材.pdf"
        make_pdf(self.pdf, pages=3)
        self.server = create_server("127.0.0.1", 0, root=self.tmp)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def request(self, path: str, method: str = "GET", payload=None, headers=None, raw: bytes | None = None):
        data = None
        request_headers = dict(headers or {})
        if raw is not None:
            data = raw
        elif payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(self.base + path, data=data, method=method, headers=request_headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            try:
                return error.code, dict(error.headers), error.read()
            finally:
                error.close()

    def json_data(self, path: str, method: str = "GET", payload=None):
        status, headers, body = self.request(path, method, payload)
        return status, json.loads(body)

    def wait_task(self, task_id: str, timeout: float = 60.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, payload = self.json_data(f"/api/tasks/{task_id}?since=0")
            task = payload["data"]
            if task["status"] not in ("pending", "running"):
                return task
            time.sleep(0.05)
        raise TimeoutError(task_id)

    def index_book(self, page: int = 1, text: str = "骨骼肌由肌腹和肌腱构成。") -> None:
        structured = self.book_dir / "processed_v3" / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        (structured / "01-肌学.md").write_text(
            f"# 第一章 肌学\n\n## 原书第 {page} 页\n\n{text}\n", encoding="utf-8"
        )
        library = Library(self.tmp / ".medical_rag" / "library.sqlite3")
        try:
            library.ingest_markdown_tree(self.book_dir / "processed_v3", "测试教材（索引）")
        finally:
            library.close()
        self.server.app.invalidate_books()

    def book_id(self) -> str:
        _, payload = self.json_data("/api/books")
        return payload["data"][0]["id"]


class StaticAndStatusTests(WebAppTestCase):
    def test_index_and_static_files(self) -> None:
        status, headers, body = self.request("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("医学教材工作台", body.decode("utf-8"))

        status, headers, body = self.request("/static/app.js")
        self.assertEqual(status, 200)
        self.assertIn("javascript", headers["Content-Type"])
        self.assertIn("pollTasks", body.decode("utf-8"))

        status, _, _ = self.request("/static/missing.js")
        self.assertEqual(status, 404)

    def test_overview_and_books_report_pdf_status(self) -> None:
        status, payload = self.json_data("/api/overview")
        self.assertEqual(status, 200)
        data = payload["data"]
        self.assertEqual(data["totals"]["books"], 1)
        self.assertIn("interpreters", data)

        status, payload = self.json_data("/api/books")
        book = payload["data"][0]
        self.assertEqual(book["title"], "测试教材")
        self.assertTrue(book["status"]["pdf"]["found"])
        self.assertEqual(book["status"]["pdf"]["pages"], 3)
        self.assertFalse(book["status"]["index"]["complete"])

    def test_detail_includes_page_range_when_indexed(self) -> None:
        self.index_book(page=1)
        status, payload = self.json_data(f"/api/books/{self.book_id()}")
        data = payload["data"]
        self.assertEqual(status, 200)
        self.assertEqual(data["page_range"], {"min": 1, "max": 1})
        self.assertEqual(data["status"]["index"]["chunks"], 1)


class PreviewTests(WebAppTestCase):
    def test_cover_and_page_are_png(self) -> None:
        book_id = self.book_id()
        status, headers, body = self.request(f"/api/books/{book_id}/cover.png?w=320")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/png")
        self.assertEqual(body[:8], b"\x89PNG\r\n\x1a\n")

        status, headers, body = self.request(f"/api/books/{book_id}/page/2.png?w=800")
        self.assertEqual(status, 200)
        self.assertEqual(body[:8], b"\x89PNG\r\n\x1a\n")

        status, _, _ = self.request(f"/api/books/{book_id}/page/99.png?w=800")
        self.assertEqual(status, 400)

    def test_pdf_range_request(self) -> None:
        book_id = self.book_id()
        size = self.pdf.stat().st_size
        status, headers, body = self.request(f"/api/books/{book_id}/pdf", headers={"Range": "bytes=0-99"})
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 100)
        self.assertEqual(headers["Content-Range"], f"bytes 0-99/{size}")
        self.assertEqual(headers["Accept-Ranges"], "bytes")

        status, headers, body = self.request(f"/api/books/{book_id}/pdf")
        self.assertEqual(status, 200)
        self.assertEqual(len(body), size)

    def test_page_text_returns_chunks(self) -> None:
        self.index_book(page=1, text="骨骼肌由肌腹和肌腱构成。")
        book_id = self.book_id()
        status, payload = self.json_data(f"/api/books/{book_id}/text/1")
        data = payload["data"]
        self.assertTrue(data["indexed"])
        self.assertEqual(len(data["chunks"]), 1)
        self.assertIn("骨骼肌", data["chunks"][0]["text"])
        self.assertTrue(data["chunks"][0]["chunk_id"])

        status, payload = self.json_data(f"/api/books/{book_id}/text/999")
        self.assertEqual(payload["data"]["chunks"], [])


class SearchTests(WebAppTestCase):
    def test_search_finds_indexed_evidence(self) -> None:
        self.index_book(page=1, text="骨骼肌由肌腹和肌腱构成。")
        status, payload = self.json_data("/api/search", "POST", {"query": "骨骼肌", "limit": 5})
        self.assertEqual(status, 200)
        results = payload["data"]["results"]
        self.assertTrue(results)
        self.assertEqual(results[0]["page"], 1)
        self.assertIn("骨骼肌", results[0]["text"])

    def test_search_rejects_empty_query(self) -> None:
        status, payload = self.json_data("/api/search", "POST", {"query": "  "})
        self.assertEqual(status, 400)
        self.assertFalse(payload["ok"])

    def test_search_scope_by_workspace_id(self) -> None:
        self.index_book(page=1, text="骨骼肌由肌腹和肌腱构成。")
        book_id = self.book_id()  # 网站书卡的 workspace id（hash）
        status, payload = self.json_data("/api/search", "POST", {"query": "骨骼肌", "book_id": book_id})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["scope"]["id"], 1)  # 解析为索引库整数 id
        self.assertTrue(payload["data"]["results"])

    def test_search_scope_accepts_index_id(self) -> None:
        self.index_book(page=1, text="骨骼肌由肌腹和肌腱构成。")
        status, payload = self.json_data("/api/search", "POST", {"query": "骨骼肌", "book_id": 1})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["scope"]["id"], 1)

    def test_search_scope_rejects_unknown_book(self) -> None:
        self.index_book(page=1, text="骨骼肌由肌腹和肌腱构成。")
        status, payload = self.json_data("/api/search", "POST", {"query": "骨骼肌", "book_id": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("没有这本教材", payload["error"])


class ImportTests(WebAppTestCase):
    def test_long_upload_filename_keeps_pdf_suffix(self) -> None:
        """超长文件名截断后必须仍是 .pdf（否则后续按后缀判断全失效）。"""
        long_name = "超长教材名" * 40 + ".pdf"
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": long_name, "title": "超长"}
        )
        self.assertEqual(status, 200)
        source = self.tmp / "长.pdf"
        make_pdf(source, pages=1)
        status, _, body = self.request(
            payload["data"]["upload_url"],
            "PUT",
            raw=source.read_bytes(),
            headers={"Content-Type": "application/pdf"},
        )
        self.assertEqual(status, 200, body)
        saved = list((self.tmp / "res" / "超长" / "PDF").glob("*.pdf"))
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0].name.endswith(".pdf"))
        self.assertLessEqual(len(saved[0].name), MAX_FILE_NAME)

    def test_path_import_creates_copy(self) -> None:
        source = self.tmp / "外部教材.pdf"
        make_pdf(source, pages=2)
        status, payload = self.json_data("/api/import", "POST", {"path": str(source), "title": "外部教材"})
        self.assertEqual(status, 200)
        task = self.wait_task(payload["data"]["task_id"])
        self.assertEqual(task["status"], "done")
        self.assertTrue((self.tmp / "res" / "外部教材" / "PDF" / "外部教材.pdf").exists())
        self.server.app.invalidate_books()
        _, books = self.json_data("/api/books")
        titles = [book["title"] for book in books["data"]]
        self.assertIn("外部教材", titles)

    def test_path_import_rejects_missing_file(self) -> None:
        status, payload = self.json_data("/api/import", "POST", {"path": str(self.tmp / "nope.pdf")})
        self.assertEqual(status, 400)

    def test_path_import_rejects_conflicting_pdf(self) -> None:
        """目标目录已有其他 PDF 时，路径导入同步返回 400，不创建任务。"""
        existing = self.tmp / "res" / "冲突书" / "PDF"
        existing.mkdir(parents=True)
        make_pdf(existing / "已有.pdf", pages=1)
        source = self.tmp / "另一本.pdf"
        make_pdf(source, pages=1)
        status, payload = self.json_data(
            "/api/import", "POST", {"path": str(source), "title": "冲突书"}
        )
        self.assertEqual(status, 400)
        self.assertIn("已存在", payload["error"])

    def test_path_import_ignores_legacy_copy_flag(self) -> None:
        """copy 参数已移除；旧客户端仍传 copy=false 时应按普通导入处理（始终复制）。"""
        source = self.tmp / "旧客户端.pdf"
        make_pdf(source, pages=1)
        status, payload = self.json_data(
            "/api/import", "POST", {"path": str(source), "title": "旧客户端教材", "copy": False}
        )
        self.assertEqual(status, 200)
        task = self.wait_task(payload["data"]["task_id"])
        self.assertEqual(task["status"], "done")
        self.assertTrue(
            (self.tmp / "res" / "旧客户端教材" / "PDF" / "旧客户端.pdf").exists()
        )

    def test_upload_over_limit_is_rejected(self) -> None:
        """超过 MAX_UPLOAD_SIZE 立即 413：不落盘、不留 .part、任务标 error。"""
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": "大.pdf", "title": "大教材"}
        )
        self.assertEqual(status, 200)
        begin = payload["data"]
        with mock.patch.object(webapp_module, "MAX_UPLOAD_SIZE", 1024):
            status, _, body = self.request(
                begin["upload_url"],
                "PUT",
                raw=b"x" * 4096,
                headers={"Content-Type": "application/pdf"},
            )
        self.assertEqual(status, 413, body)
        self.assertEqual(self.wait_task(begin["task_id"])["status"], "error")
        pdf_dir = self.tmp / "res" / "大教材" / "PDF"
        self.assertFalse((pdf_dir / "大.pdf").exists())
        self.assertEqual(list(pdf_dir.glob("*.part")), [])

    def test_upload_without_content_length_is_rejected(self) -> None:
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": "空.pdf", "title": "空教材"}
        )
        self.assertEqual(status, 200)
        status, _, body = self.request(
            payload["data"]["upload_url"],
            "PUT",
            raw=b"",
            headers={"Content-Type": "application/pdf"},
        )
        self.assertEqual(status, 411, body)

    def test_chunked_upload_is_rejected(self) -> None:
        """chunked 传输没有总体积上限，直接拒绝（要求 Content-Length）。"""
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": "分块.pdf", "title": "分块教材"}
        )
        self.assertEqual(status, 200)
        url = payload["data"]["upload_url"]
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(
                (
                    f"PUT {url} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{self.port}\r\n"
                    "Transfer-Encoding: chunked\r\n"
                    "Content-Type: application/pdf\r\n"
                    "\r\n"
                    "5\r\nhello\r\n0\r\n\r\n"
                ).encode("ascii")
            )
            response = sock.recv(8192).decode("utf-8", "replace")
        self.assertIn("411", response.split("\r\n")[0])

    def test_streamed_upload(self) -> None:
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": "上传教材.pdf", "title": "上传教材"}
        )
        self.assertEqual(status, 200)
        begin = payload["data"]
        source = self.tmp / "上传源.pdf"
        make_pdf(source, pages=2)
        raw = source.read_bytes()
        status, _, body = self.request(
            begin["upload_url"], "PUT", raw=raw, headers={"Content-Type": "application/pdf"}
        )
        self.assertEqual(status, 200, body)
        task = self.wait_task(begin["task_id"])
        self.assertEqual(task["status"], "done")
        self.assertTrue((self.tmp / "res" / "上传教材" / "PDF" / "上传教材.pdf").exists())

    def test_upload_rejects_conflicting_pdf(self) -> None:
        source = self.tmp / "冲突源.pdf"
        make_pdf(source, pages=1)
        first = self.json_data("/api/import/begin", "POST", {"filename": "a.pdf", "title": "冲突教材"})
        status, _, body = self.request(
            first[1]["data"]["upload_url"],
            "PUT",
            raw=source.read_bytes(),
            headers={"Content-Type": "application/pdf"},
        )
        self.assertEqual(status, 200, body)
        status, payload = self.json_data("/api/import/begin", "POST", {"filename": "b.pdf", "title": "冲突教材"})
        self.assertEqual(status, 400)
        self.assertIn("已存在", payload["error"])

    def test_streamed_upload_rejects_invalid_pdf(self) -> None:
        """无效 PDF 上传必须失败：不能落盘、不能留 .part、任务必须 error。"""
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": "坏的.pdf", "title": "坏教材"}
        )
        self.assertEqual(status, 200)
        begin = payload["data"]
        status, _, body = self.request(
            begin["upload_url"], "PUT", raw=b"not a pdf", headers={"Content-Type": "application/pdf"}
        )
        self.assertEqual(status, 400, body)
        task = self.wait_task(begin["task_id"])
        self.assertEqual(task["status"], "error")
        pdf_dir = self.tmp / "res" / "坏教材" / "PDF"
        self.assertFalse((pdf_dir / "坏的.pdf").exists())
        self.assertEqual(list(pdf_dir.glob("*.part")), [])


class BookConcurrencyTests(WebAppTestCase):
    """同一本教材的所有写操作（处理/重建索引/导入）必须互斥。"""

    def setUp(self) -> None:
        super().setUp()
        self.release = threading.Event()
        self.started = threading.Event()

        def slow_pipeline(current, *args, **kwargs):
            self.started.set()
            self.release.wait(timeout=30)

        patcher = mock.patch.object(webapp_module, "run_book_pipeline", slow_pipeline)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.release.set)

    def _start_process(self, book_id: str) -> str:
        status, payload = self.json_data(
            "/api/process", "POST", {"book_id": book_id, "stages": ["index"]}
        )
        self.assertEqual(status, 200, payload)
        task_id = payload["data"]["task_id"]
        self.assertTrue(self.started.wait(timeout=10), "任务未开始")
        return task_id

    def test_second_process_on_same_book_is_rejected(self) -> None:
        book_id = self.book_id()
        first_id = self._start_process(book_id)

        status, payload = self.json_data(
            "/api/process", "POST", {"book_id": book_id, "stages": ["index"]}
        )
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["data"]["task_id"], first_id)
        self.assertIn("已有任务在运行", payload["error"])

        self.release.set()
        self.assertEqual(self.wait_task(first_id)["status"], "done")

    def test_reindex_is_rejected_while_process_runs(self) -> None:
        processed = self.book_dir / "processed_v3"
        structured = processed / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        (structured / "01-肌学.md").write_text(
            "# 第一章 肌学\n\n## 原书第 1 页\n\n骨骼肌由肌腹和肌腱构成。\n", encoding="utf-8"
        )
        # _post_reindex 要求 structure.complete，而它需要 quality.json 存在
        (processed / "quality.json").write_text(
            json.dumps({"pages": 3, "page_offset": 0, "chapters": []}, ensure_ascii=False),
            encoding="utf-8",
        )
        self.server.app.invalidate_books()
        book_id = self.book_id()
        first_id = self._start_process(book_id)

        status, payload = self.json_data("/api/reindex", "POST", {"book_id": book_id})
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["data"]["task_id"], first_id)

        self.release.set()
        self.wait_task(first_id)

    def test_different_books_are_not_blocked(self) -> None:
        other = self.tmp / "res" / "另一本教材" / "PDF"
        make_pdf(other / "另一本教材.pdf", pages=1)
        self.server.app.invalidate_books()
        _, payload = self.json_data("/api/books")
        ids = [book["id"] for book in payload["data"]]
        self.assertGreaterEqual(len(ids), 2)

        first_id = self._start_process(ids[0])
        second_id = self._start_process(ids[1])
        self.assertNotEqual(first_id, second_id)
        self.release.set()

    def test_abandoned_upload_does_not_block_book(self) -> None:
        """只登记上传、不实际上传：pending 任务不应占用教材锁。"""
        status, _ = self.json_data(
            "/api/import/begin",
            "POST",
            {"filename": "测试教材.pdf", "title": "测试教材"},
        )
        self.assertEqual(status, 200)
        self._start_process(self.book_id())  # 不应被 pending 上传任务阻塞

    def test_upload_is_rejected_while_process_runs(self) -> None:
        book_id = self.book_id()
        self._start_process(book_id)
        # 上传登记：目标目录与正在处理的教材目录相同
        status, payload = self.json_data(
            "/api/import/begin", "POST", {"filename": "新.pdf", "title": "测试教材"}
        )
        self.assertEqual(status, 409, payload)
        self.assertIn("已有任务在运行", payload["error"])
        self.release.set()


class HealthTests(WebAppTestCase):
    def test_health_returns_environment_report(self) -> None:
        status, payload = self.json_data("/api/health")
        self.assertIn(status, (200, 503))
        data = payload["data"]
        self.assertIn("checks", data)
        self.assertIn("interpreters", data)
        self.assertIn("database", data)
        self.assertTrue(any(item["name"] == "Python 版本" for item in data["checks"]))

    def test_health_does_not_create_database(self) -> None:
        """健康检查是只读的，不能顺手把索引库建出来。"""
        db = self.tmp / ".medical_rag" / "library.sqlite3"
        self.assertFalse(db.exists())
        self.json_data("/api/health")
        self.assertFalse(db.exists())


class ProcessTests(WebAppTestCase):
    def test_process_validates_book(self) -> None:
        status, payload = self.json_data("/api/process", "POST", {"book_id": "missing"})
        self.assertEqual(status, 404)

    def test_process_starts_task_with_stages(self) -> None:
        book_id = self.book_id()
        status, payload = self.json_data("/api/process", "POST", {"book_id": book_id, "stages": ["index"]})
        self.assertEqual(status, 200)
        task_id = payload["data"]["task_id"]
        # 没有结构化结果时 index 阶段应快速失败，并把原因反馈到任务里
        task = self.wait_task(task_id)
        self.assertEqual(task["status"], "error")
        self.assertIn("processed_v3/structured", task["error"])

    def test_reindex_requires_structure(self) -> None:
        status, payload = self.json_data("/api/reindex", "POST", {"book_id": self.book_id()})
        self.assertEqual(status, 400)

    def test_cancel_unknown_task(self) -> None:
        status, payload = self.json_data("/api/tasks/nope/cancel", "POST", {})
        self.assertEqual(status, 400)


class SafeFileNameTests(unittest.TestCase):
    """上传文件名净化：截断保留后缀、避开 Windows 保留名、去路径与非法字符。"""

    def test_truncation_keeps_pdf_suffix(self) -> None:
        name = safe_file_name("骨" * 300 + ".pdf")
        self.assertTrue(name.endswith(".pdf"))
        self.assertEqual(len(name), MAX_FILE_NAME)

    def test_truncation_of_name_without_suffix(self) -> None:
        name = safe_file_name("长" * 300)
        self.assertTrue(name.endswith(".pdf"))
        self.assertLessEqual(len(name), MAX_FILE_NAME)

    def test_windows_reserved_names_are_escaped(self) -> None:
        for raw in ("CON.pdf", "nul.PDF", "COM1.pdf", "LPT9.pdf", "aux"):
            with self.subTest(raw=raw):
                name = safe_file_name(raw)
                self.assertTrue(name.endswith(".pdf"))
                self.assertNotIn(name[: -len(".pdf")].upper(), WINDOWS_RESERVED_NAMES)

    def test_strips_paths_and_illegal_characters(self) -> None:
        self.assertEqual(safe_file_name("../../etc/骨:学.pdf"), "骨_学.pdf")
        self.assertEqual(safe_file_name("C:\\windows\\系统.pdf"), "系统.pdf")

    def test_empty_and_dot_names_fall_back(self) -> None:
        for raw in ("", "...", "../", "   "):
            with self.subTest(raw=raw):
                self.assertEqual(safe_file_name(raw), "教材.pdf")

    def test_trailing_dots_and_spaces_removed(self) -> None:
        self.assertEqual(safe_file_name("骨学 ...pdf"), "骨学.pdf")
        self.assertEqual(safe_file_name("骨学 .pdf"), "骨学.pdf")


class AuthTests(unittest.TestCase):
    """非本机监听时的访问令牌：header / ?token= / Cookie 三条路径。"""

    TOKEN = "s3cret-token"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_auth_"))
        self.server = create_server("127.0.0.1", 0, root=self.tmp, token=self.TOKEN)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def request(self, path: str, method: str = "GET", payload=None, headers=None):
        data = None
        request_headers = dict(headers or {})
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers=request_headers
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            try:
                return error.code, dict(error.headers), error.read()
            finally:
                error.close()

    def test_api_requires_token(self) -> None:
        status, _, _ = self.request("/api/books")
        self.assertEqual(status, 401)

    def test_header_token_is_accepted(self) -> None:
        status, _, body = self.request("/api/books", headers={"X-Auth-Token": self.TOKEN})
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["ok"])

    def test_query_token_is_accepted_and_sets_cookie(self) -> None:
        """封面/预览图用 <img src> 加载，无法带自定义头，所以必须支持 Cookie。"""
        status, headers, _ = self.request(f"/?token={self.TOKEN}")
        self.assertEqual(status, 200)
        cookie = headers.get("Set-Cookie", "")
        self.assertIn("medrag_token=", cookie)
        self.assertIn("SameSite=Strict", cookie)
        self.assertIn("HttpOnly", cookie)

    def test_cookie_token_is_accepted(self) -> None:
        status, _, _ = self.request(
            "/api/overview", headers={"Cookie": f"medrag_token={self.TOKEN}"}
        )
        self.assertEqual(status, 200)

    def test_wrong_token_is_rejected(self) -> None:
        status, _, _ = self.request("/api/books", headers={"X-Auth-Token": "nope"})
        self.assertEqual(status, 401)

    def test_image_endpoint_requires_token(self) -> None:
        status, _, _ = self.request("/api/books/whatever/cover.png")
        self.assertEqual(status, 401)

    def test_mutating_endpoint_requires_token(self) -> None:
        status, _, _ = self.request("/api/process", "POST", payload={"book_id": "x"})
        self.assertEqual(status, 401)

    def test_static_assets_need_no_token(self) -> None:
        status, _, _ = self.request("/static/app.js")
        self.assertEqual(status, 200)

    def test_health_requires_token(self) -> None:
        status, _, _ = self.request("/api/health")
        self.assertEqual(status, 401)


class RemotePathGuardTests(unittest.TestCase):
    def test_remote_mode_rejects_server_local_import_and_inspect(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_remote_guard_"))
        server = create_server("0.0.0.0", 0, root=tmp, token="secret")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            def request(path: str, payload: dict) -> int:
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    base + path,
                    data=body,
                    method="POST",
                    headers={"Content-Type": "application/json", "X-Auth-Token": "secret"},
                )
                try:
                    with urllib.request.urlopen(req, timeout=5) as response:
                        return response.status
                except urllib.error.HTTPError as error:
                    return error.code

            self.assertEqual(request("/api/import", {"path": str(tmp / "secret.pdf")}), 403)
            self.assertEqual(request("/api/inspect", {"path": str(tmp / "secret.pdf")}), 403)
        finally:
            server.shutdown()
            server.server_close()


class LoopbackGuardTests(unittest.TestCase):
    def test_is_loopback(self) -> None:
        for host in ("127.0.0.1", "::1", "localhost", "localhost.", "127.0.0.5"):
            self.assertTrue(webapp_module.is_loopback(host), host)
        for host in ("0.0.0.0", "::", "192.168.1.10", "example.com"):
            self.assertFalse(webapp_module.is_loopback(host), host)

    def test_main_refuses_remote_host_without_allow_remote(self) -> None:
        """默认禁止监听非本机地址，防止局域网裸奔。"""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                webapp_module.main(["--host", "0.0.0.0", "--no-browser"])


class StyleGuardTests(unittest.TestCase):
    def test_hidden_attribute_rule_present(self) -> None:
        """作者样式的 display（grid/flex）会覆盖 UA 的 [hidden]，必须保留全局兜底规则。"""
        import medical_rag

        css = (Path(medical_rag.__file__).resolve().parent / "webui" / "style.css").read_text(encoding="utf-8")
        self.assertRegex(css, r"\[hidden\]\s*\{\s*display:\s*none\s*!important")


class TextLayerWorkflowTests(WebAppTestCase):
    def test_inspect_detects_text_layer(self) -> None:
        pdf = self.tmp / "原书.pdf"
        make_text_pdf(pdf)
        status, payload = self.json_data("/api/inspect", "POST", {"path": str(pdf)})
        self.assertEqual(status, 200)
        data = payload["data"]
        self.assertEqual(data["kind"], "text")
        self.assertEqual(data["recommended_stages"], ["text", "index"])
        self.assertIn("文字层", data["recommendation"])

    def test_inspect_reports_scan_pdf(self) -> None:
        pdf = self.tmp / "图片型.pdf"
        doc = fitz.open()
        for _ in range(2):
            page = doc.new_page(width=595, height=842)
            page.draw_rect(fitz.Rect(60, 60, 535, 780), color=(0, 0, 0), width=1)
        doc.save(str(pdf))
        doc.close()
        status, payload = self.json_data("/api/inspect", "POST", {"path": str(pdf)})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["kind"], "scan")
        self.assertIn("OCR", payload["data"]["recommendation"])
        self.assertEqual(payload["data"]["recommended_stages"][0], "ocr")

    def test_import_and_auto_process_text_book(self) -> None:
        pdf = self.tmp / "组胚原书.pdf"
        make_text_pdf(pdf, pages=3)
        status, payload = self.json_data("/api/import", "POST", {"path": str(pdf), "title": "文字层教材"})
        self.assertEqual(status, 200)
        task = self.wait_task(payload["data"]["task_id"])
        self.assertEqual(task["status"], "done")
        self.assertTrue(any("PDF 检测" in line for line in task["logs"]))

        self.server.app.invalidate_books()
        _, payload = self.json_data("/api/books")
        book = next(item for item in payload["data"] if item["title"] == "文字层教材")
        self.assertEqual(book["workflow"], "text")
        self.assertEqual(book["status"]["text"]["kind"], "text")

        # auto 模式应选择文字层工作流（text → index）而不跑 OCR
        status, payload = self.json_data("/api/process", "POST", {"book_id": book["id"], "stages": "auto"})
        self.assertEqual(status, 200)
        task = self.wait_task(payload["data"]["task_id"])
        self.assertEqual(task["status"], "done", task.get("error"))
        self.assertEqual([stage["key"] for stage in task["stages"]], ["text", "index"])

        self.server.app.invalidate_books()
        _, payload = self.json_data(f"/api/books/{book['id']}")
        detail = payload["data"]
        self.assertTrue(detail["status"]["structure"]["complete"])
        self.assertGreaterEqual(len(detail["quality"]["chapters"]), 1)
        self.assertGreater(detail["status"]["index"]["chunks"], 0)

        _, payload = self.json_data(f"/api/books/{book['id']}/text/1")
        self.assertTrue(payload["data"]["chunks"])


class HostGuardTests(WebAppTestCase):
    """Host 头只认本机/绑定地址：挡住 DNS rebinding（恶意域名解析到 127.0.0.1）。"""

    def test_forged_host_header_is_rejected(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(b"GET /api/books HTTP/1.1\r\nHost: evil.example.com\r\n\r\n")
            response = sock.recv(8192).decode("utf-8", "replace")
        self.assertIn("403", response.split("\r\n")[0])

    def test_loopback_host_header_is_accepted(self) -> None:
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(
                f"GET /api/books HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n\r\n".encode("ascii")
            )
            response = sock.recv(8192).decode("utf-8", "replace")
        self.assertIn("200", response.split("\r\n")[0])


class CrossSiteGuardTests(WebAppTestCase):
    """变更类接口：有 Origin 时必须同源，且只接受 application/json。

    注意宽松点：curl / 现有测试不带 Origin，必须照旧放行。
    """

    def test_no_origin_curl_style_post_still_works(self) -> None:
        status, payload = self.json_data("/api/search", "POST", {"query": "骨骼肌"})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])

    def test_cross_origin_post_is_rejected(self) -> None:
        status, _, body = self.request(
            "/api/search",
            "POST",
            payload={"query": "骨骼肌"},
            headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(status, 403, body)
        self.assertIn("跨站", json.loads(body)["error"])

    def test_sec_fetch_cross_site_post_is_rejected(self) -> None:
        status, _, body = self.request(
            "/api/search",
            "POST",
            payload={"query": "骨骼肌"},
            headers={"Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(status, 403, body)

    def test_same_origin_post_is_accepted(self) -> None:
        status, _, body = self.request(
            "/api/search",
            "POST",
            payload={"query": "骨骼肌"},
            headers={"Origin": f"http://127.0.0.1:{self.port}"},
        )
        self.assertEqual(status, 200, body)

    def test_text_plain_post_is_rejected(self) -> None:
        """text/plain 属 CORS 简单请求，浏览器不发预检，必须靠 Content-Type 挡。"""
        status, _, body = self.request(
            "/api/search",
            "POST",
            raw=json.dumps({"query": "骨骼肌"}).encode("utf-8"),
            headers={"Content-Type": "text/plain"},
        )
        self.assertEqual(status, 415, body)


class PayloadStatusTests(WebAppTestCase):
    """客户端错误要可区分：超限 413、坏 JSON / 坏数值 400，不能一律 500。"""

    def test_oversized_json_body_is_413(self) -> None:
        with mock.patch.object(webapp_module, "MAX_JSON_BODY", 8):
            status, _, body = self.request("/api/search", "POST", payload={"query": "骨骼肌"})
        self.assertEqual(status, 413, body)
        self.assertIn("请求体过大", json.loads(body)["error"])

    def test_malformed_json_is_400(self) -> None:
        status, _, body = self.request(
            "/api/search", "POST", raw=b"{not json", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(status, 400, body)

    def test_non_object_json_is_400(self) -> None:
        status, _, body = self.request(
            "/api/search", "POST", raw=b"[1,2]", headers={"Content-Type": "application/json"}
        )
        self.assertEqual(status, 400, body)

    def test_non_numeric_limit_falls_back_to_default(self) -> None:
        """{"limit":"abc"} 不能变成 500：回落默认值，照常检索。"""
        self.index_book(page=1, text="骨骼肌由肌腹和肌腱构成。")
        status, payload = self.json_data("/api/search", "POST", {"query": "骨骼肌", "limit": "abc"})
        self.assertEqual(status, 200)
        self.assertTrue(payload["data"]["results"])


class NonAsciiTokenTests(unittest.TestCase):
    """中文口令：hmac.compare_digest 的 str 版只接受 ASCII，必须编码成 bytes 再比较。"""

    TOKEN = "教材口令"

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_auth_cn_"))
        self.server = create_server("127.0.0.1", 0, root=self.tmp, token=self.TOKEN)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def request(self, path: str, headers=None):
        request = urllib.request.Request(self.base + path, method="GET", headers=dict(headers or {}))
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as error:
            try:
                return error.code
            finally:
                error.close()

    def test_query_token_with_chinese_is_accepted(self) -> None:
        quoted = urllib.parse.quote(self.TOKEN)
        self.assertEqual(self.request(f"/api/books?token={quoted}"), 200)

    def test_cookie_token_with_chinese_is_accepted(self) -> None:
        quoted = urllib.parse.quote(self.TOKEN)
        self.assertEqual(
            self.request("/api/overview", headers={"Cookie": f"medrag_token={quoted}"}), 200
        )

    def test_missing_or_wrong_chinese_token_is_401(self) -> None:
        self.assertEqual(self.request("/api/books"), 401)
        wrong = urllib.parse.quote("教材口令错")
        self.assertEqual(self.request(f"/api/books?token={wrong}"), 401)


class PreviewCacheTests(WebAppTestCase):
    """预览图宽度必须量化到 900/1200/1600/2000/2600 五档，缓存按 mtime 淘汰。"""

    def test_width_is_quantised_to_bucket(self) -> None:
        book_id = self.book_id()
        for width in (1100, 1150, 1200):
            status, headers, body = self.request(f"/api/books/{book_id}/page/2.png?w={width}")
            self.assertEqual(status, 200)
            self.assertEqual(headers["Content-Type"], "image/png")
            self.assertEqual(body[:8], b"\x89PNG\r\n\x1a\n")
        cache = self.tmp / ".medical_rag" / "web_cache"
        self.assertEqual(
            [path.name for path in cache.rglob("page-0002-*.png")],
            ["page-0002-w1200.png"],
        )

    def test_bucket_width_is_monotonic(self) -> None:
        self.assertEqual(webapp_module.RENDER_WIDTH_BUCKETS, (900, 1200, 1600, 2000, 2600))
        self.assertEqual(webapp_module.bucket_width(1), 900)
        self.assertEqual(webapp_module.bucket_width(900), 900)
        self.assertEqual(webapp_module.bucket_width(901), 1200)
        self.assertEqual(webapp_module.bucket_width(1201), 1600)
        self.assertEqual(webapp_module.bucket_width(99999), 2600)
        self.assertEqual(webapp_module.bucket_width("bad"), 900)

    def test_prune_cache_drops_oldest_until_under_limit(self) -> None:
        cache = self.tmp / ".medical_rag" / "web_cache"
        cache.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(4):
            path = cache / f"fake-{index}.png"
            path.write_bytes(b"x" * 1000)
            os.utime(path, (1000 + index, 1000 + index))
            paths.append(path)
        removed = webapp_module._prune_cache(self.server.app, limit_bytes=2500)
        self.assertEqual(removed, 2)
        self.assertFalse(paths[0].exists())
        self.assertTrue(paths[-1].exists())
        self.assertLessEqual(sum(path.stat().st_size for path in paths if path.exists()), 2500)


class ResidentLibraryTests(WebAppTestCase):
    """常驻 Library：同一条 keep-alive 连接上的请求不能每次重新开关库。"""

    def test_library_is_reused_within_thread(self) -> None:
        app = self.server.app
        first = app.library()
        self.assertIs(first, app.library())
        app.invalidate_library()
        second = app.library()
        self.assertIsNot(first, second)
        app.close_libraries()

    def test_connection_close_releases_thread_library(self) -> None:
        """Connection: close 的客户端每次请求换线程：连接结束必须回收常驻连接。"""
        self.index_book(page=1)
        book_id = self.book_id()
        app = self.server.app
        before = len(list(app._libraries))
        self.request(f"/api/books/{book_id}/text/1")
        deadline = time.time() + 5.0
        while time.time() < deadline and len(list(app._libraries)) > before:
            time.sleep(0.02)  # 服务端 finish() 与客户端读到响应之间有一小段竞态
        self.assertEqual(len(list(app._libraries)), before, "连接关闭后不应残留常驻连接")

    def test_close_libraries_is_honest_about_cross_thread_connections(self) -> None:
        """F3：sqlite 连接只能由 owner 线程关闭，跨线程收尾必须干账而不是静默吞错。

        以前 ``close_libraries()`` 用 ``except Exception: pass`` 把
        ``ProgrammingError`` 吞掉：``_libraries`` 账清零了、连接还活着，收尾看起来
        成功其实没有。现在别的线程的连接只登记 ``request_close()``，由 owner 线程在
        下一次使用时真正关闭；失败一律记进 ``_close_errors``。
        """
        app = self.server.app
        holder: dict[str, Library] = {}
        observed: dict[str, object] = {}
        ready = threading.Event()
        go = threading.Event()

        def worker() -> None:
            library = app.library()  # owner 线程 = 本线程
            holder["library"] = library
            ready.set()
            go.wait(timeout=10)
            try:
                library.search("上皮组织", 1)
            except Exception as exc:  # noqa: BLE001 - 关库后必须明确失败
                observed["search_error"] = exc
            else:
                observed["search_error"] = None
            try:
                library.cx.execute("SELECT 1")
            except sqlite3.ProgrammingError as exc:
                observed["cx_error"] = exc
            else:
                observed["cx_error"] = None

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=10), "owner 线程没建起常驻连接")
        foreign = holder["library"]
        self.assertEqual(foreign.owner_thread, thread.ident)
        self.assertIn(foreign, list(app._libraries))

        app.close_libraries()  # 主线程收尾：不能真的关别的线程的连接

        self.assertEqual(list(app._libraries), [], "账本应清零")
        self.assertIn(foreign, list(app._pending_close), "跨线程连接必须登记待关，不能当没发生")
        self.assertEqual(app._close_errors, [], "不该产生静默吞掉的关闭失败")

        go.set()
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        error = observed.get("search_error")
        self.assertIsNotNone(error, "关库后 owner 线程里 search 竟然还成功")
        self.assertIn("closed", str(error).lower())
        self.assertIsInstance(observed.get("cx_error"), sqlite3.ProgrammingError)

        # 真正的关闭在 owner 线程里发生了：再收一次账，待关清单必须收敛到空
        app.close_libraries()
        self.assertEqual(list(app._pending_close), [], "owner 关闭后待关账目应收敛")
        self.assertEqual(app._close_errors, [])

    def test_cross_thread_close_is_recorded_not_swallowed(self) -> None:
        """``_close_library`` 在非 owner 线程里失败时必须记账，再交给 owner 线程真关。"""
        app = self.server.app
        holder: dict[str, Library] = {}
        ready = threading.Event()
        release = threading.Event()

        def worker() -> None:
            holder["library"] = app.library()
            ready.set()
            release.wait(timeout=10)
            app.release_thread_library()  # owner 线程的正常收尾路径

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=10), "owner 线程没建起常驻连接")
        foreign = holder["library"]

        app._close_library(foreign)  # 故意在错误线程收尾
        self.assertEqual(len(app._close_errors), 1, "跨线程关闭失败被静默吞掉了")
        self.assertIn("ProgrammingError", app._close_errors[0])
        self.assertIn(foreign, list(app._pending_close))
        self.assertFalse(foreign.closed)

        release.set()
        thread.join(timeout=10)
        self.assertTrue(foreign.closed, "owner 线程收尾时应真正关闭连接")
        self.assertNotIn(foreign, list(app._pending_close))
        self.assertNotIn(foreign, list(app._libraries))

    def test_keepalive_request_is_not_blocked_by_writer(self) -> None:
        """另一条连接持写事务时，预览文本仍要立刻 200（不是 busy_timeout 30s 后 500）。"""
        self.index_book(page=1)
        book_id = self.book_id()
        db = self.tmp / ".medical_rag" / "library.sqlite3"
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        self.addCleanup(conn.close)
        conn.request("GET", "/api/books")
        first = conn.getresponse()
        first.read()  # 读完才能复用这条 keep-alive 连接
        self.assertEqual(first.status, 200)  # 先把常驻连接建起来
        writer = sqlite3.connect(db, timeout=1.0)
        writer.execute("BEGIN IMMEDIATE")
        try:
            started = time.monotonic()
            conn.request("GET", f"/api/books/{book_id}/text/1")
            response = conn.getresponse()
            body = response.read()
            elapsed = time.monotonic() - started
        finally:
            writer.rollback()
            writer.close()
        self.assertEqual(response.status, 200, body)
        self.assertLess(elapsed, 5.0)
        self.assertTrue(json.loads(body)["data"]["indexed"])


class PdfRangeSuffixTests(WebAppTestCase):
    """Range 后缀形式 bytes=-N（PDF 阅读器读尾页/尾部交叉引用表要用）。"""

    def test_suffix_range_returns_tail_bytes(self) -> None:
        book_id = self.book_id()
        size = self.pdf.stat().st_size
        status, headers, body = self.request(
            f"/api/books/{book_id}/pdf", headers={"Range": "bytes=-100"}
        )
        self.assertEqual(status, 206)
        self.assertEqual(len(body), 100)
        self.assertEqual(headers["Content-Range"], f"bytes {size - 100}-{size - 1}/{size}")

    def test_invalid_suffix_and_start_ranges_are_416(self) -> None:
        book_id = self.book_id()
        for header in ("bytes=-0", "bytes=99999999-", "bytes=99999999-100000000"):
            with self.subTest(header=header):
                status, headers, _ = self.request(
                    f"/api/books/{book_id}/pdf", headers={"Range": header}
                )
                self.assertEqual(status, 416)
                size = self.pdf.stat().st_size
                self.assertEqual(headers.get("Content-Range"), f"bytes */{size}")


if __name__ == "__main__":
    unittest.main()
