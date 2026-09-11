from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import fitz

from medical_rag import webapp as webapp_module
from medical_rag.library import Library
from medical_rag.webapp import create_server


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


if __name__ == "__main__":
    unittest.main()
