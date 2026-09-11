"""审计清单里剩余的失败路径与并发场景回归测试。

覆盖：
  1. 0 页 PDF（无效文件的一种，fitz 能打开但没有任何页面）
  2. 上传中断（声明长度远大于实发字节后断连）
  3. 上传过程中取消
  4. 两个**真实进程**同时重建同一本教材的索引
  5. 全链路：源 PDF 变短后重建结构化 → 旧章节文件清理 → 老内容不再被检索到
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import fitz

from medical_rag import pdftext, pipeline
from medical_rag.library import Library
from medical_rag.tasks import TaskManager
from medical_rag.webapp import create_server

ROOT = Path(__file__).resolve().parent.parent


def make_zero_page_pdf(path: Path) -> None:
    """手工构造 /Count 0 的 PDF：PyMuPDF 不允许 save 一个 0 页文档。"""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [] /Count 0 >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def make_chapter_pdf(
    path: Path,
    chapters: list[tuple[str, int, str]],
    front_pages: int = 2,
) -> None:
    """生成带页眉、章标题、页脚页码的文字层 PDF（front_pages 页无页码）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for number in range(1, front_pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), "系统解剖学 第一篇 运动系统", fontname="china-s", fontsize=9)
        page.insert_text((72, 240), f"前置页 {number}", fontname="china-s", fontsize=12)

    printed = 0
    for title, page_count, body in chapters:
        for index in range(page_count):
            page = doc.new_page(width=595, height=842)
            page.insert_text((72, 60), "系统解剖学 第一篇 运动系统", fontname="china-s", fontsize=9)
            if index == 0:
                page.insert_text((72, 110), title, fontname="china-s", fontsize=18)
            for row in range(6):
                page.insert_text((72, 160 + row * 26), body, fontname="china-s", fontsize=12)
            printed += 1
            page.insert_text((300, 800), str(printed), fontsize=10)
    doc.save(str(path))
    doc.close()


class ZeroPagePdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_zero_"))

    def test_validate_pdf_rejects_zero_page_pdf(self) -> None:
        pdf = self.tmp / "zero.pdf"
        make_zero_page_pdf(pdf)
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate_pdf(pdf)
        self.assertIn("0 页", str(ctx.exception))

    def test_import_rejects_zero_page_pdf_without_leaving_files(self) -> None:
        pdf = self.tmp / "zero.pdf"
        make_zero_page_pdf(pdf)
        task = TaskManager().create_manual("import", "导入")
        with self.assertRaises(ValueError):
            pipeline.import_pdf(task, self.tmp, pdf, title="空页教材")
        self.assertFalse((self.tmp / "res" / "空页教材").exists())
        self.assertEqual(list((self.tmp / "res").rglob("*")), [])


class CorruptedPdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_corrupt_"))

    def test_tail_truncated_pdf_is_flagged_as_repaired(self) -> None:
        """尾部截断（xref/trailer 丢失）MuPDF 会修复后打开，内容往往仍可用。

        因此不直接拒绝，但必须回传 repaired 标记——静默接受损坏文件正是本次要
        消除的『看起来成功』类问题。
        """
        source = self.tmp / "src" / "book.pdf"
        make_chapter_pdf(source, [("第一章 骨学", 2, "骨膜含有丰富的血管和神经。")])
        data = source.read_bytes()
        truncated = self.tmp / "src" / "截断.pdf"
        truncated.write_bytes(data[: len(data) // 2])

        info = pipeline.validate_pdf(truncated)

        self.assertTrue(info["repaired"])
        self.assertGreater(info["pages"], 0)

    def test_intact_pdf_is_not_flagged_as_repaired(self) -> None:
        source = self.tmp / "src" / "ok.pdf"
        make_chapter_pdf(source, [("第一章 骨学", 2, "骨膜含有丰富的血管和神经。")])
        self.assertFalse(pipeline.validate_pdf(source)["repaired"])

    def test_import_logs_a_warning_for_damaged_pdf(self) -> None:
        source = self.tmp / "src" / "book.pdf"
        make_chapter_pdf(source, [("第一章 骨学", 2, "骨膜含有丰富的血管和神经。")])
        data = source.read_bytes()
        damaged = self.tmp / "src" / "尾部截断.pdf"
        damaged.write_bytes(data[: len(data) // 2])

        manager = TaskManager()
        task = manager.create(
            "import",
            "导入损坏教材",
            lambda current: pipeline.import_pdf(current, self.tmp, damaged, title="损坏教材"),
        )
        deadline = time.time() + 120
        while task.status in ("pending", "running") and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(task.status, "done")
        logs = manager.snapshot(task.id)["logs"]
        self.assertTrue(any("已损坏" in line for line in logs), logs)

    def test_pdf_with_wrong_magic_is_rejected(self) -> None:
        fake = self.tmp / "src" / "假.pdf"
        fake.parent.mkdir(parents=True, exist_ok=True)
        fake.write_bytes(b"PK\x03\x04" + b"x" * 500)  # 其实是 zip/其他格式
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate_pdf(fake)
        self.assertIn("%PDF-", str(ctx.exception))


class UploadFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_upload_fail_"))
        self.server = create_server("127.0.0.1", 0, root=self.tmp)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _post(self, path: str, payload: dict) -> tuple[int, dict]:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read())
            finally:
                error.close()

    def _status_of(self, task_id: str) -> str:
        import urllib.request

        with urllib.request.urlopen(f"{self.base}/api/tasks/{task_id}?since=0", timeout=30) as response:
            return json.loads(response.read())["data"]["status"]

    def _wait_terminal(self, task_id: str, timeout: float = 60.0) -> dict:
        import urllib.request

        deadline = time.time() + timeout
        while time.time() < deadline:
            with urllib.request.urlopen(
                f"{self.base}/api/tasks/{task_id}?since=0", timeout=30
            ) as response:
                task = json.loads(response.read())["data"]
            if task["status"] not in ("pending", "running"):
                return task
            time.sleep(0.05)
        raise TimeoutError(task_id)

    def test_interrupted_upload_reports_error_and_cleans_up(self) -> None:
        status, payload = self._post(
            "/api/import/begin", {"filename": "断.pdf", "title": "断教材"}
        )
        self.assertEqual(status, 200)
        begin = payload["data"]
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(
                (
                    f"PUT {begin['upload_url']} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{self.port}\r\n"
                    "Content-Type: application/pdf\r\n"
                    "Content-Length: 1048576\r\n"
                    "\r\n"
                ).encode("ascii")
            )
            sock.sendall(b"x" * 4096)  # 远少于声明长度就断开

        task = self._wait_terminal(begin["task_id"])
        self.assertEqual(task["status"], "error")
        self.assertIn("上传中断", task["error"])
        pdf_dir = self.tmp / "res" / "断教材" / "PDF"
        self.assertEqual(list(pdf_dir.glob("*.part")), [])
        self.assertFalse((pdf_dir / "断.pdf").exists())

    def test_cancelled_upload_is_marked_cancelled_and_cleans_up(self) -> None:
        import urllib.request

        status, payload = self._post(
            "/api/import/begin", {"filename": "取消.pdf", "title": "取消教材"}
        )
        self.assertEqual(status, 200)
        begin = payload["data"]
        declared = 64 * 1024 * 1024
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        stop = threading.Event()

        def pump() -> None:
            try:
                for _ in range(40):
                    if stop.is_set():
                        return
                    sock.sendall(b"x" * (1024 * 1024))
                    time.sleep(0.02)
            except OSError:
                pass

        try:
            sock.sendall(
                (
                    f"PUT {begin['upload_url']} HTTP/1.1\r\n"
                    f"Host: 127.0.0.1:{self.port}\r\n"
                    "Content-Type: application/pdf\r\n"
                    f"Content-Length: {declared}\r\n"
                    "\r\n"
                ).encode("ascii")
            )
            feeder = threading.Thread(target=pump, daemon=True)
            feeder.start()
            deadline = time.time() + 10
            while time.time() < deadline and self._status_of(begin["task_id"]) != "running":
                time.sleep(0.05)
            self.assertEqual(self._status_of(begin["task_id"]), "running")

            cancel = urllib.request.Request(
                f"{self.base}/api/tasks/{begin['task_id']}/cancel", data=b"{}", method="POST"
            )
            with urllib.request.urlopen(cancel, timeout=30) as response:
                self.assertEqual(response.status, 200)

            time.sleep(0.5)  # 留出时间让服务端在读循环里看到取消标记
            stop.set()
            feeder.join(timeout=5)
        finally:
            sock.close()

        task = self._wait_terminal(begin["task_id"])
        self.assertEqual(task["status"], "cancelled")
        pdf_dir = self.tmp / "res" / "取消教材" / "PDF"
        self.assertEqual(list(pdf_dir.glob("*.part")), [])
        self.assertFalse((pdf_dir / "取消.pdf").exists())


class ConcurrentProcessTests(unittest.TestCase):
    """多个**独立进程**同时首次打开并重建同一个索引库。

    这里刻意用 4 个进程：新库从默认日志模式切到 WAL 需要短暂独占锁，且该锁不受
    busy_timeout 保护，两三个进程偶尔能通过、四个才稳定暴露问题（曾实测复现
    “database is locked”）。
    """

    PROCESSES = 4

    def test_concurrent_processes_rebuild_same_index(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_procs_"))
        book_dir = tmp / "res" / "并发教材"
        structured = book_dir / "processed_v3" / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        (structured / "01-肌学.md").write_text(
            "# 第一章 肌学\n\n## 原书第 10 页\n\n骨骼肌由肌腹和肌腱构成。\n", encoding="utf-8"
        )
        db = tmp / ".medical_rag" / "library.sqlite3"
        script = (
            "import sys\n"
            "from medical_rag.library import Library\n"
            "lib = Library(sys.argv[1])\n"
            "lib.ingest_markdown_tree(sys.argv[2], '并发教材')\n"
            "lib.close()\n"
            "print('OK')\n"
        )
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", script, str(db), str(book_dir / "processed_v3")],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            for _ in range(self.PROCESSES)
        ]
        for proc in procs:
            out, err = proc.communicate(timeout=300)
            self.assertEqual(proc.returncode, 0, err)
            self.assertIn("OK", out)

        library = Library(db)
        try:
            self.assertEqual(len(library.list_books()), 1)  # 不重复建书
            chunks = int(library.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            self.assertEqual(chunks, 1)  # 不重复插块
            self.assertTrue(library.fts_integrity_ok())  # 三套索引与 chunks 一致
            self.assertTrue(library.search("骨骼肌", 3))
            self.assertEqual(library.journal_mode, "wal")
        finally:
            library.close()

    def test_wal_setup_degrades_instead_of_raising(self) -> None:
        """切 WAL 失败时只能降级，不能让构造 Library 直接抛异常。

        sqlite3.Connection 是 C 扩展类型，没法用 mock.patch.object 打补丁（不允许
        设置属性），因此用一个可注入失败的假连接——反而能把重试逻辑测得最清楚。
        """
        from medical_rag.library import _enable_wal

        self.assertEqual(_enable_wal(_FlakyConnection(fail_times=99), attempts=1), "unknown")

    def test_wal_setup_retries_then_succeeds(self) -> None:
        """第一次读取就撞锁时，退避重试应能看到已被其他进程切好的 WAL。"""
        from medical_rag.library import _enable_wal

        connection = _FlakyConnection(fail_times=1, mode="wal")
        self.assertEqual(_enable_wal(connection, attempts=3), "wal")
        self.assertGreaterEqual(connection.calls, 2)

    def test_wal_is_not_rewritten_when_already_enabled(self) -> None:
        """库已是 WAL 时不应再去取独占锁（这正是多进程撞锁的根源）。"""
        from medical_rag.library import _enable_wal

        connection = _FlakyConnection(fail_times=0, mode="wal")
        self.assertEqual(_enable_wal(connection), "wal")
        self.assertEqual(connection.calls, 1)  # 只读一次，未执行切换

    def test_wal_switch_is_attempted_for_non_wal_database(self) -> None:
        from medical_rag.library import _enable_wal

        connection = _FlakyConnection(fail_times=0, mode="delete")
        self.assertEqual(_enable_wal(connection), "delete")
        self.assertGreaterEqual(connection.calls, 3)  # 读 → 切换 → 再读确认

    def test_wal_setup_works_on_real_connection(self) -> None:
        """用一个真连接校对假连接没偏离真实行为。"""
        from medical_rag.library import _enable_wal

        tmp = Path(tempfile.mkdtemp(prefix="medrag_wal_real_"))
        connection = sqlite3.connect(tmp / "probe.sqlite3")
        try:
            self.assertEqual(_enable_wal(connection), "wal")
            # 已经是 WAL：再调一次不应重新执行切换
            self.assertEqual(_enable_wal(connection), "wal")
        finally:
            connection.close()


class _Row:
    """够用的假 cursor：_enable_wal 只需要 fetchone()。"""

    def __init__(self, values):
        self._values = values

    def fetchone(self):
        return self._values


class _FlakyConnection:
    """前 ``fail_times`` 次 execute 抛 database is locked，之后返回固定模式。"""

    def __init__(self, fail_times: int = 0, mode: str = "delete"):
        self.fail_times = fail_times
        self.mode = mode
        self.calls = 0

    def execute(self, sql, *args):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise sqlite3.OperationalError("database is locked")
        return _Row((self.mode,))


class StaleContentEndToEndTests(unittest.TestCase):
    """全链路：源 PDF 变短 → 重建结构化 → 清理旧章节 → 老内容不再被检索。"""

    def test_shrinking_pdf_removes_old_chapters_from_the_index(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_e2e_stale_"))
        book_dir = tmp / "res" / "测试教材"
        pdf = book_dir / "PDF" / "测试教材.pdf"
        processed = book_dir / "processed_v3"

        # 1) 10 页、两章：第二章含“滑膜层”
        make_chapter_pdf(
            pdf,
            [("第一章 骨学", 3, "骨膜含有丰富的血管和神经。"), ("第二章 关节学", 5, "关节囊由纤维层和滑膜层构成。")],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            pdftext.build_structured(pdf, book_dir)
        self.assertEqual(len(list((processed / "structured").glob("*.md"))), 2)
        self.assertEqual(len(list((processed / "cleaned").glob("*.md"))), 10)

        library = Library(tmp / ".medical_rag" / "library.sqlite3")
        try:
            library.ingest_markdown_tree(processed, "测试教材")
            self.assertTrue(library.search("滑膜层", 3), "第二章内容应可检索")
            before_chunks = int(library.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

            # 2) 换成只有 5 页、单章的 PDF
            time.sleep(0.01)
            make_chapter_pdf(pdf, [("第一章 骨学", 3, "骨膜含有丰富的血管和神经。")])
            with contextlib.redirect_stdout(io.StringIO()):
                pdftext.build_structured(pdf, book_dir)

            self.assertEqual(len(list((processed / "structured").glob("*.md"))), 1)
            self.assertEqual(len(list((processed / "cleaned").glob("*.md"))), 5)

            # 3) 重新索引：老章节内容必须彻底消失
            library.ingest_markdown_tree(processed, "测试教材")
            self.assertEqual(library.search("滑膜层", 3), [], "旧章节内容不应再被检索到")
            self.assertTrue(library.search("骨膜", 3), "保留章节仍然可检索")

            after_chunks = int(library.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            self.assertLess(after_chunks, before_chunks)
            self.assertTrue(library.fts_integrity_ok())
            self.assertEqual(library.stale_books(), [], "重新索引后应不再过期")
        finally:
            library.close()


if __name__ == "__main__":
    unittest.main()
