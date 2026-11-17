from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import fitz

from medical_rag import pipeline
from medical_rag.tasks import TaskManager

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"
# 流水线脚本已移到 tools/，它们之间是平级导入（如 tools_structure_v3 导入
# tools_structure_v2），因此测试里必须把该目录加入 sys.path
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


def make_pdf(path: Path, pages: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Anatomy chapter {index + 1}", fontsize=14)
        page.insert_text((300, 800), str(index + 1), fontsize=10)
    doc.save(str(path))
    doc.close()


def box(text: str, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "bbox": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "xc": (x0 + x1) / 2, "text": text, "score": 0.99,
    }


def write_boxes(book_dir: Path, pages: int = 2) -> None:
    boxes_dir = book_dir / "text_v3" / "boxes"
    boxes_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, pages + 1):
        boxes = [
            box(f"骨骼肌 第{n}页正文", 300, 800, 2000, 880),
            box(str(n), 1200, 3400, 1280, 3450),
        ]
        payload = {
            "page": n, "width": 2480, "height": 3508, "boxes": boxes,
            "lines": [item["text"] for item in boxes], "avg_confidence": 0.99,
        }
        (boxes_dir / f"page-{n:04d}.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def status(
    ocr: bool = True,
    layout: bool = True,
    tables: bool = True,
    structure: bool = True,
    index: bool = True,
    workflow: str = "ocr",
) -> dict:
    return {
        "workflow": workflow,
        "ocr": {"complete": ocr},
        "layout": {"complete": layout},
        "tables": {"complete": tables},
        "structure": {"complete": structure},
        "index": {"complete": index},
    }


def make_context(root: Path) -> pipeline.PipelineContext:
    book_dir = root / "res" / "测试教材"
    context = pipeline.PipelineContext(
        root=root,
        book_dir=book_dir,
        pdf=book_dir / "PDF" / "测试教材.pdf",
        title="测试教材",
    )
    context.python = "main-python"
    context.ocr_python = "ocr-python"
    context.paddle_python = "paddle-python"
    return context


class StagePlanTests(unittest.TestCase):
    def test_auto_returns_only_missing_stages_in_order(self) -> None:
        self.assertEqual(pipeline.plan_stages(status()), ["fix", "structure", "index"])
        self.assertEqual(pipeline.plan_stages(status(index=False)), ["index"])
        self.assertEqual(pipeline.plan_stages(status(ocr=False)), ["ocr", "fix", "structure", "index"])
        self.assertEqual(
            pipeline.plan_stages(status(ocr=False, layout=False, tables=False, structure=False, index=False)),
            ["ocr", "layout", "tables", "fix", "structure", "index"],
        )

    def test_normalize_sorts_dedupes_and_supports_auto(self) -> None:
        self.assertEqual(
            pipeline.normalize_stages(["index", "ocr", "ocr"], status()),
            ["ocr", "index"],
        )
        self.assertEqual(pipeline.normalize_stages("all"), pipeline.STAGE_KEYS)
        self.assertEqual(pipeline.normalize_stages("auto", status(index=False)), ["index"])
        with self.assertRaises(ValueError):
            pipeline.normalize_stages(["nope"])


class TextWorkflowPlanTests(unittest.TestCase):
    def test_auto_uses_text_pipeline(self) -> None:
        state = status(ocr=False, layout=False, tables=False, structure=False, index=False, workflow="text")
        self.assertEqual(pipeline.plan_stages(state), ["text", "index"])
        state["structure"]["complete"] = True
        self.assertEqual(pipeline.plan_stages(state), ["index"])
        state["index"]["complete"] = True
        self.assertEqual(pipeline.plan_stages(state), ["text", "index"])

    def test_normalize_sorts_text_stage_first(self) -> None:
        self.assertEqual(pipeline.normalize_stages(["index", "text"]), ["text", "index"])

    def test_text_stage_command_targets_pdftext_module(self) -> None:
        context = make_context(Path(tempfile.mkdtemp(prefix="medrag_text_cmd_")))
        command = pipeline.stage_command(context, "text")
        self.assertEqual(command[0], "main-python")
        self.assertIn("-m", command)
        self.assertIn("medical_rag.pdftext", command)
        self.assertIn(str(context.pdf), command)
        self.assertIn(str(context.book_dir), command)


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_cmd_"))
        self.context = make_context(self.tmp)

    def test_ocr_force_adds_reset_and_uses_ocr_python(self) -> None:
        command = pipeline.stage_command(self.context, "ocr")
        self.assertEqual(command[0], "ocr-python")
        self.assertEqual(command[1], str(pipeline.TOOLS_DIR / "tools_ocr_v3.py"))
        self.assertNotIn("--reset", command)
        self.assertIn("--reset", pipeline.stage_command(self.context, "ocr", force=True))
        self.assertIn("--reset", pipeline.stage_command(
            pipeline.PipelineContext(**{**self.context.__dict__, "reset_ocr": True}), "ocr"
        ))

    def test_paddle_stages_use_paddle_python_and_layout(self) -> None:
        layout = pipeline.stage_command(self.context, "layout")
        self.assertEqual(layout[0], "paddle-python")
        self.assertIn("--device", layout)
        tables = pipeline.stage_command(self.context, "tables")
        self.assertEqual(tables[0], "paddle-python")
        self.assertIn(str(self.context.text_dir / "layout.json"), tables)

    def test_index_command_targets_processed_dir(self) -> None:
        command = pipeline.stage_command(self.context, "index")
        self.assertEqual(command[0], "main-python")
        self.assertIn("-m", command)
        self.assertIn("medical_rag.cli", command)
        self.assertIn(str(self.context.processed_dir), command)
        self.assertIn("--title", command)


class StageFailureTests(unittest.TestCase):
    def test_failed_stage_error_includes_recent_subprocess_output(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_stage_failure_"))
        context = make_context(tmp)
        make_pdf(context.pdf)

        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.StringIO("启动\n致命错误：无法写入输出目录\n")
                self._code: int | None = None

            def poll(self):
                return self._code

            def wait(self):
                self._code = 1
                return self._code

            def kill(self):
                self._code = -9

        task = TaskManager().create_manual("process", "失败阶段")
        with mock.patch.object(pipeline, "_check_prerequisites"):
            with mock.patch.object(pipeline, "stage_command", return_value=["fake-command"]):
                with mock.patch.object(pipeline.subprocess, "Popen", return_value=FakeProcess()):
                    with self.assertRaises(RuntimeError) as ctx:
                        pipeline.run_stage(task, context, "text")
        message = str(ctx.exception)
        self.assertIn("退出码 1", message)
        self.assertIn("致命错误：无法写入输出目录", message)


class ProgressParseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_prog_"))
        self.context = make_context(self.tmp)
        (self.context.book_dir / "text_v3").mkdir(parents=True, exist_ok=True)
        (self.context.book_dir / "chapters.json").write_text(
            json.dumps([[1, 1, "骨学"], [2, 5, "关节学"]], ensure_ascii=False), encoding="utf-8"
        )
        (self.context.text_dir / "layout.json").write_text(
            json.dumps({"table_pages": [2, 7]}), encoding="utf-8"
        )

    def test_ocr_line(self) -> None:
        self.assertEqual(
            pipeline._progress_from_line(self.context, "ocr", "12/368 第12页  1.2s  35 boxes  avg=0.9770", 0),
            (12, 368, "第 12 页"),
        )

    def test_text_line_uses_same_pattern(self) -> None:
        self.assertEqual(
            pipeline._progress_from_line(self.context, "text", "3/100 第3页", 0),
            (3, 100, "第 3 页"),
        )

    def test_layout_line(self) -> None:
        update = pipeline._progress_from_line(self.context, "layout", "  25/368  0.52s/页  表格页累计=3", 0)
        self.assertEqual(update[:2], (25, 368))

    def test_tables_line(self) -> None:
        self.assertEqual(
            pipeline._progress_from_line(self.context, "tables", "[第7页] 1.2s  2 个表格", 2),
            (2, 2, "第 7 页"),
        )

    def test_fix_structure_index_lines(self) -> None:
        self.assertEqual(pipeline._progress_from_line(self.context, "fix", "修改文件数: 12", 0), (1, 1, "修改 12 个文件"))
        self.assertEqual(
            pipeline._progress_from_line(self.context, "structure", "第2章 关节学: PDF 5-9 → 原书 1-5", 0)[:2],
            (2, 2),
        )
        self.assertEqual(pipeline._progress_from_line(self.context, "index", "已处理《测试教材》：总页数 3", 0), (1, 1, "索引完成"))
        self.assertIsNone(pipeline._progress_from_line(self.context, "ocr", "引擎: PP-OCR6 small", 0))


class ImportPdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_import_"))
        self.source = self.tmp / "src" / "我的教材.pdf"
        make_pdf(self.source)

    def test_copy_import_creates_book_dir(self) -> None:
        task = TaskManager().create_manual("import", "导入")
        book_dir, target = pipeline.import_pdf(task, self.tmp, self.source, title="生理学 第9版")
        self.assertTrue(target.exists())
        self.assertEqual(book_dir.name, "生理学 第9版")
        self.assertTrue((book_dir / "PDF" / "我的教材.pdf").exists())

    def test_duplicate_pdf_is_rejected(self) -> None:
        task = TaskManager().create_manual("import", "导入")
        pipeline.import_pdf(task, self.tmp, self.source, title="生理学 第9版")
        other = self.tmp / "src" / "另一本.pdf"
        make_pdf(other)
        with self.assertRaises(FileExistsError):
            pipeline.import_pdf(task, self.tmp, other, title="生理学 第9版")

    def test_validate_pdf_accepts_real_pdf(self) -> None:
        info = pipeline.validate_pdf(self.source)
        self.assertEqual(info["pages"], 2)
        self.assertGreater(info["size"], 0)

    def test_validate_pdf_rejects_malformed_files(self) -> None:
        cases = {
            "文本伪装": b"not a pdf",
            "空文件": b"",
            "只有文件头": b"%PDF-1.4\n",
        }
        for label, payload in cases.items():
            path = self.tmp / "src" / f"{label}.pdf"
            path.write_bytes(payload)
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    pipeline.validate_pdf(path)

    def test_validate_pdf_rejects_truncated_pdf(self) -> None:
        data = self.source.read_bytes()
        truncated = self.tmp / "src" / "截断.pdf"
        truncated.write_bytes(data[: len(data) // 2])
        with self.assertRaises(ValueError):
            pipeline.validate_pdf(truncated)

    def test_validate_pdf_enforces_page_limit(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate_pdf(self.source, max_pages=1)
        self.assertIn("上限", str(ctx.exception))

    def test_validate_pdf_rejects_decompression_bomb(self) -> None:
        """522 字节的 PDF 声明 20000×20000 pt 页面（300 DPI 需 ~20.8 GB），必须拒绝。"""
        bomb = self.tmp / "src" / "炸弹.pdf"
        doc = fitz.open()
        doc.new_page(width=20000, height=20000)
        doc.save(str(bomb))
        doc.close()
        self.assertLess(bomb.stat().st_size, 4096)  # 确实是很小的文件
        with self.assertRaises(ValueError) as ctx:
            pipeline.validate_pdf(bomb)
        self.assertIn("尺寸异常", str(ctx.exception))

    def test_import_invalid_pdf_fails_and_leaves_no_trace(self) -> None:
        """无效 PDF：抛错、不落盘、不留 .part，且清掉刚建的空目录。"""
        bad = self.tmp / "src" / "坏教材.pdf"
        bad.write_text("not a pdf", encoding="utf-8")
        task = TaskManager().create_manual("import", "导入")
        with self.assertRaises(ValueError):
            pipeline.import_pdf(task, self.tmp, bad, title="坏教材")
        self.assertFalse((self.tmp / "res" / "坏教材").exists())
        self.assertEqual(list((self.tmp / "res").rglob("*.part")), [])
        self.assertEqual(list((self.tmp / "res").rglob("坏教材.pdf")), [])

    def test_import_failure_is_reported_as_task_error(self) -> None:
        """走 TaskManager 时，导入失败必须让任务变成 error（不是 done）。"""
        bad = self.tmp / "src" / "坏教材.pdf"
        bad.write_text("not a pdf", encoding="utf-8")
        manager = TaskManager()
        task = manager.create(
            "import", "导入坏教材",
            lambda current: pipeline.import_pdf(current, self.tmp, bad, title="坏教材"),
        )
        deadline = time.time() + 30
        while task.status in ("pending", "running") and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(task.status, "error")
        self.assertIn("不是 PDF", task.error)

    def test_import_copy_uses_part_file_without_residue(self) -> None:
        task = TaskManager().create_manual("import", "导入")
        pipeline.import_pdf(task, self.tmp, self.source, title="生理学 第9版")
        pdf_dir = self.tmp / "res" / "生理学 第9版" / "PDF"
        self.assertEqual([p.name for p in pdf_dir.glob("*.pdf")], ["我的教材.pdf"])
        self.assertEqual(list(pdf_dir.glob("*.part")), [])


class PipelineEndToEndTests(unittest.TestCase):
    def test_structure_and_index_in_temp_root(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_e2e_"))
        book_dir = tmp / "res" / "测试教材"
        pdf = book_dir / "PDF" / "测试教材.pdf"
        make_pdf(pdf, pages=2)
        write_boxes(book_dir, pages=2)
        (book_dir / "chapters.json").write_text(json.dumps([[1, 1, "骨骼肌"]], ensure_ascii=False), encoding="utf-8")
        db = tmp / ".medical_rag" / "library.sqlite3"
        book = {"dir": str(book_dir), "pdf": str(pdf), "title": "测试教材"}
        manager = TaskManager()
        with mock.patch.dict(os.environ, {"MEDICAL_RAG_DB": str(db)}):
            task = manager.create(
                "process", "测试处理",
                lambda current: pipeline.run_book_pipeline(current, tmp, book, stages=["structure", "index"]),
            )
            deadline = time.time() + 120
            while task.status in ("pending", "running") and time.time() < deadline:
                time.sleep(0.05)
        snapshot = manager.snapshot(task.id)
        self.assertEqual(snapshot["status"], "done", snapshot.get("error"))
        self.assertEqual([stage["status"] for stage in snapshot["stages"]], ["done", "done"])
        structured = list((book_dir / "processed_v3" / "structured").glob("*.md"))
        self.assertEqual(len(structured), 1)

        import sqlite3

        connection = sqlite3.connect(db)
        try:
            chunks = connection.execute("SELECT page, text FROM chunks ORDER BY page").fetchall()
        finally:
            connection.close()
        self.assertEqual([row[0] for row in chunks], [1, 2])
        self.assertTrue(all("骨骼肌" in row[1] for row in chunks))


class StructureStaleCleanupTests(unittest.TestCase):
    """OCR 路径（tools_structure_v3）重建时，旧章节文件必须被删除。"""

    def _run(self, book_dir: Path, total_pages: int) -> None:
        import tools_structure_v3

        argv = [
            str(TOOLS_DIR / "tools_structure_v3.py"),
            "--book-dir",
            str(book_dir),
            "--total-pages",
            str(total_pages),
        ]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            tools_structure_v3.main()

    def test_rerun_removes_obsolete_chapter_files(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_struct_"))
        book_dir = tmp / "res" / "测试教材"
        write_boxes(book_dir, pages=4)
        chapters = book_dir / "chapters.json"
        chapters.write_text(
            json.dumps([[1, 1, "肌学"], [2, 3, "关节学"]], ensure_ascii=False), encoding="utf-8"
        )
        out = book_dir / "processed_v3"

        self._run(book_dir, 4)
        self.assertEqual(
            sorted(path.name for path in (out / "structured").glob("*.md")),
            ["01-肌学.md", "02-关节学.md"],
        )

        # 第二次合并为一章：02-关节学.md 不能残留
        chapters.write_text(json.dumps([[1, 1, "运动系统"]], ensure_ascii=False), encoding="utf-8")
        self._run(book_dir, 4)
        self.assertEqual(
            sorted(path.name for path in (out / "structured").glob("*.md")),
            ["01-运动系统.md"],
        )
        self.assertEqual(list(out.glob(".staging-*")), [])
        self.assertEqual(list(out.glob(".backup-*")), [])
        self.assertTrue((out / "quality.json").exists())


class StageTimeoutTests(unittest.TestCase):
    """零超时 + 读循环只认行 → 子进程静默挂起时任务永久 running、这本书永久锁死。"""

    def test_silent_stage_times_out_instead_of_hanging(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_timeout_"))
        context = make_context(tmp)
        make_pdf(context.pdf)
        started = time.time()

        class BlockingStdout:
            """永远没有行、也没有 EOF：等价于孙进程一直握着 stdout 管道。"""

            def __init__(self) -> None:
                self._stop = threading.Event()

            def __iter__(self):
                return self

            def __next__(self):
                self._stop.wait(5)
                raise StopIteration

            def close(self) -> None:
                self._stop.set()

        class SilentProcess:
            def __init__(self) -> None:
                self.stdout = BlockingStdout()
                self.killed = 0
                self._code: int | None = None

            def poll(self):
                return self._code

            def wait(self, timeout=None):
                if self._code is None:
                    self._code = 0
                return self._code

            def kill(self):
                self.killed += 1
                self._code = -9

        process = SilentProcess()
        task = TaskManager().create_manual("process", "静默阶段")
        with mock.patch.object(pipeline, "_check_prerequisites"):
            with mock.patch.object(pipeline, "stage_command", return_value=["fake-command"]):
                with mock.patch.dict(pipeline.STAGE_TIMEOUTS, {"text": 1}):
                    with mock.patch.object(pipeline.subprocess, "Popen", return_value=process):
                        with self.assertRaises(RuntimeError) as ctx:
                            pipeline.run_stage(task, context, "text")
        self.assertIn("超时", str(ctx.exception))
        self.assertLess(time.time() - started, 10)
        self.assertEqual(process.killed, 1)  # finally 里必须真的终止并回收

    def test_stage_timeout_budget_prefers_env_override(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_timeout_env_"))
        context = make_context(tmp)
        self.assertEqual(pipeline._stage_timeout(context, "layout"), 1200)
        with mock.patch.dict(os.environ, {"MEDICAL_RAG_STAGE_TIMEOUT_LAYOUT": "7"}):
            self.assertEqual(pipeline._stage_timeout(context, "layout"), 7)
        with mock.patch.dict(os.environ, {"MEDICAL_RAG_STAGE_TIMEOUT_LAYOUT": "0"}):
            self.assertIsNone(pipeline._stage_timeout(context, "layout"))  # 0 = 不限时


class CancelKillTests(unittest.TestCase):
    def test_cancel_kills_running_process(self) -> None:
        """进程还活着时，取消要立刻穿透（不必等下一个 1s 轮询）。"""

        class RunningProcess:
            def __init__(self) -> None:
                self.killed = 0

            def poll(self):
                return None  # 还在跑

            def wait(self, timeout=None):
                return 0

            def kill(self) -> None:
                self.killed += 1

        process = RunningProcess()
        task = TaskManager().create_manual("process", "取消")
        task.attach_process(process)
        task.cancel()
        self.assertEqual(process.killed, 1)

    def test_cancel_does_not_kill_already_exited_process(self) -> None:
        """父进程已退出时不要再按旧 PID 杀树。

        ``Task._process`` 会一直留到 ``_run`` 的 finally，阶段切换的空窗里它指向
        的是**上一个已退出的**子进程；Windows 的 PID 会被复用，此时
        ``taskkill /T /F`` 可能打到无关进程上。「父进程已退、孙进程还握管道」
        那种情况由 ``run_stage`` 的 finally 处理（它手里有真实 pid，且
        ``StageTimeoutTests`` 覆盖）。
        """

        class ExitedProcess:
            def __init__(self) -> None:
                self.killed = 0

            def poll(self):
                return 0  # 父进程已退出

            def wait(self, timeout=None):
                return 0

            def kill(self) -> None:
                self.killed += 1

        process = ExitedProcess()
        task = TaskManager().create_manual("process", "取消")
        task.attach_process(process)
        task.cancel()
        self.assertEqual(process.killed, 0)


class RenderCacheKeyTests(unittest.TestCase):
    def test_cache_dir_changes_when_pdf_changes(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_cachekey_"))
        context = make_context(tmp)
        make_pdf(context.pdf, pages=2)
        layout_first = pipeline.stage_command(context, "layout")
        tables = pipeline.stage_command(context, "tables")
        cache_first = layout_first[layout_first.index("--cache") + 1]
        self.assertIn("render300_", tables[tables.index("--cache") + 1])
        # 同路径换一版 PDF：缓存目录必须跟着变，否则旧页图被当新书证据
        make_pdf(context.pdf, pages=3)
        os.utime(context.pdf, (time.time() - 100, time.time() - 100))
        layout_second = pipeline.stage_command(context, "layout")
        self.assertNotEqual(cache_first, layout_second[layout_second.index("--cache") + 1])


class ToolsCommonTests(unittest.TestCase):
    """tools/ 产物原子写 + 续跑校验（半截 JSON 必须被识别为坏页）。"""

    def test_atomic_write_keeps_old_content_on_failure(self) -> None:
        import tools_common

        target = Path(tempfile.mkdtemp(prefix="medrag_atomic_")) / "page-0001.json"
        target.write_text('{"page": 1, "boxes": []}', encoding="utf-8")
        tools_common.atomic_write_text(target, '{"page": 1, "boxes": [{"text": "骨"}]}')
        self.assertEqual(len(json.loads(target.read_text(encoding="utf-8"))["boxes"]), 1)
        with mock.patch.object(tools_common.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                tools_common.atomic_write_text(target, '{"page": 1, "boxes": [')
        # 失败时旧内容完好，且不留临时文件
        self.assertEqual(len(json.loads(target.read_text(encoding="utf-8"))["boxes"]), 1)
        self.assertEqual(list(target.parent.glob("*.tmp-*")), [])

    def test_valid_page_json_rejects_truncated_and_wrong_page(self) -> None:
        import tools_common

        box_dir = Path(tempfile.mkdtemp(prefix="medrag_page_"))
        good = box_dir / "page-0001.json"
        good.write_text(json.dumps({"page": 1, "boxes": [{"text": "骨"}]}), encoding="utf-8")
        self.assertTrue(tools_common.valid_page_json(good, 1))
        self.assertFalse(tools_common.valid_page_json(good, 2))  # 页号不符
        truncated = box_dir / "page-0002.json"
        truncated.write_text('{"page": 2, "boxes": [{"text": "骨', encoding="utf-8")
        self.assertFalse(tools_common.valid_page_json(truncated, 2))

    def test_cached_render_invalidates_when_pdf_content_changes(self) -> None:
        """修复前只判 ``out.exists()``：换一版 PDF 会拿到字节相同的旧页图。"""
        import hashlib

        import tools_common

        tmp = Path(tempfile.mkdtemp(prefix="medrag_render_"))
        pdf = tmp / "book.pdf"
        cache = tmp / "cache"
        make_pdf(pdf, pages=1)
        first, _, _ = tools_common.cached_render(pdf, 1, 72, cache)
        digest_first = hashlib.sha256(first.read_bytes()).hexdigest()

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 400), "完全不同的一页", fontsize=40)
        doc.save(str(pdf))
        doc.close()
        os.utime(pdf, (time.time() - 500, time.time() - 500))

        second, _, _ = tools_common.cached_render(pdf, 1, 72, cache)
        self.assertNotEqual(digest_first, hashlib.sha256(second.read_bytes()).hexdigest())
        # 同一版 PDF 再来一次：命中缓存，字节不变
        third, _, _ = tools_common.cached_render(pdf, 1, 72, cache)
        self.assertEqual(hashlib.sha256(third.read_bytes()).hexdigest(), hashlib.sha256(second.read_bytes()).hexdigest())


class OcrResumeTests(unittest.TestCase):
    """坏页既要在续跑时重做，也不能让 manifest 汇总崩掉。"""

    def test_corrupt_page_is_skipped_with_warning(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_ocr_"))
        book = tmp / "text_v3"
        boxes = book / "boxes"
        boxes.mkdir(parents=True)
        pdf = tmp / "book.pdf"
        make_pdf(pdf, pages=2)
        (boxes / "page-0001.json").write_text(
            json.dumps({"page": 1, "boxes": [], "lines": ["正文"], "avg_confidence": 0.9}),
            encoding="utf-8",
        )
        # 模拟写入瞬间被 kill：文件存在但内容截断
        (boxes / "page-0002.json").write_text('{"page": 2, "boxes": [', encoding="utf-8")

        sys.modules.pop("tools_ocr_v3", None)
        with mock.patch.dict(sys.modules, {"rapidocr": mock.MagicMock()}):
            import tools_ocr_v3

            argv = ["tools_ocr_v3.py", str(pdf), str(book), "--pages", "1"]
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr):
                tools_ocr_v3.main()

        self.assertIn("跳过损坏", stderr.getvalue())
        manifest = json.loads((book / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual([item["page"] for item in manifest["pages"]], [1])
        self.assertEqual(manifest["corrupt_pages"], [2])


if __name__ == "__main__":
    unittest.main()
