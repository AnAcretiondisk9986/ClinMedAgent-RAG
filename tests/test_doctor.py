from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medical_rag import doctor
from medical_rag.library import Library


def make_library(root: Path) -> Path:
    """在 root 下建一个只有 1 个证据块的索引库。"""
    structured = root / "processed_v3" / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    (structured / "01-ch.md").write_text(
        "# Ch1\n\n## 原书第 10 页\n\nperiosteum vessels zzzold\n", encoding="utf-8"
    )
    db = root / ".medical_rag" / "library.sqlite3"
    lib = Library(db)
    try:
        lib.ingest_markdown_tree(root / "processed_v3", "测试教材")
    finally:
        lib.close()
    return db


def break_fts(db: Path) -> None:
    """绕过 FTS 维护直接改 chunks，模拟历史版本留下的一致性损坏。"""
    lib = Library(db)
    try:
        lib.cx.execute("UPDATE chunks SET text = 'changed content'")
        lib.cx.commit()
    finally:
        lib.close()


class CheckEnvironmentTests(unittest.TestCase):
    def test_missing_database_is_warning_not_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = doctor.check_environment(Path(tmp))
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], 0)
            self.assertFalse(report["database"]["exists"])
            self.assertIn("索引库", [item["name"] for item in report["checks"]])

    def test_healthy_database_reports_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = make_library(root)
            report = doctor.check_environment(root, db)
            self.assertEqual(report["database"]["books"], 1)
            self.assertEqual(report["database"]["chunks"], 1)
            self.assertTrue(report["database"]["fts_integrity"])
            self.assertEqual(report["errors"], 0)
            self.assertTrue(report["ok"])

    def test_detects_fts_drift_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = make_library(root)
            break_fts(db)
            report = doctor.check_environment(root, db)
            self.assertFalse(report["ok"])
            self.assertGreaterEqual(report["errors"], 1)
            drift = [item for item in report["checks"] if item["name"] == "FTS 索引一致性"]
            self.assertEqual(len(drift), 1)
            self.assertEqual(drift[0]["status"], "error")
            self.assertIn("--repair", drift[0]["hint"])

    def test_repair_fixes_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = make_library(root)
            break_fts(db)
            self.assertFalse(doctor.check_environment(root, db)["ok"])

            result = doctor.repair(root, db)

            self.assertTrue(result["repaired"])
            self.assertFalse(result["before_ok"])
            self.assertTrue(result["after_ok"])
            self.assertTrue(doctor.check_environment(root, db)["ok"])

    def test_repair_without_database_reports_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = doctor.repair(Path(tmp))
            self.assertFalse(result["repaired"])
            self.assertIn("不存在", result["reason"])

    def test_none_arguments_do_not_crash(self) -> None:
        """CLI 不传 --root/--db 时曾经会 Path(None) 崩溃。

        用临时根目录而不是真实索引库：否则这条「不崩」用例会打开 44 MB 真库，
        结果随宿主环境漂移，而且把「索引库报错」混进了本该通过的用例里。
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = doctor.check_environment(root, root / "none.sqlite3")
        self.assertTrue(report["root"])
        self.assertTrue(report["db"])
        self.assertEqual(report["errors"], 0)

    def test_format_report_renders_checks_and_conclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = doctor.format_report(doctor.check_environment(Path(tmp)))
            self.assertIn("环境自检", text)
            self.assertIn("结论", text)


class CompactTests(unittest.TestCase):
    """重建索引会累积空闲页；optimize + VACUUM 才能回收。"""

    def _churned_library(self, root: Path, rounds: int = 6) -> Path:
        structured = root / "book" / "processed_v3" / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        body = "".join(
            f"## 原书第 {page} 页\n\n骨膜含有丰富的血管和神经，关节囊由纤维层和滑膜层构成。\n\n"
            for page in range(1, 40)
        )
        (structured / "01-章节.md").write_text(f"# 第一章\n\n{body}", encoding="utf-8")
        db = root / "library.sqlite3"
        library = Library(db)
        try:
            for _ in range(rounds):
                library.ingest_markdown_tree(root / "book" / "processed_v3", "测试教材")
        finally:
            library.close()
        return db

    def test_compact_reclaims_space_and_keeps_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = self._churned_library(root)
            before_bytes = db.stat().st_size

            result = doctor.compact(root, db)

            self.assertTrue(result["compacted"])
            self.assertGreater(result["saved_bytes"], 0)
            self.assertLess(result["after_bytes"], before_bytes)
            self.assertEqual(db.stat().st_size, result["after_bytes"])
            self.assertEqual(result["chunks"], 39)
            self.assertTrue(result["fts_integrity"])

            library = Library(db)
            try:
                self.assertTrue(library.fts_integrity_ok())
                self.assertTrue(library.search("骨膜", 2))
            finally:
                library.close()

    def test_compact_is_safe_on_clean_database(self) -> None:
        """已经干净的库不能报错，也不能变坏。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = self._churned_library(root, rounds=1)
            first = doctor.compact(root, db)
            second = doctor.compact(root, db)
            self.assertTrue(second["compacted"])
            self.assertTrue(second["fts_integrity"])
            self.assertLessEqual(second["after_bytes"], first["after_bytes"])
            self.assertEqual(second["chunks"], 39)

    def test_compact_without_database_reports_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = doctor.compact(Path(tmp))
            self.assertFalse(result["compacted"])
            self.assertIn("不存在", result["reason"])


class InterpreterPackageTests(unittest.TestCase):
    def test_current_interpreter_reads_metadata(self) -> None:
        found = doctor.interpreter_package_versions(sys.executable, ("PyMuPDF",))
        self.assertIsNotNone(found["PyMuPDF"])

    def test_missing_package_reports_none(self) -> None:
        found = doctor.interpreter_package_versions(
            sys.executable, ("definitely-not-installed-package",)
        )
        self.assertIsNone(found["definitely-not-installed-package"])

    def test_broken_interpreter_does_not_raise(self) -> None:
        found = doctor.interpreter_package_versions("no-such-python-binary", ("PyMuPDF",))
        self.assertIsNone(found["PyMuPDF"])

    def test_other_interpreter_is_queried_by_subprocess(self) -> None:
        """Paddle 依赖装在另一个解释器里，必须在那个解释器里查版本。

        用 sys.executable 冒充"另一个"解释器（路径不同但内容相同）来验证走的是
        子进程分支，而不是当前进程的 importlib.metadata。断言「真的起了一个子进程」
        ——只看返回值非 None 的话，两边都会成立，改成别的实现在测试里也照样绿。
        """
        alias = Path(sys.executable).parent / ".." / Path(sys.executable).parent.name / Path(sys.executable).name
        with mock.patch(
            "medical_rag.doctor.subprocess.run", wraps=doctor.subprocess.run
        ) as patched:
            found = doctor.interpreter_package_versions(str(alias), ("PyMuPDF",))
        self.assertIsNotNone(found["PyMuPDF"])
        self.assertEqual(patched.call_count, 1)
        self.assertEqual(patched.call_args.args[0][0], str(alias))

    def test_probe_import_really_imports_modules(self) -> None:
        ok, detail = doctor._probe_import(sys.executable, ("fitz",))
        self.assertTrue(ok, detail)

    def test_interpreter_info_marks_missing_venvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = doctor.interpreter_info(Path(tmp))
            self.assertFalse(info["venv_ocr"])
            self.assertFalse(info["venv_ocr312"])
            self.assertTrue(info["python"])


class OutputAccessCheckTests(unittest.TestCase):
    """`doctor` 要提前发现被别的安全上下文写过的（打不开的）产物目录。"""

    def _root_with_book(self, tmp: str) -> Path:
        root = Path(tmp)
        (root / "res" / "某教材" / "processed_v3" / "structured").mkdir(parents=True)
        return root

    def test_reports_ok_for_accessible_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_book(tmp)
            checks = doctor._output_access_check(root)
            self.assertEqual([c["name"] for c in checks], ["产物权限"])
            self.assertEqual(checks[0]["status"], "ok")

    def test_reports_warning_with_hint_when_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_book(tmp)
            with mock.patch(
                "medical_rag.outputs.blocked_outputs", return_value=["cleaned", "quality.json"]
            ):
                checks = doctor._output_access_check(root)
        self.assertEqual(checks[0]["status"], "warn")
        self.assertIn("某教材", checks[0]["detail"])
        self.assertIn("cleaned", checks[0]["detail"])
        self.assertIn("takeown", checks[0]["hint"])

    def test_check_is_included_in_environment_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self._root_with_book(tmp)
            report = doctor.check_environment(root, Path(tmp) / "none.sqlite3")
        self.assertIn("产物权限", [item["name"] for item in report["checks"]])

    def test_missing_res_directory_is_not_a_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(doctor._output_access_check(Path(tmp)), [])


class ReadOnlyDatabaseTests(unittest.TestCase):
    """自检必须真只读：只读介质/受限 ACL 下不能用写模式打开索引库。"""

    def test_self_check_does_not_touch_database(self) -> None:
        """构造 Library 会 DDL + PRAGMA 写库；自检前后库文件 mtime 必须相同。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = make_library(root)
            before = db.stat().st_mtime_ns
            report = doctor.check_environment(root, db)
            after = db.stat().st_mtime_ns
        self.assertEqual(report["database"]["books"], 1)
        self.assertEqual(before, after, "doctor 自检改写了索引库（说明用了读写连接）")

    def test_unreadable_database_is_warning_not_error(self) -> None:
        """只读连接失败（被占用/权限受限）时是 warn；不能把健康库判成 error/503。"""
        import sqlite3
        import time

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = make_library(root)
            started = time.time()
            with mock.patch(
                "medical_rag.workspace.open_readonly",
                side_effect=sqlite3.OperationalError("unable to open database file"),
            ):
                report = doctor.check_environment(root, db)
            elapsed = time.time() - started
        self.assertLess(elapsed, 2)  # 不排 busy_timeout，不能卡 30 秒
        self.assertTrue(report["ok"], "只读打不开是环境限制，不该让健康检查失败")
        self.assertEqual(report["errors"], 0)
        db_checks = [item for item in report["checks"] if item["name"] == "索引库"]
        self.assertEqual(db_checks[0]["status"], "warn")

    @unittest.skipUnless(
        (Path(__file__).resolve().parent.parent / ".build" / "bench" / "db" / "library.sqlite3").exists(),
        "缺少基准库副本",
    )
    def test_readonly_copy_reports_real_counts(self) -> None:
        """把可写库副本改成只读文件属性后，自检仍要读出真实统计（immutable 回退）。"""
        import shutil

        source = Path(__file__).resolve().parent.parent / ".build" / "bench" / "db" / "library.sqlite3"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / ".medical_rag" / "library.sqlite3"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            os.chmod(target, 0o444)
            try:
                report = doctor.check_environment(root, target)
            finally:
                os.chmod(target, 0o666)
        self.assertTrue(report["database"]["exists"])
        self.assertGreaterEqual(report["database"]["books"], 1)
        self.assertGreaterEqual(report["database"]["chunks"], 1)
        self.assertIn("索引库", [item["name"] for item in report["checks"]])


class ReadOnlyMigrationTests(unittest.TestCase):
    """F4：只读介质上的旧 schema 库要降级打开并留 warning，而不是整个库打不开。"""

    def test_old_schema_readonly_library_degrades_with_warning(self) -> None:
        import sqlite3

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db = make_library(root)
            # 造出“老库”：user_version 落后，需要迁移；真正迁移要写 page1
            connection = sqlite3.connect(db)
            try:
                connection.execute("PRAGMA user_version = 0")
                connection.commit()
            finally:
                connection.close()
            os.chmod(db, 0o444)
            try:
                library = Library(db)  # 以前这里抛 OperationalError: readonly database
            finally:
                os.chmod(db, 0o666)
            try:
                self.assertTrue(library.readonly, "只读库迁移失败应降级为只读模式")
                self.assertTrue(library.warnings, "降级必须留下可展示的 warning")
                self.assertIn("只读", library.warnings[0])
                # 降级后检索照旧可用（按现有结构尽力检索）
                self.assertEqual(len(library.list_books()), 1)
                self.assertEqual(library.search("periosteum", 3)[0]["page"], 10)
                # 写操作要给中文说明，而不是裸 sqlite3.OperationalError
                for call in (
                    library.rebuild_fts_index,
                    library.rebuild_vector_index,
                    lambda: library.ingest(root / "x.pdf"),
                ):
                    with self.subTest(call=call):
                        with self.assertRaises(RuntimeError) as ctx:
                            call()
                        self.assertIn("只读", str(ctx.exception))
            finally:
                library.close()


if __name__ == "__main__":
    unittest.main()
