from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

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
        """CLI 不传 --root/--db 时曾经会 Path(None) 崩溃。"""
        report = doctor.check_environment(None, None)
        self.assertTrue(report["root"])
        self.assertTrue(report["db"])

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

        用 sys.executable 冒充\"另一个\"解释器（路径不同但内容相同）来验证走的是
        子进程分支，而不是当前进程的 importlib.metadata。
        """
        alias = Path(sys.executable).parent / ".." / Path(sys.executable).parent.name / Path(sys.executable).name
        found = doctor.interpreter_package_versions(str(alias), ("PyMuPDF",))
        self.assertIsNotNone(found["PyMuPDF"])

    def test_interpreter_info_marks_missing_venvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            info = doctor.interpreter_info(Path(tmp))
            self.assertFalse(info["venv_ocr"])
            self.assertFalse(info["venv_ocr312"])
            self.assertTrue(info["python"])


if __name__ == "__main__":
    unittest.main()
