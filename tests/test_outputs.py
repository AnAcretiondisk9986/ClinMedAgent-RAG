from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medical_rag.outputs import (
    BACKUP_PREFIX,
    STAGING_PREFIX,
    chapter_file_name,
    is_internal_output,
    staged_output_dir,
)


class StagedOutputDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_outputs_"))
        self.target = self.tmp / "processed_v3"

    def _write(self, staging: Path, chapters: list[str], pages: int) -> None:
        cleaned = staging / "cleaned"
        structured = staging / "structured"
        cleaned.mkdir(parents=True, exist_ok=True)
        structured.mkdir(parents=True, exist_ok=True)
        for number in range(1, pages + 1):
            (cleaned / f"page-{number:04d}.md").write_text(f"page {number}", encoding="utf-8")
        for name in chapters:
            (structured / name).write_text(f"# {name}", encoding="utf-8")
        (staging / "quality.json").write_text("{}", encoding="utf-8")

    def _residue(self) -> list[str]:
        return sorted(
            path.name
            for path in self.target.iterdir()
            if path.name.startswith((STAGING_PREFIX, BACKUP_PREFIX))
        )

    def test_replaces_and_removes_stale_outputs(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md", "02-乙.md"], pages=3)
        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")),
            ["01-甲.md", "02-乙.md"],
        )
        self.assertEqual(len(list((self.target / "cleaned").glob("*.md"))), 3)

        # 第二次只产出 1 章 1 页：旧的 02-乙.md 与多余页必须消失
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md"], pages=1)
        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")), ["01-甲.md"]
        )
        self.assertEqual(
            sorted(p.name for p in (self.target / "cleaned").glob("*.md")), ["page-0001.md"]
        )
        self.assertTrue((self.target / "quality.json").exists())

    def test_missing_managed_entry_is_deleted(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md"], pages=1)
        self.assertTrue((self.target / "quality.json").exists())
        # 第二次不产出 quality.json → 旧文件应被删除，而不是留下来冒充新结果
        with staged_output_dir(self.target) as staging:
            structured = staging / "structured"
            structured.mkdir(parents=True)
            (structured / "01-甲.md").write_text("x", encoding="utf-8")
        self.assertFalse((self.target / "quality.json").exists())

    def test_failure_keeps_previous_output_and_leaves_no_residue(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md", "02-乙.md"], pages=3)
        before = sorted(p.name for p in (self.target / "structured").glob("*.md"))

        with self.assertRaises(RuntimeError):
            with staged_output_dir(self.target) as staging:
                self._write(staging, ["01-新.md"], pages=1)
                raise RuntimeError("生成失败")

        self.assertEqual(
            sorted(p.name for p in (self.target / "structured").glob("*.md")), before
        )
        self.assertTrue((self.target / "quality.json").exists())
        self.assertEqual(self._residue(), [])

    def test_no_residue_after_success(self) -> None:
        with staged_output_dir(self.target) as staging:
            self._write(staging, ["01-甲.md"], pages=1)
        self.assertEqual(self._residue(), [])

    def test_staging_dir_is_created_inside_target(self) -> None:
        with staged_output_dir(self.target) as staging:
            self.assertEqual(staging.parent, self.target)
            self.assertTrue(staging.name.startswith(STAGING_PREFIX))


class NamingTests(unittest.TestCase):
    def test_chapter_file_name_sanitises_illegal_characters(self) -> None:
        self.assertEqual(chapter_file_name(2, "骨/关节:总论"), "02-骨_关节_总论.md")

    def test_chapter_file_name_falls_back_for_empty_title(self) -> None:
        self.assertEqual(chapter_file_name(3, "   "), "03-第3章.md")

    def test_is_internal_output(self) -> None:
        self.assertTrue(is_internal_output(Path("processed_v3/.staging-abc/structured/01.md")))
        self.assertTrue(is_internal_output(Path("processed_v3/.backup-xyz/01.md")))
        self.assertFalse(is_internal_output(Path("processed_v3/structured/01.md")))


if __name__ == "__main__":
    unittest.main()
