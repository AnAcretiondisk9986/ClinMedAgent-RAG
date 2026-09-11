from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medical_rag.library import Library
from medical_rag.qa import extract_concepts, plan_question


class QuestionPlanningTests(unittest.TestCase):
    def test_extracts_domain_terms_from_exam_wording(self) -> None:
        self.assertEqual(extract_concepts("请简述肩关节的组成和特点"), ["肩关节"])
        self.assertEqual(extract_concepts("心脏的位置和外形"), ["心脏", "外形"])

    def test_strips_trailing_connectives_without_breaking_terms(self) -> None:
        self.assertEqual(extract_concepts("肩关节由哪些结构组成？有什么特点？"), ["肩关节"])
        self.assertEqual(extract_concepts("自由基对细胞的影响"), ["自由基对细胞", "影响"])

    def test_classifies_choice_and_definition(self) -> None:
        choice = plan_question("关于骨的描述，正确的是：\nA. 长骨\nB. 短骨")
        self.assertEqual(choice.question_type, "choice")
        self.assertEqual([item["label"] for item in choice.options], ["A", "B"])
        self.assertEqual(plan_question("什么是骨膜？").question_type, "definition")


class LibraryAnswerTests(unittest.TestCase):
    def test_answer_question_returns_citations_and_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            (structured / "01-骨学.md").write_text(
                "# 第一章 骨学\n\n## 原书第 18 页\n\n骨主要由骨质、骨膜和骨髓构成。\n\n"
                "## 原书第 19 页\n\n骨质分为骨密质和骨松质。\n",
                encoding="utf-8",
            )
            lib = Library(root / "library.sqlite3")
            try:
                report = lib.ingest_markdown_tree(root, "测试教材")
                self.assertEqual(report.pages, 19)
                result = lib.answer_question("请简述骨的构造", limit=2)
                self.assertEqual(result["status"], "evidence_found")
                self.assertTrue(result["evidence"])
                self.assertEqual(result["evidence"][0]["book"], "测试教材")
                self.assertEqual(result["evidence"][0]["page"], 18)
                self.assertTrue(result["evidence"][0]["context"])
            finally:
                lib.close()


class LibraryIngestTests(unittest.TestCase):
    def test_staging_dirs_are_not_indexed(self) -> None:
        """structured/ 缺失时的 rglob 兜底必须跳过 .staging-/.backup- 目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / ".staging-abc" / "structured"
            stale.mkdir(parents=True)
            (stale / "01-陈旧.md").write_text(
                "# 第一章 陈旧\n\n## 原书第 1 页\n\n这是上一轮生成残留的过期内容。\n",
                encoding="utf-8",
            )
            backup = root / ".backup-xyz"
            backup.mkdir()
            (backup / "01-旧.md").write_text(
                "# 第一章 旧\n\n## 原书第 1 页\n\n备份目录里的内容。\n", encoding="utf-8"
            )
            # 正式 structured/ 不存在 → 触发 rglob 兜底分支
            cleaned = root / "cleaned"
            cleaned.mkdir()
            (cleaned / "page-0001.md").write_text(
                "# PDF第 1 页\n\n骨骼肌由肌腹和肌腱构成。\n", encoding="utf-8"
            )
            lib = Library(root / "library.sqlite3")
            try:
                lib.ingest_markdown_tree(root, "测试教材")
                texts = [row[0] for row in lib.cx.execute("SELECT text FROM chunks").fetchall()]
            finally:
                lib.close()
            self.assertTrue(any("骨骼肌" in text for text in texts))
            self.assertFalse(any("过期内容" in text for text in texts))
            self.assertFalse(any("备份目录" in text for text in texts))

    def test_inner_headings_become_sections_and_chunk_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            (structured / "02-关节学.md").write_text(
                "# 第二章 关节学\n\n## 原书第 32 页\n\n### 第一节 总论\n\n"
                "骨与骨之间借纤维结缔组织、软骨和骨相连结，称骨连结。\n\n"
                "#### 一、直接连结\n\n"
                "骨与骨之间借纤维结缔组织或软骨直接相连，连结之间无间隙。\n",
                encoding="utf-8",
            )
            lib = Library(root / "library.sqlite3")
            try:
                lib.ingest_markdown_tree(root, "测试教材")
                rows = lib.cx.execute("SELECT section, text FROM chunks ORDER BY rowid").fetchall()
                self.assertTrue(any("第一节 总论" in r["section"] for r in rows))
                self.assertTrue(any("一、直接连结" in r["section"] for r in rows))
                # 标题作为新块的开头，且标题文本可被检索
                self.assertTrue(rows[0]["text"].startswith("第一节 总论"))
                self.assertTrue(rows[1]["text"].startswith("一、直接连结"))
            finally:
                lib.close()


if __name__ == "__main__":
    unittest.main()
