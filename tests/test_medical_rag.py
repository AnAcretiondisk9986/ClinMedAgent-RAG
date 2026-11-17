from __future__ import annotations

import json
import os
import subprocess
import sys
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


class FtsIndexConsistencyTests(unittest.TestCase):
    """chunks_fts 是 external-content FTS5 表，重建后旧词条必须消失。"""

    @staticmethod
    def _hits(lib: Library, term: str) -> int:
        return int(
            lib.cx.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (term,)
            ).fetchone()[0]
        )

    def _make_book(self, root: Path, text: str) -> Path:
        structured = root / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        chapter = structured / "01-ch.md"
        chapter.write_text(f"# Ch1\n\n## 原书第 10 页\n\n{text}\n", encoding="utf-8")
        return chapter

    def test_reingest_removes_stale_fts_terms(self) -> None:
        """重建索引后，只存在于旧版本的词不能再被 FTS 命中（防幽灵结果）。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chapter = self._make_book(root, "periosteum contains vessels zzzold")
            lib = Library(root / "library.sqlite3")
            try:
                lib.ingest_markdown_tree(root, "测试教材")
                self.assertEqual(self._hits(lib, "zzzold"), 1)

                chapter.write_text(
                    "# Ch1\n\n## 原书第 10 页\n\nbone substance is dense zzznew\n",
                    encoding="utf-8",
                )
                lib.ingest_markdown_tree(root, "测试教材")

                self.assertEqual(self._hits(lib, "zzzold"), 0)
                self.assertEqual(self._hits(lib, "periosteum"), 0)
                self.assertEqual(self._hits(lib, "vessels"), 0)
                self.assertEqual(self._hits(lib, "zzznew"), 1)
                self.assertTrue(lib.fts_integrity_ok())
            finally:
                lib.close()

    def test_rebuild_fts_index_repairs_drift(self) -> None:
        """模拟历史版本留下的一致性损坏，rebuild 应能修复。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_book(root, "periosteum vessels zzzold")
            lib = Library(root / "library.sqlite3")
            try:
                lib.ingest_markdown_tree(root, "测试教材")
                # 绕过 FTS 维护直接改 chunks，制造漂移
                lib.cx.execute("UPDATE chunks SET text = 'bone substance dense zzzedited'")
                lib.cx.commit()
                self.assertFalse(lib.fts_integrity_ok())

                lib.rebuild_fts_index()

                self.assertTrue(lib.fts_integrity_ok())
                self.assertEqual(self._hits(lib, "zzzedited"), 1)
                self.assertEqual(self._hits(lib, "zzzold"), 0)
            finally:
                lib.close()

    def test_two_connections_rebuild_same_book(self) -> None:
        """两个连接（模拟两个进程）先后重建同一本书：不重复建书、索引保持一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_book(root, "alpha beta gamma")
            db = root / "library.sqlite3"
            first, second = Library(db), Library(db)
            try:
                first.ingest_markdown_tree(root, "测试教材")
                second.ingest_markdown_tree(root, "测试教材")
                self.assertEqual(len(first.list_books()), 1)
                self.assertEqual(
                    first.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1
                )
                self.assertTrue(first.fts_integrity_ok())
            finally:
                first.close()
                second.close()


class McpLimitTests(unittest.TestCase):
    """F2：MCP 客户端传 ``{"limit": null}`` 时必须回正常工具结果。

    以前 ``mcp_server`` 把原始 JSON 值直塞给 ``Library.search()``，
    ``int(None)`` 变成 ``isError`` 工具结果，客户端只看得到一条 TypeError。
    走真实子进程 + stdio，顺带覆盖 main() 的读写循环。
    """

    def _run_mcp(self, db: Path, requests: list[dict]) -> list[dict]:
        payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in requests) + "\n"
        env = {**os.environ, "MEDICAL_RAG_DB": str(db)}
        process = subprocess.run(
            [sys.executable, "-m", "medical_rag.mcp_server"],
            input=payload,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=180,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(process.returncode, 0, process.stderr[-1000:])
        return [json.loads(line) for line in process.stdout.splitlines() if line.strip()]

    def test_limit_null_is_a_normal_tool_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            structured = root / "structured"
            structured.mkdir()
            (structured / "01.md").write_text(
                "# 第一章\n\n## 原书第 18 页\n\n上皮组织由密集排列的上皮细胞和少量细胞外基质组成。\n",
                encoding="utf-8",
            )
            db = root / "library.sqlite3"
            lib = Library(db)
            try:
                lib.ingest_markdown_tree(root, "测试教材")
            finally:
                lib.close()
            responses = self._run_mcp(
                db,
                [
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": "search_medical_books",
                            "arguments": {"query": "上皮组织", "limit": None},
                        },
                    },
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {
                            "name": "answer_medical_question",
                            "arguments": {"question": "上皮组织由什么组成", "limit": None},
                        },
                    },
                ],
            )
        by_id = {item.get("id"): item for item in responses}
        search = by_id[2]["result"]
        self.assertNotIn("isError", search, f"limit=null 变成了工具错误：{search}")
        self.assertTrue(search["structuredContent"])
        self.assertEqual(search["structuredContent"][0]["page"], 18)
        answer = by_id[3]["result"]
        self.assertNotIn("isError", answer, f"limit=null 变成了工具错误：{answer}")
        self.assertEqual(answer["structuredContent"]["status"], "evidence_found")


if __name__ == "__main__":
    unittest.main()
