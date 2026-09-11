from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medical_rag.library import Library, _match_reasons, _proximity, _quality_penalty


def make_library(root: Path, pages: tuple[tuple[int, str], ...]) -> Library:
    structured = root / "book" / "processed_v3" / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    body = "".join(f"## 原书第 {page} 页\n\n{text}\n\n" for page, text in pages)
    (structured / "01-章节.md").write_text(f"# 第一章 测试\n\n{body}", encoding="utf-8")
    library = Library(root / "library.sqlite3")
    library.ingest_markdown_tree(root / "book" / "processed_v3", "测试教材")
    return library


class QualityPenaltyTests(unittest.TestCase):
    """目录页 / 图注 / 过短文本必须降权，否则会把正文挤下去。"""

    def test_toc_dotted_lines_are_penalised(self) -> None:
        penalty, reasons = _quality_penalty("骨膜 …… 12\n关节学 …… 20")
        self.assertLess(penalty, 1.0)
        self.assertTrue(any("目录页" in reason for reason in reasons))

    def test_figure_caption_is_penalised(self) -> None:
        penalty, reasons = _quality_penalty("[图注/图例] 图1 骨的结构示意图，示骨膜与骨密质。")
        self.assertLess(penalty, 1.0)
        self.assertTrue(any("图注" in reason for reason in reasons))

    def test_short_header_like_text_is_penalised(self) -> None:
        penalty, reasons = _quality_penalty("系统解剖学 第一篇 运动系统")
        self.assertLess(penalty, 1.0)
        self.assertTrue(any("过短" in reason for reason in reasons))

    def test_normal_prose_is_not_penalised(self) -> None:
        penalty, reasons = _quality_penalty(
            "骨膜含有丰富的血管、神经和淋巴管，对骨的营养和再生有重要作用。"
        )
        self.assertEqual(penalty, 1.0)
        self.assertEqual(reasons, [])


class ProximityTests(unittest.TestCase):
    def test_single_concept_has_no_proximity_signal(self) -> None:
        self.assertEqual(_proximity("骨膜含有血管", ["骨膜"]), 0.0)

    def test_close_concepts_score_higher_than_scattered(self) -> None:
        close = _proximity("骨膜含有丰富的血管和神经。", ["骨膜", "血管"])
        far = _proximity("骨膜" + "填充" * 200 + "血管", ["骨膜", "血管"])
        self.assertGreater(close, far)
        self.assertEqual(close, 1.0)
        self.assertEqual(far, 0.0)


class MatchReasonTests(unittest.TestCase):
    def test_reasons_include_signals_and_penalties(self) -> None:
        reasons = _match_reasons(
            exact=True,
            phrase_hits=["骨膜结构"],
            proximity=1.0,
            section_hits=1,
            heading_hit=True,
            matched=["骨膜"],
            penalties=["疑似目录页（点线引导）"],
        )
        joined = " / ".join(reasons)
        for expected in ("完整题干原样出现", "相邻概念命中", "概念集中出现", "章节标题匹配", "段落开头命中", "命中概念", "目录页"):
            self.assertIn(expected, joined)


class RankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_rank_"))
        # 10 页是目录式点线行（“骨膜”出现 3 次），11 页是真正正文（只出现 1 次）
        self.library = make_library(
            self.tmp,
            (
                (10, "骨膜 …… 12\n骨膜与骨的关系 …… 13\n骨膜的组织结构 …… 14"),
                (11, "骨膜含有丰富的血管、神经和淋巴管，对骨的营养和再生有重要作用。"),
            ),
        )

    def tearDown(self) -> None:
        self.library.close()

    def test_toc_page_ranks_below_real_prose(self) -> None:
        results = self.library.search("骨膜", 5)
        self.assertTrue(results)
        self.assertEqual(results[0]["page"], 11)
        toc = [item for item in results if item["page"] == 10]
        self.assertTrue(toc)
        self.assertTrue(any("目录页" in reason for reason in toc[0]["match_reason"]))

    def test_every_result_explains_why_it_matched(self) -> None:
        for item in self.library.search("骨膜", 5):
            self.assertIn("match_reason", item)
            self.assertTrue(item["match_reason"])
            self.assertTrue(any("命中概念" in reason for reason in item["match_reason"]))

    def test_internal_cache_field_is_not_exposed(self) -> None:
        for item in self.library.search("骨膜", 5):
            self.assertNotIn("norm_text", item)


class DedupTests(unittest.TestCase):
    def test_identical_chunks_are_deduplicated(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_dedup_"))
        duplicate = "骨膜含有丰富的血管、神经和淋巴管。"
        library = make_library(tmp, ((10, duplicate), (11, duplicate)))
        try:
            pack = library.answer_question("骨膜的构成", 5)
            texts = [item["text"] for item in pack["evidence"]]
            self.assertEqual(len(texts), len(set(texts)))
            self.assertEqual(len(texts), 1)
        finally:
            library.close()

    def test_contained_evidence_is_dropped(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_contain_"))
        library = make_library(
            tmp,
            (
                (10, "骨膜"),
                (11, "骨膜含有丰富的血管、神经和淋巴管，对骨的营养有重要作用。"),
            ),
        )
        try:
            evidence = library.answer_question("骨膜", 5)["evidence"]
            self.assertEqual(len(evidence), 1)
            self.assertIn("血管", evidence[0]["text"])  # 保留信息量更大的那条
        finally:
            library.close()

    def test_at_most_two_chunks_per_page(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_perpage_"))
        library = make_library(
            tmp,
            (
                (
                    10,
                    "骨膜含有丰富的血管。\n\n### 一、骨膜的结构\n\n骨膜分为内外两层。"
                    "\n\n### 二、骨膜的功能\n\n骨膜参与骨的生长和修复。",
                ),
            ),
        )
        try:
            evidence = library.answer_question("骨膜", 5)["evidence"]
            pages = [item["page"] for item in evidence]
            self.assertEqual(pages.count(10), 2)
        finally:
            library.close()


class AdjacentPageTests(unittest.TestCase):
    def test_adjacent_pages_are_annotated(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_adjacent_"))
        library = make_library(
            tmp,
            (
                (10, "骨膜分为骨外膜和骨内膜两层。"),
                (11, "骨膜的外层含有丰富的血管和神经。"),
            ),
        )
        try:
            evidence = library.answer_question("骨膜的分层", 5)["evidence"]
            self.assertGreaterEqual(len(evidence), 2)
            by_page = {item["page"]: item for item in evidence}
            self.assertEqual(set(by_page), {10, 11})
            # 不假设两条证据的相对顺序，只验证“相邻页互指”这个性质
            for page, item in by_page.items():
                self.assertEqual(item["adjacent_pages"], [11 if page == 10 else 10])
                self.assertTrue(any("相邻页" in reason for reason in item["match_reason"]))
                # 每条证据仍保留自己的页码与 chunk_id，保证引用可校验
                self.assertTrue(item["chunk_id"])
        finally:
            library.close()


if __name__ == "__main__":
    unittest.main()
