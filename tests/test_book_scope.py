from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medical_rag import bridge
from medical_rag.library import Library
from medical_rag.qa import (
    DEFAULT_BOOK_ALIASES,
    load_book_aliases,
    match_book,
    match_book_ex,
    strip_book_mention,
)

BOOKS = [
    {"id": 3, "title": "系统解剖学 第5版（OCR v3）", "path": "/data/res/系统解剖学/processed_v3"},
    {
        "id": 4,
        "title": "组织学与胚胎学 人卫 第10版",
        "path": "/data/res/组织学与胚胎学 人卫 第10版/processed_v3",
    },
]


def make_two_book_library(root: Path) -> Library:
    specs = (
        ("系统解剖学", "系统解剖学 第5版", "骨主要由骨质、骨膜和骨髓构成。"),
        ("组织学与胚胎学", "组织学与胚胎学 第10版", "上皮组织由密集排列的上皮细胞和少量细胞外基质组成。"),
    )
    for folder, title, text in specs:
        structured = root / folder / "processed_v3" / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        (structured / "01-章节.md").write_text(
            f"# 第一章\n\n## 原书第 10 页\n\n{text}\n", encoding="utf-8"
        )
    library = Library(root / "library.sqlite3")
    library.ingest_markdown_tree(root / "系统解剖学" / "processed_v3", "系统解剖学 第5版")
    library.ingest_markdown_tree(root / "组织学与胚胎学" / "processed_v3", "组织学与胚胎学 第10版")
    return library


class MatchBookTests(unittest.TestCase):
    def test_full_title_and_core_title(self) -> None:
        result = match_book("系统解剖学中骨的构造", BOOKS)
        self.assertIsNotNone(result)
        book, mention = result  # type: ignore[misc]
        self.assertEqual(book["id"], 3)
        self.assertEqual(mention, "系统解剖学")

    def test_core_title_ignores_version_and_publisher(self) -> None:
        result = match_book("《组织学与胚胎学》里的受精过程", BOOKS)
        self.assertIsNotNone(result)
        book, mention = result  # type: ignore[misc]
        self.assertEqual(book["id"], 4)
        self.assertEqual(mention, "组织学与胚胎学")

    def test_abbreviations(self) -> None:
        for text, expected in (("组胚里上皮组织如何分类", 4), ("系解中肩关节的组成", 3)):
            result = match_book(text, BOOKS)
            self.assertIsNotNone(result, text)
            book, mention = result  # type: ignore[misc]
            self.assertEqual(book["id"], expected)
            self.assertIn(mention, text)

    def test_ambiguous_or_absent_mention(self) -> None:
        self.assertIsNone(match_book("《系统解剖学》和《组织学与胚胎学》都讲了什么", BOOKS))
        self.assertIsNone(match_book("骨的构造", BOOKS))
        self.assertIsNone(match_book("", BOOKS))

    def test_strip_only_standalone_mentions(self) -> None:
        self.assertEqual(strip_book_mention("组胚里上皮组织如何分类", "组胚"), "上皮组织如何分类")
        self.assertEqual(strip_book_mention("系解中肩关节的组成", "系解"), "肩关节的组成")
        self.assertEqual(strip_book_mention("系统解剖学中骨的构造", "系统解剖学"), "骨的构造")
        self.assertEqual(strip_book_mention("《组织学与胚胎学》里的受精过程", "组织学与胚胎学"), "受精过程")
        # “组织”嵌在“上皮组织”里，不能把医学名词拆坏
        self.assertEqual(strip_book_mention("上皮组织有哪些分类", "组织"), "上皮组织有哪些分类")
        # “系统”嵌在“神经系统”里，保留原句
        self.assertEqual(strip_book_mention("神经系统的组成", "系统"), "神经系统的组成")


class AliasTighteningTests(unittest.TestCase):
    """旧实现把任意 2–4 字子序列当缩写，导致普通医学词错误限定教材范围。"""

    def test_generic_medical_words_are_not_aliases(self) -> None:
        # 审计发现的两个具体误判
        self.assertIsNone(match_book("神经系统的组成", BOOKS))
        self.assertIsNone(match_book("组织的分类", BOOKS))
        self.assertIsNone(match_book("人体系统的组成", BOOKS))

    def test_alias_embedded_in_a_word_is_not_matched(self) -> None:
        """“生理功能”里的“生理”、“免疫应答”里的“免疫”都不是书名缩写。"""
        physiology = [{"id": 9, "title": "生理学 第9版", "path": "/data/res/生理学/processed_v3"}]
        immunology = [{"id": 8, "title": "医学免疫学", "path": "/data/res/医学免疫学/processed_v3"}]
        self.assertIsNone(match_book("生理功能有哪些", physiology))
        self.assertIsNone(match_book("免疫应答的过程", immunology))
        # 但作为独立引用时仍然认得
        self.assertIsNotNone(match_book("生理里动作电位的产生", physiology))
        self.assertIsNotNone(match_book("《免疫》里抗体的结构", immunology))

    def test_explicit_alias_is_medium_confidence(self) -> None:
        match = match_book_ex("组胚里上皮组织如何分类", BOOKS)
        self.assertIsNotNone(match)
        self.assertEqual(match.book["id"], 4)
        self.assertEqual(match.confidence, "medium")
        self.assertEqual(match.mention, "组胚")

    def test_full_title_is_high_confidence(self) -> None:
        match = match_book_ex("系统解剖学中骨的构造", BOOKS)
        self.assertIsNotNone(match)
        self.assertEqual(match.confidence, "high")
        self.assertEqual(match.book["id"], 3)

    def test_alias_table_is_explicit_and_extensible(self) -> None:
        books = [{"id": 7, "title": "医学影像学 第3版", "path": "/data/res/医学影像学/processed_v3"}]
        # 默认表里没有“影像”这个缩写 → 不自动限定
        self.assertIsNone(match_book("影像里肺纹理增多", books))
        aliases = {**DEFAULT_BOOK_ALIASES, "影像": ("医学影像学",)}
        match = match_book_ex("影像里肺纹理增多", books, aliases)
        self.assertIsNotNone(match)
        self.assertEqual(match.confidence, "medium")

    def test_aliases_file_is_merged_over_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "aliases.json").write_text(
                json.dumps({"影像": ["医学影像学"], "口组": "口腔组织病理学"}, ensure_ascii=False),
                encoding="utf-8",
            )
            aliases = load_book_aliases(base)
            self.assertEqual(aliases["影像"], ("医学影像学",))
            self.assertEqual(aliases["口组"], ("口腔组织病理学",))  # 字符串也接受
            self.assertIn("组胚", aliases)  # 内置项保留

    def test_broken_aliases_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "aliases.json").write_text("{ not json", encoding="utf-8")
            self.assertEqual(load_book_aliases(base), DEFAULT_BOOK_ALIASES)

    def test_substring_book_titles_stay_ambiguous(self) -> None:
        """《生理学》是《病理生理学》的子串：两者都命中时宁可不限域、全库检索。"""
        books = [
            {"id": 1, "title": "生理学 第9版", "path": "/d/res/生理学/processed_v3"},
            {"id": 2, "title": "病理生理学 第3版", "path": "/d/res/病理生理学/processed_v3"},
        ]
        # 缩写是精确的：只有“病生”能指《病理生理学》，不会同时撞上《生理学》
        self.assertEqual(match_book_ex("病生里发热的机制", books).book["id"], 2)
        # 而完整书名“病理生理学”里嵌了“生理学”，两个书名同时命中 → 不强行路由
        self.assertIsNone(match_book_ex("病理生理学中发热的机制", books))
        # 单独只说《生理学》时仍能正常定位
        self.assertEqual(match_book_ex("生理学里动作电位", books).book["id"], 1)


class LibraryScopeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_scope_"))
        self.library = make_two_book_library(self.tmp)
        self.ids = {row["title"]: row["id"] for row in self.library.list_books()}

    def tearDown(self) -> None:
        self.library.close()

    def test_search_filters_by_book(self) -> None:
        anatomy_id = self.ids["系统解剖学 第5版"]
        histology_id = self.ids["组织学与胚胎学 第10版"]

        global_hits = self.library.search("上皮细胞", 5)
        self.assertTrue(global_hits)
        self.assertTrue(all("组织学" in item["book"] for item in global_hits))

        self.assertTrue(self.library.search("骨膜", 5, book=anatomy_id))
        self.assertTrue(self.library.search("骨膜", 5, book="系解"))
        self.assertEqual(self.library.search("骨膜", 5, book=histology_id), [])
        self.assertEqual(self.library.search("骨膜", 5, book="组胚"), [])
        self.assertTrue(self.library.search("上皮细胞", 5, book="组胚"))
        self.assertEqual(self.library.search("上皮细胞", 5, book="不存在的书"), [])

    def test_answer_question_scopes_to_named_book(self) -> None:
        histology_id = self.ids["组织学与胚胎学 第10版"]
        pack = self.library.answer_question("组胚里上皮组织的构成", 3)
        self.assertEqual(pack["status"], "evidence_found")
        self.assertEqual(pack["book_scope"]["book_id"], histology_id)
        self.assertEqual(pack["book_scope"]["source"], "question")
        self.assertEqual(pack["plan"]["question"], "上皮组织的构成")  # 书名已从题干剥离
        self.assertTrue(all("组织学" in item["book"] for item in pack["evidence"]))
        self.assertIn("组织学与胚胎学", pack["answer_guidance"])

    def test_answer_question_reports_scope_confidence(self) -> None:
        pack = self.library.answer_question("组胚里上皮组织的构成", 3)
        self.assertEqual(pack["book_scope"]["confidence"], "medium")
        self.assertIn("组胚", pack["book_scope"]["reason"])

        pack = self.library.answer_question("组织学与胚胎学中上皮组织的构成", 3)
        self.assertEqual(pack["book_scope"]["confidence"], "high")

    def test_answer_question_does_not_scope_on_generic_words(self) -> None:
        """“组织的分类”不得被限定成《组织学与胚胎学》，应保持全库检索。"""
        pack = self.library.answer_question("组织的分类有哪些", 5)
        self.assertNotIn("book_scope", pack)
        self.assertEqual(pack["plan"]["question"], "组织的分类有哪些")

    def test_answer_question_explicit_book(self) -> None:
        pack = self.library.answer_question("骨的构造", 3, book="组胚")
        self.assertEqual(pack["book_scope"]["source"], "explicit")
        self.assertEqual(pack["status"], "no_evidence")  # 组胚里没有骨的内容

    def test_answer_question_invalid_book(self) -> None:
        pack = self.library.answer_question("骨的构造", 3, book="不存在的书")
        self.assertEqual(pack["status"], "invalid_book")


class BridgeScopeTests(unittest.TestCase):
    def test_bridge_accepts_book_scope(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_bridge_scope_"))
        library = make_two_book_library(tmp)
        try:
            search = bridge.handle({"action": "search", "query": "上皮细胞", "book": "组胚", "limit": 2}, library)
            self.assertTrue(search["ok"])
            self.assertTrue(search["data"])
            self.assertTrue(all("组织学" in item["book"] for item in search["data"]))

            answer = bridge.handle({"action": "answer", "question": "组胚里上皮组织的构成", "limit": 2}, library)
            self.assertTrue(answer["ok"])
            self.assertEqual(answer["data"]["book_scope"]["source"], "question")
            self.assertIn("组织学", answer["data"]["book_scope"]["title"])
        finally:
            library.close()


if __name__ == "__main__":
    unittest.main()
