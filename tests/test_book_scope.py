from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medical_rag import bridge
from medical_rag.library import Library
from medical_rag.qa import match_book, strip_book_mention

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
