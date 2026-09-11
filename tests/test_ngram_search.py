from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medical_rag.library import Library, _normalise


def bigram_match(query: str) -> str:
    """把查询转成 OR-of-bigrams 的 FTS5 表达式（与 _candidate_rows 一致）。"""
    normalized = _normalise(query)
    grams = sorted({normalized[i:i + 2] for i in range(max(0, len(normalized) - 1))})
    return " OR ".join(f'"{gram}"' for gram in grams if gram.strip())


def make_library(root: Path) -> Library:
    specs = (
        (
            "系统解剖学",
            "系统解剖学 第5版",
            (
                (10, "骨主要由骨质、骨膜和骨髓构成。"),
                (11, "骨膜含有丰富的血管、神经和淋巴管。"),
                (12, "关节由关节面、关节囊和关节腔构成。"),
            ),
        ),
        (
            "组织学与胚胎学",
            "组织学与胚胎学 第10版",
            (
                (20, "上皮组织由密集排列的上皮细胞和少量细胞外基质组成。"),
                (21, "结缔组织由细胞和大量细胞外基质构成。"),
            ),
        ),
    )
    for folder, _title, pages in specs:
        structured = root / folder / "processed_v3" / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        body = "".join(f"## 原书第 {page} 页\n\n{text}\n\n" for page, text in pages)
        (structured / "01-章节.md").write_text(f"# 第一章\n\n{body}", encoding="utf-8")
    library = Library(root / "library.sqlite3")
    library.ingest_markdown_tree(root / "系统解剖学" / "processed_v3", "系统解剖学 第5版")
    library.ingest_markdown_tree(root / "组织学与胚胎学" / "processed_v3", "组织学与胚胎学 第10版")
    return library


def force_full_scan(lib: Library):
    """把 _candidate_rows 换成永远全表扫描，用于对比召回是否一致。"""

    def rows(q: str, tokens: list[str], book_ids: set[int] | None = None):
        sql = (
            "SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book,b.path,"
            "NULL AS norm_text FROM chunks c JOIN books b ON b.id=c.book_id"
        )
        if book_ids:
            placeholders = ",".join("?" for _ in book_ids)
            return lib.cx.execute(
                f"{sql} WHERE c.book_id IN ({placeholders})", tuple(sorted(book_ids))
            ).fetchall()
        return lib.cx.execute(sql).fetchall()

    return rows


class NgramIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_ngram_"))
        self.db = self.tmp / "library.sqlite3"
        self.library = make_library(self.tmp)

    def tearDown(self) -> None:
        self.library.close()

    def _count(self, sql: str, params: tuple = ()) -> int:
        return int(self.library.cx.execute(sql, params).fetchone()[0])

    def test_unicode61_cannot_match_chinese_but_bigram_index_can(self) -> None:
        """这是引入二元组索引的根本原因，必须记录下来。"""
        match = bigram_match("上皮组织")
        plain = self._count("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?", (match,))
        ngram = self._count(
            "SELECT COUNT(*) FROM chunks_fts_ngram WHERE bigrams MATCH ?", (match,)
        )
        self.assertEqual(plain, 0)  # 连续中文被当成一个 token，子串查不中
        self.assertGreaterEqual(ngram, 1)  # 二元组切成独立 token 后可命中

    def test_every_chunk_is_indexed(self) -> None:
        chunks = self._count("SELECT COUNT(*) FROM chunks")
        indexed = self._count("SELECT COUNT(*) FROM chunks_fts_ngram")
        self.assertEqual(chunks, indexed)
        self.assertTrue(self.library.fts_integrity_ok())

    def test_search_finds_chinese_substring(self) -> None:
        hits = self.library.search("上皮组织由什么组成", 3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["page"], 20)
        self.assertIn("上皮组织", hits[0]["text"])

    def test_search_respects_book_scope(self) -> None:
        histology = next(
            row for row in self.library.list_books() if "组织学" in row["title"]
        )
        self.assertTrue(self.library.search("上皮细胞", 3, book=histology["id"]))
        self.assertEqual(
            self.library.search("关节囊", 3, book=histology["id"]), []
        )

    def test_prefilter_narrows_candidates(self) -> None:
        total = self._count("SELECT COUNT(*) FROM chunks")
        rows = self.library._candidate_rows("关节囊由什么构成", ["关节囊"], None)
        self.assertLess(len(rows), total)
        self.assertTrue(rows)

    def test_single_char_token_falls_back_to_full_scan(self) -> None:
        """单字词无法用二元组表达，必须回退全表扫描，否则召回下降。"""
        rows = self.library._candidate_rows(_normalise("骨"), [_normalise("骨")], None)
        self.assertEqual(len(rows), self._count("SELECT COUNT(*) FROM chunks"))

    def test_prefilter_preserves_recall(self) -> None:
        for query in (
            "上皮组织由什么组成",
            "骨膜的构成",
            "关节囊和关节腔",
            "结缔组织的细胞外基质",
            "淋巴管",
        ):
            with self.subTest(query=query):
                indexed = [(item["chunk_id"], item["score"]) for item in self.library.search(query, 5)]
                with mock.patch.object(self.library, "_candidate_rows", force_full_scan(self.library)):
                    scanned = [
                        (item["chunk_id"], item["score"]) for item in self.library.search(query, 5)
                    ]
                self.assertEqual(indexed, scanned)

    def test_reindex_does_not_duplicate_ngram_rows(self) -> None:
        before = self._count("SELECT COUNT(*) FROM chunks_fts_ngram")
        for _ in range(3):
            self.library.ingest_markdown_tree(
                self.tmp / "系统解剖学" / "processed_v3", "系统解剖学 第5版"
            )
        self.assertEqual(self._count("SELECT COUNT(*) FROM chunks_fts_ngram"), before)
        self.assertTrue(self.library.fts_integrity_ok())


class NgramMaintenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_ngram_fix_"))
        self.db = self.tmp / "library.sqlite3"
        self.library = make_library(self.tmp)

    def tearDown(self) -> None:
        self.library.close()

    def _count(self, sql: str) -> int:
        return int(self.library.cx.execute(sql).fetchone()[0])

    def test_legacy_database_backfills_ngram_index(self) -> None:
        """老库（只有 chunks、没有二元组索引）打开时应自动回填。"""
        chunks = self._count("SELECT COUNT(*) FROM chunks")
        self.library.cx.execute("DELETE FROM chunks_fts_ngram")
        self.library.cx.commit()
        self.assertEqual(self._count("SELECT COUNT(*) FROM chunks_fts_ngram"), 0)
        self.library.close()

        reopened = Library(self.db)
        try:
            self.assertEqual(self._count_count(reopened), chunks)
            self.assertTrue(reopened.fts_integrity_ok())
        finally:
            reopened.close()

    @staticmethod
    def _count_count(lib: Library) -> int:
        return int(lib.cx.execute("SELECT COUNT(*) FROM chunks_fts_ngram").fetchone()[0])

    def test_integrity_detects_missing_ngram_rows(self) -> None:
        self.assertTrue(self.library.fts_integrity_ok())
        self.library.cx.execute(
            "DELETE FROM chunks_fts_ngram WHERE rowid = (SELECT MIN(rowid) FROM chunks)"
        )
        self.library.cx.commit()
        self.assertFalse(self.library.fts_integrity_ok())

    def test_rebuild_restores_ngram_index(self) -> None:
        self.library.cx.execute("DELETE FROM chunks_fts_ngram")
        self.library.cx.commit()
        self.assertFalse(self.library.fts_integrity_ok())

        self.library.rebuild_fts_index()

        self.assertTrue(self.library.fts_integrity_ok())
        self.assertTrue(self.library.search("上皮组织", 3))


if __name__ == "__main__":
    unittest.main()
