from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import fitz

from medical_rag.library import (
    BOOK_FINGERPRINT_COLUMNS,
    SCHEMA_VERSION,
    Library,
)
from medical_rag.workspace import scan_books

LEGACY_BOOKS_DDL = """
CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    pages INTEGER NOT NULL,
    extractable_pages INTEGER NOT NULL DEFAULT 0,
    image_only_pages INTEGER NOT NULL DEFAULT 0,
    added_at TEXT NOT NULL
);
CREATE TABLE chunks (
    rowid INTEGER PRIMARY KEY,
    chunk_id TEXT NOT NULL UNIQUE,
    book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    section TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', content_rowid='rowid');
"""


def make_pdf(path: Path, pages: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for index in range(pages):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), f"Anatomy page {index + 1}", fontsize=14)
    doc.save(str(path))
    doc.close()


def make_legacy_db(path: Path) -> None:
    """构造旧版本（无指纹列）的索引库。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(LEGACY_BOOKS_DDL)
        connection.execute(
            "INSERT INTO books(title,path,pages,extractable_pages,image_only_pages,added_at)"
            " VALUES('旧教材','/nowhere/processed_v3',10,10,0,'2026-01-01T00:00:00')"
        )
        connection.execute(
            "INSERT INTO chunks(chunk_id,book_id,page,section,text) VALUES('c1',1,1,'s','旧内容')"
        )
        connection.execute("INSERT INTO chunks_fts(rowid,text,section) VALUES(1,'旧内容','s')")
        connection.commit()
    finally:
        connection.close()


class SchemaMigrationTests(unittest.TestCase):
    def test_new_database_has_fingerprint_columns_and_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lib = Library(Path(tmp) / "new.sqlite3")
            try:
                columns = {row["name"] for row in lib.cx.execute("PRAGMA table_info(books)")}
                for name, _ in BOOK_FINGERPRINT_COLUMNS:
                    self.assertIn(name, columns)
                version = lib.cx.execute("PRAGMA user_version").fetchone()[0]
                self.assertEqual(version, SCHEMA_VERSION)
            finally:
                lib.close()

    def test_legacy_database_is_migrated_without_data_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "legacy.sqlite3"
            make_legacy_db(db)

            lib = Library(db)
            try:
                columns = {row["name"] for row in lib.cx.execute("PRAGMA table_info(books)")}
                for name, _ in BOOK_FINGERPRINT_COLUMNS:
                    self.assertIn(name, columns)
                self.assertEqual(
                    lib.cx.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
                )
                # 老数据保留
                self.assertEqual(len(lib.list_books()), 1)
                self.assertEqual(lib.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], 1)
                # FTS 仍可用
                self.assertTrue(lib.fts_integrity_ok())
            finally:
                lib.close()

    def test_migration_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "legacy.sqlite3"
            make_legacy_db(db)
            for _ in range(3):
                lib = Library(db)
                lib.close()


class FingerprintIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_fp_"))
        self.book_dir = self.tmp / "res" / "测试教材"
        self.pdf = self.book_dir / "PDF" / "测试教材.pdf"
        make_pdf(self.pdf, pages=2)
        self.out = self.book_dir / "processed_v3"
        structured = self.out / "structured"
        structured.mkdir(parents=True, exist_ok=True)
        (structured / "01-肌学.md").write_text(
            "# 第一章 肌学\n\n## 原书第 1 页\n\n骨骼肌由肌腹和肌腱构成。\n", encoding="utf-8"
        )
        (self.out / "quality.json").write_text(
            json.dumps(
                {
                    "pages": 2,
                    "page_offset": 12,
                    "pipeline": "text-layer（PyMuPDF 直接解析，未使用 OCR）",
                    "chapters": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        self.db = self.tmp / ".medical_rag" / "library.sqlite3"

    def _ingest(self) -> Library:
        lib = Library(self.db)
        lib.ingest_markdown_tree(self.out, "测试教材")
        return lib

    def test_markdown_ingest_records_source_and_quality_fingerprint(self) -> None:
        lib = self._ingest()
        try:
            row = lib.list_books()[0]
            # Windows 下 resolve() 可能返回 8.3 短名，比较文件身份而非字符串
            self.assertTrue(os.path.samefile(row["source_path"], self.pdf))
            self.assertEqual(row["source_size"], self.pdf.stat().st_size)
            self.assertAlmostEqual(row["source_mtime"], self.pdf.stat().st_mtime, places=3)
            self.assertEqual(len(row["source_sha256"]), 64)
            self.assertEqual(len(row["content_hash"]), 64)
            self.assertEqual(row["page_offset"], 12)
            self.assertTrue(os.path.samefile(row["quality_path"], self.out / "quality.json"))
            self.assertIn("text-layer", row["pipeline_version"])
            self.assertTrue(row["indexed_at"])
        finally:
            lib.close()

    def test_pdf_ingest_records_source_fingerprint(self) -> None:
        lib = Library(self.db)
        try:
            lib.ingest(self.pdf, "直抽教材")
            row = lib.list_books()[0]
            self.assertTrue(os.path.samefile(row["source_path"], self.pdf))
            self.assertEqual(len(row["source_sha256"]), 64)
            self.assertEqual(row["content_hash"], row["source_sha256"])
            self.assertTrue(row["indexed_at"])
        finally:
            lib.close()

    def test_reindex_updates_content_hash_and_indexed_at(self) -> None:
        lib = self._ingest()
        try:
            before = lib.list_books()[0]
            (self.out / "structured" / "01-肌学.md").write_text(
                "# 第一章 肌学\n\n## 原书第 1 页\n\n骨骼肌由肌腹和肌腱构成，共 600 余块。\n",
                encoding="utf-8",
            )
            lib.ingest_markdown_tree(self.out, "测试教材")
            after = lib.list_books()[0]
            self.assertNotEqual(before["content_hash"], after["content_hash"])
            # 仍是同一本书，不重复建记录
            self.assertEqual(len(lib.list_books()), 1)
            self.assertGreaterEqual(after["indexed_at"], before["indexed_at"])
        finally:
            lib.close()

    def test_freshness_is_clean_right_after_ingest(self) -> None:
        lib = self._ingest()
        try:
            info = lib.book_freshness(lib.list_books()[0]["id"])
            self.assertTrue(info["has_fingerprint"])
            self.assertFalse(info["stale"], info["reason"])
            self.assertEqual(info["pipeline_version"][:12], "text-layer（P")
        finally:
            lib.close()

    def test_freshness_detects_replaced_pdf(self) -> None:
        lib = self._ingest()
        try:
            book_id = lib.list_books()[0]["id"]
            make_pdf(self.pdf, pages=3)  # 换掉 PDF：mtime 与大小都变
            info = lib.book_freshness(book_id)
            self.assertTrue(info["stale"])
            self.assertIn("源 PDF 已变化", info["reason"])
            self.assertEqual([item["title"] for item in lib.stale_books()], ["测试教材"])
        finally:
            lib.close()

    def test_verify_source_hash_catches_same_size_change(self) -> None:
        """mtime/大小都没变时廉价检查看不出来，sha256 强校验必须能发现。"""
        lib = self._ingest()
        try:
            book_id = lib.list_books()[0]["id"]
            original_mtime = self.pdf.stat().st_mtime
            raw = bytearray(self.pdf.read_bytes())
            raw[-1] ^= 0xFF
            self.pdf.write_bytes(bytes(raw))
            os.utime(self.pdf, (original_mtime, original_mtime))  # mtime 也保持原样

            # 大小与 mtime 都没变，廉价检查看不出来
            self.assertFalse(lib.book_freshness(book_id)["stale"])

            verified = lib.verify_source_hash(book_id)
            self.assertFalse(verified["hash_matches"])
            self.assertTrue(verified["stale"])
            self.assertIn("sha256", verified["reason"])
        finally:
            lib.close()

    def test_freshness_flags_legacy_index_without_fingerprint(self) -> None:
        db = self.tmp / "legacy.sqlite3"
        make_legacy_db(db)
        lib = Library(db)
        try:
            info = lib.book_freshness(1)
            self.assertFalse(info["has_fingerprint"])
            self.assertTrue(info["stale"])
            self.assertIn("旧版本建立", info["reason"])
            self.assertEqual(len(lib.stale_books()), 1)
        finally:
            lib.close()

    def test_workspace_status_exposes_staleness(self) -> None:
        lib = self._ingest()
        lib.close()
        books = scan_books(self.tmp, self.db)
        self.assertEqual(len(books), 1)
        self.assertFalse(books[0]["status"]["index"]["stale"])
        self.assertIsNotNone(books[0]["status"]["index"]["indexed_at"])

        make_pdf(self.pdf, pages=3)
        books = scan_books(self.tmp, self.db)
        self.assertTrue(books[0]["status"]["index"]["stale"])
        self.assertIn("过期", books[0]["status"]["index"]["stale_reason"])


if __name__ == "__main__":
    unittest.main()
