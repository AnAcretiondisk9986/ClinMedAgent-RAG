"""目录整体移动/复制后索引仍然可用的回归测试。

背景：老版本把 ``F:\\AI Agent检索知识库\\res\\...`` 这种绝对路径写进索引库，
项目目录一移动，书卡变“未索引”、按书检索报 400、doctor 报“源 PDF 不存在”。
现在库里只存相对项目根目录的路径，读取时按当前根目录还原；老库打开时自动迁移。
本测试真的把整个假项目目录搬到新位置，再验证索引、检索、书卡与重建索引都正常。
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import fitz

from medical_rag.library import Library
from medical_rag.workspace import scan_books


def make_pdf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=420, height=300)
    page.insert_text((40, 80), text, fontsize=14)
    doc.save(str(path))
    doc.close()


def make_book(root: Path, name: str, text: str) -> Path:
    """建一本最小教材：PDF（可提取文字）+ processed_v3/structured/*.md。"""
    book_dir = root / "res" / name
    make_pdf(book_dir / "PDF" / f"{name}.pdf", text)
    structured = book_dir / "processed_v3" / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    (structured / "01-第一章.md").write_text(
        f"# 第一章\n\n## 原书第 12 页\n\n{text}\n", encoding="utf-8"
    )
    return book_dir / "processed_v3"


class RelocationTests(unittest.TestCase):
    def setUp(self) -> None:
        # resolve()：Windows 的 TEMP 可能是 8.3 短路径（ANACRE~1），而 Library 内部
        # 会把根目录解析成长路径，测试里统一成长路径再比较
        self.tmp = Path(tempfile.mkdtemp(prefix="medical-rag-move-")).resolve()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build(self) -> tuple[Path, Path, int, list[str]]:
        """在“旧位置”建库，返回 (旧根目录, 新根目录, book_id, 移动前的 chunk_id 列表)。"""
        old_root = self.tmp / "旧位置" / "AI Agent检索知识库"
        structured = make_book(old_root, "系统解剖学", "骨由骨质、骨膜和骨髓构成。")
        db = old_root / ".medical_rag" / "library.sqlite3"
        lib = Library(db)
        try:
            lib.ingest_markdown_tree(structured, "系统解剖学")
            stored = lib.cx.execute("SELECT path, source_path FROM books").fetchone()
            self.assertFalse(Path(stored["path"]).is_absolute(), "索引里不应再存绝对路径")
            self.assertFalse(Path(stored["source_path"]).is_absolute(), "来源 PDF 也应是相对路径")
            book_id = lib.list_books()[0]["id"]
            chunk_ids = [
                row[0] for row in lib.cx.execute("SELECT chunk_id FROM chunks ORDER BY rowid")
            ]
        finally:
            lib.close()
        return old_root, self.tmp / "新位置" / "改名后的项目", book_id, chunk_ids

    def test_library_matches_and_searches_after_move(self) -> None:
        old_root, new_root, book_id, _ = self._build()
        new_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_root), str(new_root))
        self.assertFalse(old_root.exists())

        lib = Library(new_root / ".medical_rag" / "library.sqlite3")
        try:
            books = lib.list_books()
            self.assertEqual(len(books), 1)
            self.assertTrue(Path(books[0]["path"]).is_relative_to(new_root))
            self.assertTrue(Path(books[0]["source_path"]).exists())
            # 时效检查：目录移动不算“源 PDF 变化”
            self.assertFalse(lib.book_freshness(book_id)["stale"])
            self.assertEqual(lib.stale_books(), [])
            # 检索照常，返回的 path 指向新位置
            results = lib.search("骨质", limit=3)
            self.assertTrue(results)
            self.assertEqual(results[0]["path"], books[0]["path"])
        finally:
            lib.close()

        # 网页书库扫描：1 张卡、显示已索引，按书检索能拿到索引 id
        scanned = scan_books(new_root)
        self.assertEqual(len(scanned), 1)
        self.assertTrue(scanned[0]["status"]["index"]["complete"])
        self.assertEqual(scanned[0]["status"]["index"]["book_id"], book_id)

    def test_reindex_after_move_updates_instead_of_duplicating(self) -> None:
        old_root, new_root, book_id, chunk_ids = self._build()
        new_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_root), str(new_root))
        db = new_root / ".medical_rag" / "library.sqlite3"
        structured = new_root / "res" / "系统解剖学" / "processed_v3"

        lib = Library(db)
        try:
            lib.ingest_markdown_tree(structured, "系统解剖学")
            books = lib.list_books()
            self.assertEqual(len(books), 1, "移动后重建索引不应多出一本书")
            self.assertEqual(books[0]["id"], book_id)
            rebuilt = [
                row[0] for row in lib.cx.execute("SELECT chunk_id FROM chunks ORDER BY rowid")
            ]
            self.assertEqual(rebuilt, chunk_ids, "相对路径生成 chunk_id，重建后 id 不变")
        finally:
            lib.close()

    def test_legacy_absolute_paths_are_migrated(self) -> None:
        """模拟老版本库：把库里的路径改回旧绝对路径，再整体移动目录。"""
        old_root, new_root, book_id, _ = self._build()
        db = old_root / ".medical_rag" / "library.sqlite3"
        connection = sqlite3.connect(db)
        try:
            row = connection.execute(
                "SELECT path, source_path FROM books"
            ).fetchone()
            connection.execute(
                "UPDATE books SET path=?, source_path=?",
                (
                    str(old_root / Path(row[0])).replace("/", os.sep),
                    str(old_root / Path(row[1])).replace("/", os.sep),
                ),
            )
            connection.commit()
        finally:
            connection.close()

        new_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_root), str(new_root))

        lib = Library(new_root / ".medical_rag" / "library.sqlite3", root=new_root)
        try:
            stored = lib.cx.execute("SELECT path, source_path FROM books").fetchone()
            self.assertFalse(Path(stored["path"]).is_absolute(), "老绝对路径应被迁移成相对路径")
            self.assertTrue(Path(lib.list_books()[0]["source_path"]).exists())
            self.assertFalse(lib.book_freshness(book_id)["stale"])
        finally:
            lib.close()
        self.assertTrue(scan_books(new_root)[0]["status"]["index"]["complete"])


if __name__ == "__main__":
    unittest.main()
