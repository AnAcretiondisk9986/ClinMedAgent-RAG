from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

import fitz

from medical_rag import pdftext


def make_text_pdf(path: Path, pages: int = 10, front_pages: int = 2) -> None:
    """生成带页眉、章节标题和页脚页码的文字层 PDF；前 front_pages 页无页码。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    content_pages = pages - front_pages
    for number in range(1, pages + 1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 60), "系统解剖学 第一篇 运动系统", fontname="china-s", fontsize=9)
        if number <= front_pages:
            page.insert_text((72, 240), f"前置页 {number}", fontname="china-s", fontsize=12)
            continue
        chapter = "第一章 骨骼肌" if number <= front_pages + content_pages // 2 else "第二章 关节学"
        page.insert_text((72, 110), chapter, fontname="china-s", fontsize=18)
        for row in range(6):
            page.insert_text(
                (72, 160 + row * 26),
                "肌由肌腹和肌腱构成，关节由关节面、关节囊和关节腔构成。",
                fontname="china-s",
                fontsize=12,
            )
        page.insert_text((300, 800), str(number - front_pages), fontsize=10)
    doc.save(path)
    doc.close()


def make_scan_pdf(path: Path, pages: int = 2) -> None:
    """只有图形、没有任何文字的图片型 PDF。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    for number in range(pages):
        page = doc.new_page(width=595, height=842)
        page.draw_rect(fitz.Rect(60, 60, 535, 780), color=(0, 0, 0), width=1)
        page.draw_line(fitz.Point(80, 140 + number * 40), fitz.Point(480, 150 + number * 40), color=(0, 0, 0), width=2)
    doc.save(path)
    doc.close()


class AnalyzeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_pdftext_test_"))

    def test_text_pdf_is_detected(self) -> None:
        pdf = self.tmp / "text.pdf"
        make_text_pdf(pdf, pages=10)
        info = pdftext.analyze(pdf)
        self.assertEqual(info["kind"], "text")
        self.assertEqual(info["pages"], 10)
        # 前 2 页是只有标题的前置页（文字不足 30 字），不计入“有文字页”
        self.assertEqual(info["text_pages"], 8)
        self.assertGreater(info["chars"], 500)
        self.assertIn("文字层", pdftext.kind_label(info["kind"]))

    def test_scan_pdf_is_detected(self) -> None:
        pdf = self.tmp / "scan.pdf"
        make_scan_pdf(pdf)
        info = pdftext.analyze(pdf)
        self.assertEqual(info["kind"], "scan")
        self.assertEqual(info["text_pages"], 0)
        self.assertIn("OCR", pdftext.recommendation(info))

    def test_cache_reuses_and_invalidates(self) -> None:
        pdf = self.tmp / "text.pdf"
        make_text_pdf(pdf, pages=4)
        cache = self.tmp / ".cache"
        first = pdftext.cached_analyze(pdf, cache)
        self.assertEqual(len(list(cache.glob("*.json"))), 1)
        second = pdftext.cached_analyze(pdf, cache)
        self.assertEqual(first, second)

        newer = time.time() + 5
        os.utime(pdf, (newer, newer))
        third = pdftext.cached_analyze(pdf, cache)
        self.assertNotEqual(third["checked_at"], first["checked_at"])
        self.assertAlmostEqual(third["mtime"], newer, places=0)


class BuildStructuredTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_pdftext_build_"))

    def test_build_with_offset_chapters_and_headers(self) -> None:
        pdf = self.tmp / "book.pdf"
        make_text_pdf(pdf, pages=10)
        book_dir = self.tmp / "res" / "测试教材"
        with contextlib.redirect_stdout(io.StringIO()) as output:
            quality = pdftext.build_structured(pdf, book_dir)

        self.assertEqual(quality["pages"], 10)
        self.assertEqual(quality["page_offset"], 2)
        self.assertTrue(quality["pipeline"].startswith("text-layer"))
        chapters = quality["chapters"]
        self.assertEqual(
            [(chapter["num"], chapter["pdf_start"], chapter["title"]) for chapter in chapters],
            [(1, 3, "骨骼肌"), (2, 7, "关节学")],
        )
        self.assertEqual(chapters[0]["printed_start"], 1)
        self.assertEqual(chapters[0]["printed_end"], 4)

        structured = sorted((book_dir / "processed_v3" / "structured").glob("*.md"))
        self.assertEqual([path.name for path in structured], ["01-骨骼肌.md", "02-关节学.md"])
        content = structured[0].read_text(encoding="utf-8")
        self.assertIn("## 原书第 1 页", content)
        self.assertNotIn("系统解剖学 第一篇 运动系统", content)  # 跨页页眉已剔除
        self.assertEqual(len(list((book_dir / "processed_v3" / "cleaned").glob("*.md"))), 10)

        quality_json = json.loads((book_dir / "processed_v3" / "quality.json").read_text(encoding="utf-8"))
        self.assertEqual(quality_json["page_offset"], 2)
        self.assertEqual(len(quality_json["chapters"]), 2)
        self.assertIn("10/10", output.getvalue())

    def test_cli_rejects_scan_pdf(self) -> None:
        pdf = self.tmp / "scan.pdf"
        make_scan_pdf(pdf)
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit):
                pdftext.main([str(pdf), str(self.tmp / "book")])

    def test_cli_detect_only_accepts_scan_pdf(self) -> None:
        pdf = self.tmp / "scan.pdf"
        make_scan_pdf(pdf)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            pdftext.main([str(pdf), str(self.tmp / "book"), "--detect-only"])
        self.assertIn('"kind": "scan"', output.getvalue())
        self.assertFalse((self.tmp / "book").exists())


if __name__ == "__main__":
    unittest.main()
