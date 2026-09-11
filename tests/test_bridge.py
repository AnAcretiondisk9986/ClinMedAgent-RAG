from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from medical_rag import bridge
from medical_rag.library import Library

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_library(root: Path) -> Library:
    structured = root / "structured"
    structured.mkdir()
    (structured / "01-骨学.md").write_text(
        "# 第一章 骨学\n\n## 原书第 18 页\n\n骨主要由骨质、骨膜和骨髓构成。\n\n"
        "## 原书第 19 页\n\n骨质分为骨密质和骨松质。\n",
        encoding="utf-8",
    )
    library = Library(root / "library.sqlite3")
    library.ingest_markdown_tree(root, "测试教材")
    return library


class BridgeHandleTests(unittest.TestCase):
    def test_search_get_chunk_and_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _make_library(Path(tmp))
            try:
                search = bridge.handle({"action": "search", "query": "骨的构造", "limit": 2}, library)
                self.assertTrue(search["ok"])
                self.assertTrue(search["data"])
                chunk_id = search["data"][0]["chunk_id"]

                chunk = bridge.handle({"action": "get_chunk", "chunk_id": chunk_id}, library)
                self.assertTrue(chunk["ok"])
                self.assertEqual(chunk["data"]["page"], 18)

                answer = bridge.handle({"action": "answer", "question": "请简述骨的构造", "limit": 2}, library)
                self.assertTrue(answer["ok"])
                self.assertEqual(answer["data"]["status"], "evidence_found")
            finally:
                library.close()

    def test_ping_reports_database_and_books(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _make_library(Path(tmp))
            try:
                ping = bridge.handle({"action": "ping"}, library)
                self.assertTrue(ping["ok"])
                self.assertEqual(ping["data"]["books"], 1)
                self.assertIn("library.sqlite3", ping["data"]["database"])
            finally:
                library.close()

    def test_rejects_unknown_action_and_empty_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _make_library(Path(tmp))
            try:
                unknown = bridge.handle({"action": "nope"}, library)
                self.assertFalse(unknown["ok"])
                self.assertIn("未知 action", unknown["error"])
                empty = bridge.handle({"action": "search", "query": "  "}, library)
                self.assertFalse(empty["ok"])
                missing = bridge.handle({"action": "get_chunk", "chunk_id": "deadbeef"}, library)
                self.assertFalse(missing["ok"])
            finally:
                library.close()


class BridgeProcessTests(unittest.TestCase):
    def test_subprocess_round_trip_keeps_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            library = _make_library(Path(tmp))
            database = str(library.db)
            library.close()
            request = json.dumps(
                {"action": "search", "query": "骨的构造", "limit": 1, "db": database},
                ensure_ascii=False,
            )
            result = subprocess.run(
                [sys.executable, "-m", "medical_rag.bridge"],
                input=request + "\n",
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=PROJECT_ROOT,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["data"][0]["book"], "测试教材")
            self.assertIn("骨主要由骨质", payload["data"][0]["text"])


if __name__ == "__main__":
    unittest.main()
