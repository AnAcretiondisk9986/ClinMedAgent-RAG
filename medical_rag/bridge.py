"""JSON bridge for agent platforms that call CLI tools instead of MCP.

Reads one JSON request per line from stdin and writes one JSON response per
line to stdout. This is the transport used by the pi extension
(``.pi/extensions/medical-rag.ts``) and can be reused by any other harness.

Examples
--------
    echo '{"action":"search","query":"肩关节的组成","limit":3}' | python -m medical_rag.bridge
    echo '{"action":"answer","question":"请简述骨的构造"}' | python -m medical_rag.bridge

Request fields
--------------
action      ping | list_books | search | get_chunk | answer
db          optional path to a library.sqlite3 (defaults to the project DB)
book        optional textbook scope: id, title or abbreviation (search/answer)
query       search text (action=search)
question    exam question (action=answer)
chunk_id    stable chunk id (action=get_chunk)
limit       maximum number of evidence chunks

Response
--------
{"ok": true, "data": ...} or {"ok": false, "error": "..."}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .library import Library

VERSION = "0.2.0"
ACTIONS = ("ping", "list_books", "search", "get_chunk", "answer")


def _open_library(db: str | None) -> Library:
    return Library(Path(db).expanduser()) if db else Library()


def _positive_int(value: Any, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def _book_arg(request: dict[str, Any]) -> Any:
    """book 字段：支持 id（数字）或书名/缩写字符串；未提供时返回 None。"""
    book = request.get("book")
    if book is None or (isinstance(book, str) and not book.strip()):
        return None
    return book


def handle(request: dict[str, Any], library: Library | None = None) -> dict[str, Any]:
    """Execute one bridge request. Exposed for tests and in-process callers."""
    action = str(request.get("action") or "").strip()
    owned = library is None
    if library is None:
        library = _open_library(request.get("db"))
    try:
        if action == "ping":
            books = library.list_books()
            return {
                "ok": True,
                "data": {
                    "server": "medical-rag-bridge",
                    "version": VERSION,
                    "database": str(library.db),
                    "books": len(books),
                    "titles": [book["title"] for book in books],
                },
            }
        if action == "list_books":
            return {"ok": True, "data": library.list_books()}
        if action == "search":
            query = str(request.get("query") or "").strip()
            if not query:
                return {"ok": False, "error": "query 不能为空"}
            return {
                "ok": True,
                "data": library.search(
                    query, _positive_int(request.get("limit"), 5), book=_book_arg(request)
                ),
            }
        if action == "get_chunk":
            chunk_id = str(request.get("chunk_id") or "").strip()
            if not chunk_id:
                return {"ok": False, "error": "chunk_id 不能为空"}
            chunk = library.get_chunk(chunk_id)
            if chunk is None:
                return {"ok": False, "error": f"未找到 chunk_id={chunk_id}"}
            return {"ok": True, "data": chunk}
        if action == "answer":
            question = str(request.get("question") or "").strip()
            if not question:
                return {"ok": False, "error": "question 不能为空"}
            return {
                "ok": True,
                "data": library.answer_question(
                    question, _positive_int(request.get("limit"), 6), book=_book_arg(request)
                ),
            }
        return {"ok": False, "error": f"未知 action：{action or '(空)'}；可用：{', '.join(ACTIONS)}"}
    except Exception as exc:  # keep the bridge alive for the next request
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if owned:
            library.close()


def main() -> None:
    # Windows pipes default to a legacy code page; force UTF-8 end to end so
    # Chinese queries and citations survive the round trip.
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("请求必须是 JSON 对象")
            response = handle(request)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
