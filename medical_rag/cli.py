from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .library import Library


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="本地医学教材检索库")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="导入或重新索引 PDF")
    ingest.add_argument("pdf", type=Path)
    ingest.add_argument("--title")
    search = sub.add_parser("search", help="检索本地教材证据")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--book", help="限定教材：id、书名或缩写（如 组胚）")
    answer = sub.add_parser("answer", help="分析题目并整理教材证据（由 Agent 生成最终答案）")
    answer.add_argument("question")
    answer.add_argument("--limit", type=int, default=6)
    answer.add_argument("--book", help="限定教材：id、书名或缩写；不填则自动识别题干点名的教材")
    answer.add_argument("--json", action="store_true", help="以 JSON 输出证据包")
    it = sub.add_parser("ingest-text", help="索引处理后的 Markdown 目录")
    it.add_argument("directory", type=Path)
    it.add_argument("--title")
    sub.add_parser("list", help="列出已导入书籍")
    args = parser.parse_args()
    library = Library()
    if args.command == "ingest":
        print(library.ingest(args.pdf, args.title))
    elif args.command == "search":
        for item in library.search(args.query, args.limit, book=getattr(args, "book", None)):
            score = f" · score={item['score']}" if item.get('score') is not None else ""
            print(f"[{item['book']} · 第{item['page']}页 · {item['section']}{score}]\n{item['text']}\nchunk_id={item['chunk_id']}\n")
    elif args.command == "answer":
        import json
        result = library.answer_question(args.question, args.limit, book=getattr(args, "book", None))
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            plan = result.get("plan", {})
            print(f"题型：{plan.get('question_type', '未知')}\n核心概念：{'、'.join(plan.get('concepts', [])) or '未识别'}")
            scope = result.get("book_scope")
            if scope:
                origin = "题干点名" if scope.get("source") == "question" else "显式指定"
                print(f"检索范围：《{scope['title']}》（{origin}）")
            print(f"状态：{result.get('status')}\n作答提示：{result.get('answer_guidance', '')}\n")
            if not result.get("evidence"):
                print("知识库没有检索到足够证据。")
            for item in result.get("evidence", []):
                print(f"[{item['book']} · 第{item['page']}页 · {item['section']} · score={item['retrieval_score']}]\n{item['text']}\nchunk_id={item['chunk_id']}\n")
    elif args.command == "ingest-text":
        print(library.ingest_markdown_tree(args.directory, args.title))
    elif args.command == "list":
        for book in library.list_books():
            print(f"{book['title']} · {book['pages']}页 · 可提取文字 {book['extractable_pages']}页 · 图片页 {book['image_only_pages']}页\n  {book['path']}")


if __name__ == "__main__":
    main()
