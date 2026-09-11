from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .doctor import check_environment, compact, format_report, repair
from .library import Library


def _human_mb(size: int) -> str:
    return f"{size / 1024 / 1024:.2f} MB"


def _run_doctor(args: argparse.Namespace) -> None:
    """维护动作（--repair / --compact）优先，否则跑环境自检。"""
    if args.repair or args.compact:
        exit_code = 0
        if args.repair:
            result = repair(args.root, args.db)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif result.get("repaired"):
                print(f"FTS 索引已重建：{result['chunks']} 个证据块，一致性检查通过。")
            else:
                print(f"未修复：{result.get('reason') or '重建后仍不一致'}")
            exit_code = 0 if result.get("repaired") else 1
        if args.compact:
            result = compact(args.root, args.db)
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif result.get("compacted"):
                print(
                    f"索引库已压缩：{_human_mb(result['before_bytes'])} → "
                    f"{_human_mb(result['after_bytes'])}（释放 {_human_mb(result['saved_bytes'])}），"
                    f"{result['chunks']} 个证据块，一致性 {'通过' if result['fts_integrity'] else '未通过'}"
                )
            else:
                print(f"未压缩：{result.get('reason')}")
            if not result.get("compacted"):
                exit_code = 1
        raise SystemExit(exit_code)
    report = check_environment(args.root, args.db, deep=args.deep)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    raise SystemExit(0 if report["ok"] else 1)


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
    doctor = sub.add_parser("doctor", help="环境自检（解释器、依赖、索引库、工作区）")
    doctor.add_argument("--root", type=Path, help="项目根目录（默认自动定位）")
    doctor.add_argument("--db", type=Path, help="library.sqlite3 路径")
    doctor.add_argument("--deep", action="store_true", help="额外启动子解释器验证 OCR/Paddle 依赖可导入（较慢）")
    doctor.add_argument("--repair", action="store_true", help="重建 FTS 索引，修复 chunks 与 chunks_fts 不一致")
    doctor.add_argument(
        "--compact",
        action="store_true",
        help="压缩索引库（FTS optimize + VACUUM），回收重建索引后累积的空闲页",
    )
    doctor.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()

    if args.command == "doctor":
        _run_doctor(args)

    library = Library()
    try:
        _dispatch(args, library)
    finally:
        library.close()


def _dispatch(args: argparse.Namespace, library: Library) -> None:
    if args.command == "ingest":
        print(library.ingest(args.pdf, args.title))
    elif args.command == "search":
        for item in library.search(args.query, args.limit, book=getattr(args, "book", None)):
            score = f" · score={item['score']}" if item.get('score') is not None else ""
            print(f"[{item['book']} · 第{item['page']}页 · {item['section']}{score}]\n{item['text']}\nchunk_id={item['chunk_id']}\n")
    elif args.command == "answer":
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
