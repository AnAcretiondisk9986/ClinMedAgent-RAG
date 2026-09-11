"""生成 v3 结构化 Markdown（支持任意教材）：

  - 非表格页：复用 tools_structure_v2 的 xy-cut 版面重建（从 <book>/text_v3/boxes 读 PP-OCRv6 结果）；
  - 表格页：直接采用 <book>/text_v3/tables/page-XXXX.md（PP-StructureV3 的 HTML 表格结果），
    并将其标题层级下移一级，以适配章节文件里的 “## 原书第 N 页” 分页标题；表格框内的文本框
    会先从正文重建中剔除，避免表格文字被重复收录；
  - 章节：优先读取 <book-dir>/chapters.json（人工校对或上一次检测结果），否则扫描章标题文本框
    自动检测；检测不到时按单章输出。跳过目录页（同页出现多个章标题）与点线目录行。

输出：<book-dir>/processed_v3/{cleaned,structured,quality.json}

用法：
  python tools_structure_v3.py                              # 默认 res/系统解剖学
  python tools_structure_v3.py --book-dir res/生理学         # 任意教材目录
  python tools_structure_v3.py --inspect 80                 # 调试单页
  python tools_structure_v3.py --detect-chapters --json     # 只输出检测到的章节
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import tools_structure_v2 as v2

from medical_rag.outputs import chapter_file_name, staged_output_dir

ROOT = Path(__file__).resolve().parent
DEFAULT_BOOK_DIR = ROOT / "res" / "系统解剖学"

RE_CHAPTER_HEAD = re.compile(r"^第([一二三四五六七八九十百零两〇0-9]{1,4})章\s*(.*)$")
RE_TOC_DOTS = re.compile(r"[…]{2,}|[·．.]{3,}|\s\d{1,4}\s*$")

# 表格框内的文本框会按页从正文重建中剔除（由 configure 从 layout.json 载入）
_TABLE_BOXES: dict[int, list[list[float]]] = {}
_TABLE_DIR: Path | None = None
_orig_load_page = v2.load_page


def _load_page_no_tables(n: int) -> dict:
    payload = _orig_load_page(n)
    boxes = _TABLE_BOXES.get(n)
    if not boxes:
        return payload

    def inside(b: dict) -> bool:
        cx = b["xc"]
        cy = (b["y0"] + b["y1"]) / 2
        return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)

    payload["boxes"] = [b for b in payload["boxes"] if not inside(b)]
    return payload


v2.load_page = _load_page_no_tables


def count_pages(boxes_dir: Path) -> int:
    """OCR 结果中的最大 PDF 页号。"""
    largest = 0
    for path in boxes_dir.glob("page-*.json"):
        match = re.search(r"page-(\d+)", path.stem)
        if match:
            largest = max(largest, int(match.group(1)))
    return largest


def detect_chapters(boxes_dir: Path, total_pages: int) -> list[dict]:
    """扫描文本框自动检测章标题，返回 [{"num", "pdf_start", "title"}]。

    规则：
      - 同一页出现多个 “第X章” 视为目录页，整页跳过；
      - 含点线引导 / 尾部页码的行视为目录行，跳过；
      - 每个章号只取首次出现的页作为起始页，标题在该章前若干页内取最完整的一个。
    """
    occurrences: dict[str, dict] = {}
    order: list[str] = []
    for n in range(1, total_pages + 1):
        path = boxes_dir / f"page-{n:04d}.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        boxes = payload.get("boxes") or []
        texts = [str(b.get("text") or "").strip() for b in sorted(boxes, key=lambda b: b.get("y0", 0))]
        if any("目录" == t or "目 录" == t for t in texts):
            continue
        raw_matches: list[tuple[str, str, str]] = []
        for text in texts:
            if not text or len(text) > 40:
                continue
            match = RE_CHAPTER_HEAD.match(text)
            if match:
                raw_matches.append((match.group(1), match.group(2).strip(), text))
        if len(raw_matches) > 1:  # 目录页：同一页出现多个章标题
            continue
        if not raw_matches:
            continue
        key, title, raw_text = raw_matches[0]
        if RE_TOC_DOTS.search(raw_text):
            continue  # 点线引导 / 尾部页码，属于目录行
        title = re.sub(r"\s+", "", title)
        if key not in occurrences:
            occurrences[key] = {"num": len(order) + 1, "pdf_start": n, "title": title}
            order.append(key)
        elif title and len(title) > len(occurrences[key]["title"]) and n - occurrences[key]["pdf_start"] <= 12:
            occurrences[key]["title"] = title
    return [occurrences[key] for key in order]


def load_chapters(path: Path) -> list[dict]:
    """读取人工校对/缓存的章节 JSON。

    支持两种格式：
      [[1, 18, "骨学"], ...]
      {"chapters": [{"num": 1, "pdf_start": 18, "title": "骨学"}, ...]}
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("chapters", []) if isinstance(raw, dict) else raw
    chapters: list[dict] = []
    for index, item in enumerate(items, 1):
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            num, start, title = item[0], item[1], item[2]
        elif isinstance(item, dict):
            num = item.get("num", index)
            start = item.get("pdf_start") or item.get("start")
            title = item.get("title") or ""
        else:
            continue
        if start is None:
            continue
        chapters.append({"num": int(num), "pdf_start": int(start), "title": str(title).strip() or f"第{num}章"})
    chapters.sort(key=lambda c: c["pdf_start"])
    return chapters


def configure(
    boxes_dir: Path,
    tables_dir: Path,
    layout_path: Path,
    out_dir: Path,
    total_pages: int,
    chapters: list[dict],
) -> None:
    """把本次运行的路径/章节配置注入 v2 与 v3 的模块级变量。"""
    global _TABLE_DIR
    v2.SRC = boxes_dir
    v2.TOTAL_PAGES = total_pages
    v2.CHAPTERS = [(c["num"], c["pdf_start"], c["title"]) for c in chapters]
    v2.OUT = out_dir
    _TABLE_DIR = tables_dir

    _TABLE_BOXES.clear()
    if layout_path.exists():
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        for page, value in (layout.get("pages") or {}).items():
            if value.get("table_boxes"):
                _TABLE_BOXES[int(page)] = value["table_boxes"]


def demote_headings(md: str) -> str:
    """表格页 Markdown 的 ## 标题下移为 ###，避免与分页标题同级。"""
    return "\n".join(("###" + line[2:]) if line.startswith("## ") else line for line in md.splitlines())


def rebuild_page(n: int, headers: set[str] | None = None):
    """正文用 PP-OCRv6 的 xy-cut 重建；表格页额外追加表格 HTML。"""
    blocks, info = v2.rebuild_page(n, headers)
    table_md = (_TABLE_DIR / f"page-{n:04d}.md") if _TABLE_DIR else None
    if table_md and table_md.exists():
        text = table_md.read_text(encoding="utf-8").strip()
        table_blocks = [b for b in re.split(r"\n{2,}", text) if b.strip()]
        if table_blocks:
            blocks = blocks + table_blocks
            info["table_page"] = True
            info["tables"] = len(table_blocks)
    return blocks, info


def scan_globals_safe() -> tuple[int, int, set[str]]:
    """页脚页码校准；检测不到时回退为 原书页 = PDF 页（offset = 0）。"""
    try:
        return v2.scan_globals()
    except SystemExit:
        print("[警告] 没有检测到页脚印刷页码，按原书页码 = PDF 页码处理（offset = 0）")
        return 0, 0, set()


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 v3 结构化 Markdown（任意教材）")
    ap.add_argument("--book-dir", type=Path, default=DEFAULT_BOOK_DIR, help="教材资源目录（默认 res/系统解剖学）")
    ap.add_argument("--boxes", type=Path, help="OCR 文本框目录（默认 <book-dir>/text_v3/boxes）")
    ap.add_argument("--tables", type=Path, help="表格 Markdown 目录（默认 <book-dir>/text_v3/tables）")
    ap.add_argument("--layout", type=Path, help="layout.json（默认 <book-dir>/text_v3/layout.json）")
    ap.add_argument("--out", type=Path, help="输出目录（默认 <book-dir>/processed_v3）")
    ap.add_argument("--chapters", type=Path, help="章节 JSON（默认 <book-dir>/chapters.json，存在则使用）")
    ap.add_argument("--total-pages", type=int, default=0, help="PDF 总页数（默认按 OCR 结果推断）")
    ap.add_argument("--inspect", type=int, help="只重建指定页并打印结果（调试用）")
    ap.add_argument("--detect-chapters", action="store_true", help="只检测章节并输出，不重建")
    ap.add_argument("--json", action="store_true", help="与 --detect-chapters 搭配，输出 JSON")
    args = ap.parse_args()

    book_dir = args.book_dir.expanduser().resolve()
    boxes_dir = (args.boxes or book_dir / "text_v3" / "boxes").expanduser().resolve()
    tables_dir = (args.tables or book_dir / "text_v3" / "tables").expanduser().resolve()
    layout_path = (args.layout or book_dir / "text_v3" / "layout.json").expanduser().resolve()
    out_dir = (args.out or book_dir / "processed_v3").expanduser().resolve()
    total_pages = args.total_pages or count_pages(boxes_dir)
    if total_pages <= 0:
        raise SystemExit(f"未找到 OCR 结果：{boxes_dir}")

    if args.detect_chapters:
        chapters = detect_chapters(boxes_dir, total_pages)
        if args.json:
            print(json.dumps(chapters, ensure_ascii=False, indent=2))
        else:
            for chapter in chapters:
                print(f"第{chapter['num']}章 {chapter['title']}（PDF 第 {chapter['pdf_start']} 页）")
        return

    chapters_path = args.chapters or book_dir / "chapters.json"
    chapters: list[dict] = []
    if chapters_path.exists():
        try:
            chapters = load_chapters(chapters_path)
            print(f"章节：使用 {chapters_path}（{len(chapters)} 章）")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"[警告] 章节文件不可用（{exc}），改用自动检测")
    if not chapters:
        chapters = detect_chapters(boxes_dir, total_pages)
        if chapters:
            print(f"章节：自动检测到 {len(chapters)} 章")
    if not chapters:
        chapters = [{"num": 1, "pdf_start": 1, "title": "正文"}]
        print("[警告] 未检测到章标题，按单章输出")

    configure(boxes_dir, tables_dir, layout_path, out_dir, total_pages, chapters)

    if args.inspect:
        blocks, info = rebuild_page(args.inspect)
        print(f"PDF 第 {args.inspect} 页  表格页={info.get('table_page', False)}\n")
        print("\n\n".join(blocks))
        return

    offset, samples, headers = scan_globals_safe()

    cleaned_dir = out_dir / "cleaned"
    structured_dir = out_dir / "structured"

    page_content: dict[int, list[str]] = {}
    table_pages: list[int] = []
    # 写进暂存目录，全部成功后再整体替换：章节变少时旧 Markdown 不会残留
    # 而继续被索引（见 medical_rag/outputs.py）。
    with staged_output_dir(out_dir) as staging:
        cleaned_dir = staging / "cleaned"
        structured_dir = staging / "structured"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        structured_dir.mkdir(parents=True, exist_ok=True)

        for n in range(1, total_pages + 1):
            if not (boxes_dir / f"page-{n:04d}.json").exists():
                continue
            blocks, info = rebuild_page(n, headers)
            if info.get("table_page"):
                table_pages.append(n)
            page_content[n] = blocks
            header = f"# PDF第 {n} 页（原书第 {n - offset} 页）"
            (cleaned_dir / f"page-{n:04d}.md").write_text(
                header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8"
            )

        quality_chapters = []
        for idx, chapter in enumerate(chapters):
            num, start, title = chapter["num"], chapter["pdf_start"], chapter["title"]
            end = chapters[idx + 1]["pdf_start"] - 1 if idx + 1 < len(chapters) else total_pages
            body: list[str] = [
                f"# 第{num}章 {title}",
                "",
                f"> 来源范围：原书第 {start - offset}-{end - offset} 页（PDF 第 {start}-{end} 页）。",
                "> 识别：rapidocr 3.9.2 + PP-OCRv6（GPU）；表格页：PP-StructureV3（HTML 表格）。",
                "> 页码为页脚校准后的原书页码（原书页 = PDF 页 − %d）。" % offset,
                "",
            ]
            for n in range(start, end + 1):
                if n not in page_content:
                    continue
                body.append(f"## 原书第 {n - offset} 页")
                body.append("")
                body.extend(page_content[n])
                body.append("")
            (structured_dir / chapter_file_name(num, title)).write_text(
                "\n".join(body), encoding="utf-8"
            )
            quality_chapters.append(
                {"num": num, "title": title, "pdf_start": start, "pdf_end": end,
                 "printed_start": start - offset, "printed_end": end - offset}
            )
            print(f"第{num}章 {title}: PDF {start}-{end} → 原书 {start - offset}-{end - offset}")

        (staging / "quality.json").write_text(
            json.dumps(
                {
                    "pages": total_pages,
                    "page_offset": offset,
                    "page_offset_samples": samples,
                    "auto_chapters": not chapters_path.exists(),
                    "running_headers": sorted(headers),
                    "pipeline": "text_v3 boxes(PP-OCRv6) -> xy-cut 重建；表格页用 PP-StructureV3 markdown",
                    "table_pages": sorted(table_pages),
                    "chapters": quality_chapters,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(f"完成：{out_dir}  共 {len(chapters)} 章，表格页 {len(table_pages)} 页")


if __name__ == "__main__":
    main()
