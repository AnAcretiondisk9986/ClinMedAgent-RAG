"""PDF 文字层检测与“文字层工作流”。

适用对象：自带完整文字层的数字版 PDF（如人卫电子教材），这类 PDF 不需要跑
GPU OCR 流水线，直接抽取文字即可建立可检索索引。

两个能力：

1. ``analyze`` / ``cached_analyze``：逐页统计可提取文字，判定 PDF 属于
   ``text``（文字层）/ ``scan``（图片型）/ ``mixed``（混合）；缓存按 PDF
   mtime 失效，供书库状态与导入检测复用。
2. ``build_structured``：把文字层重建为与 OCR 流水线完全一致的目录结构
   ``processed_v3/{cleaned,structured,quality.json}``，包含 xy-cut 分栏阅读
   顺序、页眉剔除、页脚印刷页码校准（原书页 = PDF 页 − 偏移）、章节识别和
   标题层级（###/####/#####），之后可直接用 ``ingest_markdown_tree`` 建索引。

命令行（网站流水线的 text 阶段就是调用它）::

    python -m medical_rag.pdftext <book.pdf> <book_dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from .outputs import chapter_file_name, staged_output_dir

MIN_PAGE_CHARS = 30  # 单页至少这么多非空白字符才算“有文字”
# 判定规则：以文字层覆盖率为主，平均字数只用来兜底“每页只有水印/页码”的情况
RATIO_TEXT = 0.5    # 至少一半页面有文字
AVG_TEXT = 40       # 且平均每页不少于 40 字
RATIO_SCAN = 0.15   # 有文字的页 ≤15% → 图片型
AVG_SCAN = 20       # 或平均每页不足 20 字（基本是水印/页码） → 图片型
KIND_LABELS = {"text": "文字层 PDF", "scan": "图片型 PDF", "mixed": "混合型 PDF"}

CN_NUM = "一二三四五六七八九十百零两〇"
RE_CHAPTER = re.compile(rf"^第\s*([{CN_NUM}0-9０-９]{{1,4}})\s*章\s*(.*)$")
RE_SECTION = re.compile(rf"^第[{CN_NUM}]+节")
RE_SUB = re.compile(rf"^[{CN_NUM}]+、")
RE_SUBSUB = re.compile(rf"^[（(][{CN_NUM}]+[)）]")
RE_FIG = re.compile(r"^图\s*\d+")
RE_BARE_NUM = re.compile(r"^\d{1,4}$")
RE_TOC_DOTS = re.compile(r"[…]{2,}|(?:[.．]\s*){3,}")


def kind_label(kind: str | None) -> str:
    return KIND_LABELS.get(str(kind), "未知类型")


def recommendation(info: dict[str, Any]) -> str:
    """给前端/日志用的一句话建议。"""
    if info.get("kind") == "text":
        return (
            f"检测到文字层：{info['text_pages']}/{info['pages']} 页可提取文字"
            f"（共 {info['chars']} 字），可直接解析入库，无需 OCR。"
        )
    return (
        f"未检测到可用文字层（{info['text_pages']}/{info['pages']} 页有文字），"
        "将使用 OCR 流水线。"
    )


# --------------------------------------------------------------- 文字层检测
def analyze(pdf: Path | str) -> dict[str, Any]:
    """逐页统计可提取文字，判断 PDF 类型。"""
    pdf = Path(pdf)
    stat = pdf.stat()
    doc = fitz.open(pdf)
    try:
        pages = len(doc)
        text_pages = 0
        total_chars = 0
        image_pages = 0
        for page in doc:
            text = re.sub(r"\s+", "", page.get_text("text") or "")
            length = len(text)
            total_chars += length
            if length >= MIN_PAGE_CHARS:
                text_pages += 1
            if page.get_images(full=True):
                image_pages += 1
    finally:
        doc.close()

    ratio = text_pages / pages if pages else 0.0
    average = total_chars / pages if pages else 0.0
    if ratio >= RATIO_TEXT and average >= AVG_TEXT:
        kind = "text"
    elif ratio <= RATIO_SCAN or average < AVG_SCAN:
        kind = "scan"
    else:
        kind = "mixed"
    return {
        "pdf": str(pdf.resolve()),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "pages": pages,
        "text_pages": text_pages,
        "chars": total_chars,
        "avg_chars": round(average, 1),
        "text_ratio": round(ratio, 3),
        "image_pages": image_pages,
        "kind": kind,
        "checked_at": time.time(),
    }


def cached_analyze(pdf: Path | str, cache_dir: Path | str) -> dict[str, Any]:
    """带缓存的检测结果；PDF 修改后自动失效。"""
    pdf = Path(pdf)
    cache = Path(cache_dir)
    try:
        mtime = pdf.stat().st_mtime
        resolved = str(pdf.resolve())
    except OSError:
        return {"kind": "scan", "pages": 0, "text_pages": 0, "chars": 0, "error": "PDF 不可读"}

    key = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    cache_file = cache / f"{key}.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            if str(data.get("pdf")) == resolved and abs(float(data.get("mtime", -1)) - mtime) < 1e-6:
                return data
        except (OSError, ValueError, TypeError):
            pass

    data = analyze(pdf)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        temp = cache_file.with_suffix(".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, cache_file)
    except OSError:
        pass
    return data


# ------------------------------------------------------------- 版面重建工具
def _join_block(text: str) -> str:
    """块内换行拼接：中文直接相连，拉丁词之间补空格。"""
    parts = [part.strip() for part in str(text).splitlines() if part.strip()]
    joined = ""
    for part in parts:
        if joined and re.search(r"[A-Za-z0-9]$", joined) and re.match(r"[A-Za-z0-9]", part):
            joined += " " + part
        else:
            joined += part
    return joined


def _gaps(blocks: list[dict], axis: str) -> list[tuple[float, float]]:
    spans = sorted((block[f"{axis}0"], block[f"{axis}1"]) for block in blocks)
    merged = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(left[1], right[0]) for left, right in zip(merged, merged[1:])]


def _xycut(blocks: list[dict]) -> list[dict]:
    """对文本块做简化 xy-cut，恢复分栏阅读顺序。"""
    if len(blocks) <= 3:
        return sorted(blocks, key=lambda b: (round(b["y0"]), b["x0"]))
    x0 = min(b["x0"] for b in blocks)
    x1 = max(b["x1"] for b in blocks)
    for gap_start, gap_end in sorted(_gaps(blocks, "x"), key=lambda g: g[1] - g[0], reverse=True):
        if gap_end - gap_start < max(0.06 * (x1 - x0), 30):
            continue
        near = [b for b in blocks if (b["x0"] + b["x1"]) / 2 <= gap_start]
        far = [b for b in blocks if (b["x0"] + b["x1"]) / 2 >= gap_end]
        if not near or not far or len(near) + len(far) != len(blocks):
            continue
        return _xycut(near) + _xycut(far)
    y0 = min(b["y0"] for b in blocks)
    y1 = max(b["y1"] for b in blocks)
    for gap_start, gap_end in sorted(_gaps(blocks, "y"), key=lambda g: g[1] - g[0], reverse=True):
        if gap_end - gap_start < max(0.05 * (y1 - y0), 24):
            continue
        near = [b for b in blocks if (b["y0"] + b["y1"]) / 2 <= gap_start]
        far = [b for b in blocks if (b["y0"] + b["y1"]) / 2 >= gap_end]
        if not near or not far or len(near) + len(far) != len(blocks):
            continue
        return _xycut(near) + _xycut(far)
    return sorted(blocks, key=lambda b: (round(b["y0"]), b["x0"]))


def _classify(text: str) -> str:
    """把一段文字映射成 Markdown 块（标题层级 / 图注）。"""
    text = text.strip()
    if not text:
        return ""
    if RE_CHAPTER.match(text):
        return text  # 章名保留为正文行，文件级标题由生成器写入
    if RE_SECTION.match(text) and len(text) <= 24:
        return f"### {text}"
    if RE_SUB.match(text) and len(text) <= 28:
        return f"#### {text}"
    if RE_SUBSUB.match(text) and len(text) <= 28:
        return f"##### {text}"
    if RE_FIG.match(text) and len(text) <= 80:
        return f"[图注/图例] {text}"
    return text


def _dominant_offset(samples: list[tuple[int, int]]) -> tuple[int, int, int]:
    """由 (PDF 页, 页脚印刷页码) 推断主偏移量。"""
    if not samples:
        return 0, 0, 0
    counter = Counter(number - printed for number, printed in samples if 0 <= number - printed <= 60)
    if not counter:
        return 0, 0, len(samples)
    offset, count = counter.most_common(1)[0]
    return offset, count, len(samples)


def _load_chapters(path: Path | None) -> list[dict]:
    if path is None or not Path(path).exists():
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
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
    chapters.sort(key=lambda chapter: chapter["pdf_start"])
    return chapters


def _detect_chapters(page_lines: list[list[str]]) -> list[dict]:
    """从文字层检测章标题，规则与 OCR 版一致（跳过目录页与点线行）。"""
    occurrences: dict[str, dict] = {}
    order: list[str] = []
    digits_map = str.maketrans("０１２３４５６７８９", "0123456789")
    for number, lines in enumerate(page_lines, 1):
        if any(line.replace(" ", "").replace("\u3000", "") in ("目录", "目次") for line in lines):
            continue  # 目录页
        matches: list[tuple[str, str, str]] = []
        for text in lines:
            if not text or len(text) > 60 or text.startswith("#") or text.startswith("["):
                continue
            match = RE_CHAPTER.match(text)
            if not match:
                continue
            if RE_TOC_DOTS.search(text):
                continue  # 点线引导行，属于目录
            matches.append((match.group(1), match.group(2).strip(), text))
        if len(matches) > 1:  # 同页出现多个章标题 → 目录页
            continue
        if not matches:
            continue
        key, title, raw = matches[0]
        key = key.strip().translate(digits_map)
        title = re.sub(r"\s+", "", title)
        title = re.sub(r"[.．·…]+\d{1,4}$", "", title)
        if key not in occurrences:
            occurrences[key] = {"num": len(order) + 1, "pdf_start": number, "title": title}
            order.append(key)
        elif title and len(title) > len(occurrences[key]["title"]) and number - occurrences[key]["pdf_start"] <= 12:
            occurrences[key]["title"] = title
    return [occurrences[key] for key in order]


# --------------------------------------------------------------- 结构化构建
def build_structured(
    pdf: Path | str,
    book_dir: Path | str,
    chapters_path: Path | str | None = None,
) -> dict[str, Any]:
    """从 PDF 文字层生成 processed_v3/{cleaned,structured,quality.json}。"""
    pdf = Path(pdf)
    book_dir = Path(book_dir)
    if chapters_path is None:
        default = book_dir / "chapters.json"
        chapters_path = default if default.exists() else None

    doc = fitz.open(pdf)
    try:
        total = len(doc)
        pages: list[dict] = []
        top_counter: Counter[str] = Counter()
        footer_samples: list[tuple[int, int]] = []
        for number, page in enumerate(doc, start=1):
            height = float(page.rect.height)
            raw_blocks = [
                block
                for block in page.get_text("blocks", sort=True)
                if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()
            ]
            blocks = _xycut([
                {
                    "x0": float(block[0]),
                    "y0": float(block[1]),
                    "x1": float(block[2]),
                    "y1": float(block[3]),
                    "text": str(block[4]),
                }
                for block in raw_blocks
            ])
            items: list[tuple[dict, str]] = []
            for block in blocks:
                text = _join_block(block["text"])
                if not text:
                    continue
                items.append((block, text))
                if block["y1"] < height * 0.08 and len(text) <= 25:
                    top_counter[text] += 1
                if block["y0"] > height * 0.9:
                    match = RE_BARE_NUM.match(text)
                    if match:
                        footer_samples.append((number, int(match.group(0))))
            pages.append({"height": height, "items": items})
            print(f"{number}/{total} 第{number}页", flush=True)
    finally:
        doc.close()

    headers = {text for text, count in top_counter.items() if count >= max(8, total // 15)}
    offset, offset_support, offset_samples = _dominant_offset(footer_samples)

    page_lines: list[list[str]] = []
    for page in pages:
        lines: list[str] = []
        for block, text in page["items"]:
            if block["y1"] < page["height"] * 0.08 and text in headers:
                continue  # 跨页高频页眉
            if block["y0"] > page["height"] * 0.9 and RE_BARE_NUM.match(text):
                continue  # 页脚页码
            line = _classify(text)
            if line:
                lines.append(line)
        page_lines.append(lines)

    chapters = _load_chapters(Path(chapters_path) if chapters_path else None)
    if not chapters:
        chapters = _detect_chapters(page_lines)
    if not chapters:
        chapters = [{"num": 1, "pdf_start": 1, "title": "正文"}]

    out = book_dir / "processed_v3"
    quality_chapters: list[dict] = []
    # 写进暂存目录，全部成功后再整体替换：旧章节文件不会残留，重建失败也不会
    # 留下半套结果（见 medical_rag/outputs.py）。
    with staged_output_dir(out) as staging:
        cleaned_dir = staging / "cleaned"
        structured_dir = staging / "structured"
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        structured_dir.mkdir(parents=True, exist_ok=True)

        for number, lines in enumerate(page_lines, start=1):
            header = f"# PDF第 {number} 页（原书第 {number - offset} 页）"
            (cleaned_dir / f"page-{number:04d}.md").write_text(
                header + "\n\n" + "\n\n".join(lines) + "\n", encoding="utf-8"
            )

        for index, chapter in enumerate(chapters):
            num, start, title = chapter["num"], max(1, chapter["pdf_start"]), chapter["title"]
            end = chapters[index + 1]["pdf_start"] - 1 if index + 1 < len(chapters) else total
            end = min(total, end)
            body = [
                f"# 第{num}章 {title}",
                "",
                f"> 来源范围：原书第 {start - offset}-{end - offset} 页（PDF 第 {start}-{end} 页）。",
                "> 来源：PDF 文字层直接解析（未使用 OCR）。",
                f"> 页码为页脚校准后的原书页码（原书页 = PDF 页 − {offset}）。",
                "",
            ]
            for number in range(start, end + 1):
                lines = page_lines[number - 1]
                if not lines:
                    continue
                body.append(f"## 原书第 {number - offset} 页")
                body.append("")
                body.extend(lines)
                body.append("")
            (structured_dir / chapter_file_name(num, title)).write_text(
                "\n".join(body), encoding="utf-8"
            )
            print(f"第{num}章 {title}: PDF {start}-{end} → 原书 {start - offset}-{end - offset}", flush=True)
            quality_chapters.append(
                {
                    "num": num,
                    "title": title,
                    "pdf_start": start,
                    "pdf_end": end,
                    "printed_start": start - offset,
                    "printed_end": end - offset,
                }
            )

        quality = {
            "pages": total,
            "page_offset": offset,
            "page_offset_samples": offset_samples,
            "page_offset_support": offset_support,
            "auto_chapters": not _load_chapters(Path(chapters_path) if chapters_path else None),
            "running_headers": sorted(headers),
            "pipeline": "text-layer（PyMuPDF 直接解析，未使用 OCR）",
            "table_pages": [],
            "chapters": quality_chapters,
        }
        (staging / "quality.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"完成：{out}  共 {len(chapters)} 章，页码偏移 {offset}", flush=True)
    return quality


# -------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="从 PDF 文字层生成结构化 Markdown（无需 OCR）")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("book_dir", type=Path)
    parser.add_argument("--chapters", type=Path, help="章节 JSON（默认 <book_dir>/chapters.json）")
    parser.add_argument("--detect-only", action="store_true", help="只检测文字层并输出 JSON，不生成结构化文件")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        raise SystemExit(f"PDF 不存在：{args.pdf}")

    info = analyze(args.pdf)
    print(
        f"文字层检测：{info['text_pages']}/{info['pages']} 页有文字，共 {info['chars']} 字"
        f"（{kind_label(info['kind'])}）",
        flush=True,
    )
    if args.detect_only:
        print(json.dumps(info, ensure_ascii=False), flush=True)
        return
    if info["kind"] != "text":
        raise SystemExit(
            f"该 PDF 文字层不足（{kind_label(info['kind'])}，仅 {info['text_pages']}/{info['pages']} 页有文字）；"
            "请改用 OCR 流水线（ocr → layout → tables → fix → structure）"
        )
    build_structured(args.pdf, args.book_dir, chapters_path=args.chapters)


if __name__ == "__main__":
    main()
