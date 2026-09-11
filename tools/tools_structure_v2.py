"""从 text_v2/boxes 的 OCR 坐标重建结构化 Markdown：
  1. xy-cut 递归分栏，恢复正确的阅读顺序（修复左右栏逐行交错）；
  2. 图内标签预提取：短、无标点的独立文本框（颅骨/肱骨…）统一移到页尾，
     避免混入正文行；
  3. 行聚类 + 段落重建（缩进/行距/标点/结构信号），正文成段；
  4. 标题层级：第X节 -> ###，一、/（一）-> ####/#####；章名保留为正文行；
  5. 用页脚印刷页码校准“原书页码”（PDF 页 - 偏移 = 原书页），引用不再偏移；
  6. 顶部页眉按跨页频率剔除。

用法：
  python tools_structure_v2.py                 # 全量构建 processed_v2/
  python tools_structure_v2.py --inspect 18    # 调试单页版面重建
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "res" / "系统解剖学" / "text_v2" / "boxes"
OUT = ROOT / "res" / "系统解剖学" / "processed_v2"

# 章节边界沿用 v1 人工校对过的结果（PDF 页码），标题取自 v1 的结构化文件名。
CHAPTERS: list[tuple[int, int, str]] = [
    (1, 18, "骨学"),
    (2, 43, "关节学"),
    (3, 64, "肌学"),
    (4, 99, "消化系统"),
    (5, 120, "呼吸系统"),
    (6, 135, "泌尿系统"),
    (7, 143, "男性生殖系统"),
    (8, 150, "女性生殖系统"),
    (9, 160, "腹膜"),
    (10, 168, "心血管系统"),
    (11, 213, "淋巴系统"),
    (12, 228, "视器"),
    (13, 241, "前庭蜗器"),
    (14, 260, "中枢神经系统"),
    (15, 294, "周围神经系统"),
    (16, 336, "神经系统的传导通路"),
    (17, 349, "脑和脊髓的被膜、血管及脑脊液循环"),
    (18, 362, "内分泌系统"),
]
TOTAL_PAGES = 368

CJK = r"\u4e00-\u9fff"
END_PUNCT = "。！？；：)）》】"
ALL_PUNCT = "。，、；：！？（）()[]【】《》<>“”‘’…—·,.;:!?~"

CN_NUM = "一二三四五六七八九十百零"
RE_CHAPTER = re.compile(rf"^第[{CN_NUM}]+章")
RE_SECTION = re.compile(rf"^第[{CN_NUM}]+节")
RE_SUB = re.compile(rf"^[{CN_NUM}]+、")
RE_SUBSUB = re.compile(rf"^[（(][{CN_NUM}]+[)）]")
RE_ENUM = re.compile(r"^（?\d{1,3}[.、．）)：]")
RE_FIG_CAPTION = re.compile(r"^图\s*\d+")
RE_BARE_NUM = re.compile(r"^\d{1,4}$")
RE_LOOSE_HEAD = re.compile(rf"^[、][{CJK}]{{2,10}}$")  # OCR 丢失编号的小节标题，如“、骨的分类”
RE_HAS_TEXT = re.compile(rf"[{CJK}A-Za-z]")

# 这些短行是版面文字而非图内标签
LABEL_WHITELIST = {"学习目标", "思考题", "目录", "目 录", "绪论", "小结", "掌握", "了解"}
# 图注标签最多提取多少个，超过视为表格页，放弃提取
MAX_LABELS_PER_PAGE = 20


def load_page(n: int) -> dict:
    return json.loads((SRC / f"page-{n:04d}.json").read_text(encoding="utf-8"))


def merge_intervals(values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    spans = sorted(values)
    merged: list[list[float]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def gaps_along(boxes: list[dict], axis: str, min_gap: float) -> list[tuple[float, float]]:
    """Find empty (gap_start, gap_end) intervals along x or y spanning all boxes."""
    if not boxes:
        return []
    spans = [(b[f"{axis}0"], b[f"{axis}1"]) for b in boxes]
    merged = merge_intervals(spans)
    return [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if b[0] - a[1] >= min_gap]


def cluster_lines(boxes: list[dict]) -> list[list[dict]]:
    """Group boxes into visual lines by vertical centre overlap."""
    lines: list[list[dict]] = []
    for box in sorted(boxes, key=lambda b: ((b["y0"] + b["y1"]) / 2, b["x0"])):
        centre = (box["y0"] + box["y1"]) / 2
        if lines:
            current = lines[-1]
            ref = statistics.mean((b["y0"] + b["y1"]) / 2 for b in current)
            height = statistics.median(b["y1"] - b["y0"] for b in current)
            if abs(centre - ref) <= 0.6 * max(height, 8.0):
                current.append(box)
                continue
        lines.append([box])
    return [sorted(line, key=lambda b: b["x0"]) for line in lines]


def xycut(boxes: list[dict], depth: int = 0) -> list[list[list[dict]]]:
    """Recursive xy-cut. Returns leaf blocks; each block is a list of visual lines."""
    if len(boxes) <= 2 or depth >= 8:
        return [cluster_lines(boxes)]
    x0 = min(b["x0"] for b in boxes)
    x1 = max(b["x1"] for b in boxes)
    y0 = min(b["y0"] for b in boxes)
    y1 = max(b["y1"] for b in boxes)
    region_w, region_h = x1 - x0, y1 - y0

    cuts = [
        ("x", max(0.025 * region_w, 50.0)),
        ("y", max(0.04 * region_h, 45.0)),
    ]
    for axis, min_gap in cuts:
        for g0, g1 in sorted(gaps_along(boxes, axis, min_gap), key=lambda g: g[1] - g[0], reverse=True):
            near = [b for b in boxes if b[f"{axis}1"] <= g0]
            far = [b for b in boxes if b[f"{axis}0"] >= g1]
            spanning = [b for b in boxes if b not in near and b not in far]
            if spanning or not near or not far:
                continue
            if max(len(near), len(far)) < 3:
                continue
            return xycut(near, depth + 1) + xycut(far, depth + 1)
    return [cluster_lines(boxes)]


def join_lines(parts: list[str]) -> str:
    """Join wrapped lines: CJK joins with no space; ASCII word wraps get a space."""
    text = ""
    for part in parts:
        if text and re.search(r"[A-Za-z0-9_-]$", text) and re.match(r"[A-Za-z0-9_-]", part):
            text += " "
        text += part
    return text


def leaf_char_width(lines: list[list[dict]]) -> float:
    widths = []
    for line in lines:
        for box in line:
            n = len(box["text"])
            if n >= 2:
                widths.append((box["x1"] - box["x0"]) / n)
    return statistics.median(widths) if widths else 30.0


def is_structural(text: str) -> bool:
    """Lines that always start a new paragraph (headings, enumerations, captions)."""
    return bool(
        RE_CHAPTER.match(text) or RE_SECTION.match(text) or RE_SUB.match(text)
        or RE_SUBSUB.match(text) or RE_ENUM.match(text) or RE_FIG_CAPTION.match(text)
        or RE_LOOSE_HEAD.match(text)
    )


def heading_like(text: str) -> bool:
    """Standalone headings / captions that must not absorb the following line.

    Enumerated list items ("3. …") are paragraphs, not headings — they wrap.
    """
    return bool(
        RE_CHAPTER.match(text) or RE_SECTION.match(text) or RE_SUB.match(text)
        or RE_SUBSUB.match(text) or RE_FIG_CAPTION.match(text) or RE_LOOSE_HEAD.match(text)
    )


def build_paragraphs(lines: list[list[dict]]) -> list[str]:
    """Merge visual lines into paragraphs using indent / spacing / punctuation cues."""
    if not lines:
        return []
    char_w = leaf_char_width(lines)
    line_h = statistics.median(b["y1"] - b["y0"] for line in lines for b in line)
    gaps = [
        lines[i + 1][0]["y0"] - max(b["y1"] for b in lines[i])
        for i in range(len(lines) - 1)
    ]
    normal = [g for g in gaps if g < 3 * line_h]
    spacing = statistics.median(normal) if normal else line_h * 0.35
    modal_x0 = Counter(round(line[0]["x0"], -1) for line in lines).most_common(1)[0][0]

    paragraphs: list[list[str]] = [[b["text"] for b in lines[0]]]
    for i in range(1, len(lines)):
        texts = [b["text"] for b in lines[i]]
        first = texts[0]
        prev = join_lines(paragraphs[-1])
        gap = gaps[i - 1]
        indent = lines[i][0]["x0"] - modal_x0 >= 1.5 * char_w
        prev_is_heading = heading_like(paragraphs[-1][0]) and len(prev) <= 30
        if is_structural(first) or indent or prev_is_heading:
            paragraphs.append(texts)
        elif not prev.endswith(tuple(END_PUNCT)):
            # 上一行句子未结束（如列表项/正文折行），无论行距都应续接
            paragraphs[-1].extend(texts)
        elif gap > 1.3 * spacing:
            paragraphs.append(texts)
        else:
            paragraphs[-1].extend(texts)
    return [join_lines(p) for p in paragraphs]


def classify_block(text: str) -> str:
    """Map a reconstructed paragraph to a markdown block."""
    text = text.strip()
    if not text:
        return ""
    if RE_CHAPTER.match(text):
        return text  # 章名保持正文行（文件的 h1 已由生成器写入）
    if RE_SECTION.match(text) and len(text) <= 24:
        return f"### {text}"
    if RE_SUB.match(text) and len(text) <= 28:
        return f"#### {text}"
    if RE_SUBSUB.match(text) and len(text) <= 28:
        return f"##### {text}"
    if RE_FIG_CAPTION.match(text) and len(text) <= 60:
        return f"[图注/图例] {text}"
    return text


def looks_like_label(text: str) -> bool:
    """图内标签：短、无标点、非标题/编号/图题/页码。"""
    if text in LABEL_WHITELIST or not RE_HAS_TEXT.search(text):
        return False
    if len(text) > 8:
        return False
    if any(ch in text for ch in ALL_PUNCT):
        return False
    return not (
        RE_CHAPTER.match(text) or RE_SECTION.match(text) or RE_SUB.match(text)
        or RE_SUBSUB.match(text) or RE_ENUM.match(text) or RE_FIG_CAPTION.match(text)
        or RE_BARE_NUM.match(text)
    )


def split_labels(boxes: list[dict]) -> tuple[list[dict], list[str]]:
    """Pull figure-label boxes out of the layout flow (page has a 图 caption)."""
    has_fig = any(RE_FIG_CAPTION.match(b["text"]) for b in boxes)
    if not has_fig:
        return boxes, []
    kept, labels = [], []
    for box in boxes:
        if looks_like_label(box["text"]):
            labels.append(box["text"])
        else:
            kept.append(box)
    if len(labels) > MAX_LABELS_PER_PAGE:  # 大概率是表格页，保留原样
        return boxes, []
    return kept, labels


def rebuild_page(n: int, headers: set[str] | None = None) -> tuple[list[str], dict]:
    """Return (markdown blocks for the page body, debug info)."""
    payload = load_page(n)
    width, height, boxes = payload["width"], payload["height"], payload["boxes"]
    headers = headers or set()

    printed = None
    body_boxes = []
    labels: list[str] = []
    has_fig = any(RE_FIG_CAPTION.match(b["text"]) for b in boxes)
    for box in boxes:
        text = box["text"]
        top_band = box["y1"] < height * 0.07
        bottom_band = box["y0"] > height * 0.90
        if RE_BARE_NUM.match(text) and (bottom_band or top_band):
            if bottom_band:
                printed = int(text)
            continue  # 页码不进正文
        if top_band and text in headers:
            continue  # 跨页高频页眉
        if top_band and has_fig and looks_like_label(text):
            labels.append(text)  # 页顶的孤立短框是图标签，不能当页眉丢弃
            continue
        body_boxes.append(box)

    body_boxes, more_labels = split_labels(body_boxes)
    labels.extend(more_labels)
    blocks: list[str] = []
    for leaf in xycut(body_boxes):
        for text in build_paragraphs(leaf):
            block = classify_block(text)
            if block:
                blocks.append(block)
    blocks.extend(f"[图注/图例] {label}" for label in labels)
    return blocks, {"page": n, "printed": printed, "boxes": len(boxes), "blocks": blocks}


def scan_globals() -> tuple[int, int, set[str]]:
    """One pass over all pages: page-number offset + frequent running headers."""
    offsets: list[int] = []
    header_counter: Counter[str] = Counter()
    for n in range(1, TOTAL_PAGES + 1):
        path = SRC / f"page-{n:04d}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        height = payload["height"]
        for box in payload["boxes"]:
            if box["y1"] < height * 0.07 and len(box["text"]) <= 20:
                header_counter[box["text"]] += 1
            if RE_BARE_NUM.match(box["text"]) and box["y0"] > height * 0.90:
                offset = n - int(box["text"])
                if 0 <= offset <= 60:
                    offsets.append(offset)
    if not offsets:
        raise SystemExit("没有检测到页脚印刷页码，无法校准原书页码")
    offset, count = Counter(offsets).most_common(1)[0]
    support = count / len(offsets)
    headers = {t for t, c in header_counter.items() if c >= 15}
    print(f"页码校准：{count}/{len(offsets)} 个检测点支持偏移 = {offset}（置信 {support:.0%}）")
    print(f"页眉剔除：{sorted(headers)}")
    return offset, len(offsets), headers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", type=int, help="只重建指定页并打印结果（调试用）")
    args = ap.parse_args()

    if args.inspect:
        blocks, info = rebuild_page(args.inspect)
        print(f"PDF 第 {args.inspect} 页，{info['boxes']} 个 box，页脚印刷页码 {info['printed']}\n")
        print("\n\n".join(blocks))
        return

    offset, samples, headers = scan_globals()

    cleaned_dir = OUT / "cleaned"
    structured_dir = OUT / "structured"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    structured_dir.mkdir(parents=True, exist_ok=True)

    page_content: dict[int, list[str]] = {}
    for n in range(1, TOTAL_PAGES + 1):
        if not (SRC / f"page-{n:04d}.json").exists():
            continue
        blocks, _ = rebuild_page(n, headers)
        page_content[n] = blocks
        header = f"# PDF第 {n} 页（原书第 {n - offset} 页）"
        (cleaned_dir / f"page-{n:04d}.md").write_text(
            header + "\n\n" + "\n\n".join(blocks) + "\n", encoding="utf-8"
        )

    quality_chapters = []
    for idx, (num, start, title) in enumerate(CHAPTERS):
        end = CHAPTERS[idx + 1][1] - 1 if idx + 1 < len(CHAPTERS) else TOTAL_PAGES
        chapter_no = f"{num:02d}"
        body: list[str] = [
            f"# 第{num}章 {title}",
            "",
            f"> 来源范围：原书第 {start - offset}-{end - offset} 页（PDF 第 {start}-{end} 页）。",
            "> 版面重构：xy-cut 分栏阅读顺序、图内标签分离、段落重建、标题层级（###/####）；页码为校准后的原书页码。",
            "",
        ]
        for n in range(start, end + 1):
            if n not in page_content:
                continue
            body.append(f"## 原书第 {n - offset} 页")
            body.append("")
            body.extend(page_content[n])
            body.append("")
        (structured_dir / f"{chapter_no}-{title}.md").write_text("\n".join(body), encoding="utf-8")
        quality_chapters.append(
            {"num": num, "title": title, "pdf_start": start, "pdf_end": end,
             "printed_start": start - offset, "printed_end": end - offset}
        )
        print(f"第{num}章 {title}: PDF {start}-{end} → 原书 {start - offset}-{end - offset}")

    (OUT / "quality.json").write_text(
        json.dumps(
            {
                "pages": TOTAL_PAGES,
                "page_offset": offset,
                "page_offset_samples": samples,
                "running_headers": sorted(headers),
                "pipeline": "text_v2 boxes -> 图内标签分离 -> xy-cut 分栏 -> 段落重建 -> 标题层级/图注标记",
                "chapters": quality_chapters,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"完成：{OUT}")


if __name__ == "__main__":
    main()
