"""v3 结果的定向 OCR 纠错。

只做“在本教材语境下不可能有其他含义”的安全替换，避免误伤正常用字。
对 text_v3/{pages,boxes,tables} 生效（boxes/*.json 里的 text 字段也会同步修正）。

用法：
  python tools_fix_ocr_v3.py            # 默认修 res/系统解剖学/text_v3
  python tools_fix_ocr_v3.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = ROOT / "res" / "系统解剖学" / "text_v3"

# 顺序有意义：先长词，再单字
FIXES: list[tuple[str, str]] = [
    ("排骨", "腓骨"),
    ("肩肿骨", "肩胛骨"),
    ("腔胫骨", "胫骨"),
    ("下降聘帆", "下降腭帆"),
    ("聘帆", "腭帆"),
    ("聘", "腭"),
    ("挠", "桡"),
    ("於", "于"),
    ("內", "内"),
    ("脈", "脉"),
    ("孟下结节", "盂下结节"),
]


def fix_text(text: str) -> tuple[str, dict[str, int]]:
    hits: dict[str, int] = {}
    for wrong, right in FIXES:
        n = text.count(wrong)
        if n:
            # 已经被前一条规则修掉的位置不会重复计数
            text = text.replace(wrong, right)
            hits[wrong] = n
    return text, hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total: dict[str, int] = {}
    files_changed = 0

    for sub, pattern in [("pages", "*.md"), ("tables", "*.md")]:
        d = args.target / sub
        if not d.exists():
            continue
        for f in sorted(d.glob(pattern)):
            raw = f.read_text(encoding="utf-8")
            fixed, hits = fix_text(raw)
            if hits:
                files_changed += 1
                for k, v in hits.items():
                    total[k] = total.get(k, 0) + v
                if not args.dry_run:
                    f.write_text(fixed, encoding="utf-8")

    boxes = args.target / "boxes"
    if boxes.exists():
        for f in sorted(boxes.glob("*.json")):
            payload = json.loads(f.read_text(encoding="utf-8"))
            changed = False
            for b in payload.get("boxes", []):
                new, hits = fix_text(b.get("text", ""))
                if hits:
                    b["text"] = new
                    changed = True
                    for k, v in hits.items():
                        total[k] = total.get(k, 0) + v
            if changed:
                for i, line in enumerate(payload.get("lines", [])):
                    payload["lines"][i] = fix_text(line)[0]
                files_changed += 1
                if not args.dry_run:
                    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    print(f"{'[dry-run] ' if args.dry_run else ''}修改文件数: {files_changed}")
    print("替换统计:", dict(sorted(total.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
