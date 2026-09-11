"""全书版面检测：用 PP-DocLayout 找出含表格/图片的页，供表格流水线选择。

运行在 Python 3.12 环境（.venv-ocr312，paddle GPU）。

用法：
  .venv-ocr312/Scripts/python.exe tools_layout_v3.py <book.pdf> <out.json> \
      [--dpi 300] [--device gpu] [--cache .cache/render]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path


def register_cuda_dlls() -> None:
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nv.exists():
        return
    for d in nv.rglob("bin"):
        try:
            os.add_dll_directory(str(d))
        except OSError:
            pass
        os.environ["PATH"] = str(d) + ";" + os.environ.get("PATH", "")


register_cuda_dlls()

from paddleocr import LayoutDetection  # noqa: E402


def render(pdf_path: str, page_number: int, dpi: int, cache_dir: Path) -> tuple[Path, int, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"page-{page_number:04d}.png"
    if out.exists():
        from PIL import Image

        with Image.open(out) as im:
            return out, im.width, im.height
    import fitz

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        scale = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
        pix.save(str(out))
    return out, pix.width, pix.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--cache", default=".cache/render")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    args = ap.parse_args()

    import fitz

    with fitz.open(args.pdf) as doc:
        total = len(doc)
    end = args.end or total

    model = LayoutDetection(device=args.device)
    cache = Path(args.cache)
    pages: dict[str, dict] = {}

    t_all = time.time()
    for n in range(args.start, end + 1):
        img, iw, ih = render(str(args.pdf), n, args.dpi, cache)
        t0 = time.time()
        results = model.predict(input=str(img))
        dt = time.time() - t0
        labels: Counter[str] = Counter()
        table_boxes: list[list[float]] = []
        for r in results:
            j = r.json if isinstance(r.json, dict) else {}
            d = j.get("res", j)
            for b in d.get("boxes", []):
                labels[b["label"]] += 1
                if b["label"] == "table":
                    table_boxes.append([round(float(x), 1) for x in b["coordinate"]])
        pages[str(n)] = {
            "labels": dict(labels),
            "tables": len(table_boxes),
            "table_boxes": table_boxes,
            "img_w": iw,
            "img_h": ih,
            "seconds": round(dt, 2),
        }
        if n % 25 == 0 or n == end:
            print(f"  {n}/{end}  {dt:.2f}s/页  表格页累计={sum(1 for v in pages.values() if v['tables'])}", flush=True)

    table_pages = sorted(int(k) for k, v in pages.items() if v["tables"] > 0)
    args.out.write_text(
        json.dumps(
            {
                "dpi": args.dpi,
                "device": args.device,
                "pages": pages,
                "table_pages": table_pages,
                "elapsed_seconds": round(time.time() - t_all, 1),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"完成：{args.out} | 含表格页 {len(table_pages)} 页 | 用时 {time.time() - t_all:.0f}s")
    print("表格页:", table_pages)


if __name__ == "__main__":
    main()
