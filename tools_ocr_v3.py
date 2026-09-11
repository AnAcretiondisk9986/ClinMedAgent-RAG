"""OCR v3：用 rapidocr 3.x（PP-OCRv6 / PP-OCRv5）+ onnxruntime GPU 逐页识别。

  - 识别引擎：rapidocr 3.9.2 + PP-OCRv6；
  - 可启用 CUDA（需要 pip 安装的 nvidia cu12 运行库，脚本会自动写入 PATH）；
  - 输出的 boxes/*.json 字段与 tools_structure_v3.py（底层复用 tools_structure_v2.py）兼容。

用法：
  python tools_ocr_v3.py <book.pdf> <out_dir> [--dpi 300] [--start 27] [--end 30]
                         [--version v6] [--model small] [--cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# --- 让 onnxruntime 找到 pip 安装的 CUDA/cuDNN DLL（必须在 import 之前） ---------
def _register_cuda_dlls() -> list[str]:
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nv.exists():
        return []
    dirs = [str(d) for d in nv.rglob("bin")]
    if dirs:
        os.environ["PATH"] = ";".join(dirs) + ";" + os.environ.get("PATH", "")
        for d in dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass
    return dirs


_REGISTERED = _register_cuda_dlls()

from rapidocr import ModelType, OCRVersion, RapidOCR  # noqa: E402

VERSIONS = {"v4": OCRVersion.PPOCRV4, "v5": OCRVersion.PPOCRV5, "v6": OCRVersion.PPOCRV6}
MODELS = {
    "tiny": ModelType.TINY,
    "small": ModelType.SMALL,
    "mobile": ModelType.MOBILE,
    "medium": ModelType.MEDIUM,
    "server": ModelType.SERVER,
}


def clean(s: str) -> str:
    s = s.replace("\u0000", "")
    return re.sub(r"[ \t]+", " ", s).strip()


def order_boxes(boxes: list[dict], width: int) -> list[dict]:
    """两栏检测：横向最大间隙 > 页宽 18% 时先左栏后右栏，否则按 y 再 x。"""
    if not boxes:
        return []
    xs = sorted(b["xc"] for b in boxes)
    gaps = [(xs[i + 1] - xs[i], i) for i in range(len(xs) - 1)]
    gap, idx = max(gaps, default=(0, 0))
    split = (xs[idx] + xs[idx + 1]) / 2 if gap > width * 0.18 else None
    if split:
        left = sorted([b for b in boxes if b["xc"] <= split], key=lambda b: (b["y0"], b["x0"]))
        right = sorted([b for b in boxes if b["xc"] > split], key=lambda b: (b["y0"], b["x0"]))
        return left + right
    return sorted(boxes, key=lambda b: (b["y0"], b["x0"]))


def ocr_page(engine: RapidOCR, pdf_path: str, page_number: int, dpi: int) -> dict:
    import fitz

    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        scale = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
        img = pix.tobytes("png")

    res = engine(img)
    boxes: list[dict] = []
    if res is not None and res.boxes is not None:
        for box, text, score in zip(res.boxes, res.txts, res.scores):
            text = clean(str(text))
            if not text:
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            boxes.append(
                {
                    "bbox": [[round(float(x), 1), round(float(y), 1)] for x, y in box],
                    "x0": round(min(xs), 1),
                    "x1": round(max(xs), 1),
                    "y0": round(min(ys), 1),
                    "y1": round(max(ys), 1),
                    "xc": round(sum(xs) / len(xs), 1),
                    "text": text,
                    "score": round(float(score), 4),
                }
            )
    ordered = order_boxes(boxes, pix.width)
    return {
        "page": page_number,
        "width": pix.width,
        "height": pix.height,
        "boxes": boxes,
        "lines": [b["text"] for b in ordered],
        "avg_confidence": round(sum(b["score"] for b in boxes) / len(boxes), 4) if boxes else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OCR v3（PP-OCRv6 + GPU）")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--pages", default=None, help="逗号分隔的页码列表（优先于 start/end）")
    ap.add_argument("--version", default="v6", choices=list(VERSIONS))
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（默认尝试 CUDA）")
    ap.add_argument("--reset", action="store_true", help="忽略已有结果，全部重跑")
    args = ap.parse_args()

    pages_dir = args.out / "pages"
    boxes_dir = args.out / "boxes"
    pages_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "Det.ocr_version": VERSIONS[args.version],
        "Det.model_type": MODELS[args.model],
        "Rec.ocr_version": VERSIONS[args.version],
        "Rec.model_type": MODELS[args.model],
        "EngineConfig.onnxruntime.use_cuda": not args.cpu,
    }
    engine = RapidOCR(params=params)

    import fitz

    with fitz.open(args.pdf) as doc:
        total = len(doc)
    end = args.end or total
    if args.pages:
        wanted = sorted({int(x) for x in args.pages.split(",") if x.strip()})
        todo = [
            n
            for n in wanted
            if 1 <= n <= total and (args.reset or not (boxes_dir / f"page-{n:04d}.json").exists())
        ]
    else:
        todo = [
            n
            for n in range(args.start, end + 1)
            if args.reset or not (boxes_dir / f"page-{n:04d}.json").exists()
        ]
    print(
        f"引擎: PP-OCR{args.version[-1]} {args.model} | {'CPU' if args.cpu else 'CUDA(可能回退)'} | "
        f"共 {total} 页，待处理 {len(todo)} 页",
        file=sys.stderr,
        flush=True,
    )

    for i, n in enumerate(todo, 1):
        t0 = time.time()
        payload = ocr_page(engine, str(args.pdf), n, args.dpi)
        (boxes_dir / f"page-{n:04d}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )
        (pages_dir / f"page-{n:04d}.md").write_text(
            f"# 第 {n} 页\n\n" + "\n".join(payload["lines"]) + "\n", encoding="utf-8"
        )
        print(
            f"{i}/{len(todo)} 第{n}页  {time.time() - t0:.1f}s  "
            f"{len(payload['boxes'])} boxes  avg={payload['avg_confidence']:.4f}",
            file=sys.stderr,
            flush=True,
        )

    manifest = []
    book_parts = []
    for n in range(1, total + 1):
        box_file = boxes_dir / f"page-{n:04d}.json"
        if not box_file.exists():
            continue
        payload = json.loads(box_file.read_text(encoding="utf-8"))
        manifest.append(
            {
                "page": n,
                "file": f"pages/page-{n:04d}.md",
                "boxes": len(payload["boxes"]),
                "lines": len(payload["lines"]),
                "avg_confidence": payload["avg_confidence"],
            }
        )
        book_parts.append(f"## 第 {n} 页\n\n" + "\n".join(payload["lines"]))
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.pdf.resolve()),
                "engine": f"rapidocr 3.x PP-OCR{args.version[-1]} {args.model}",
                "cuda": not args.cpu,
                "pages": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out / "book.md").write_text(
        "# 系统解剖学 第5版（OCR v3）\n\n" + "\n\n".join(book_parts) + "\n", encoding="utf-8"
    )
    print(f"完成：{args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
