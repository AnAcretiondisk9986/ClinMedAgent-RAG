"""8G 显存友好的表格流水线（GPU）：

思路：不从整页跑 PP-StructureV3（会同时驻留版面+2 套表格结构+2 套单元格检测+server OCR，
8G 显存会被打满而换页，反而比 CPU 慢）。改成：
  1. 复用 tools_layout_v3.py 已检测到的表格框（坐标统一在 300dpi 空间）；
  2. 从整页裁出表格区域；
  3. 用 TableRecognitionPipelineV2（use_layout_detection=False）只对裁剪图做表格结构+单元格 OCR。

这样常驻显存约 1.3GB，单表约 1-2 秒。

用法（3.12 环境）：
  .venv-ocr312/Scripts/python.exe tools_table_v3_gpu.py <book.pdf> <out_dir> \
      --layout res/系统解剖学/text_v3/layout.json [--ocr mobile|server] [--device gpu]

-layout 由 tools_layout_v3.py 生成，记录了每页的表格框与渲染尺寸；
--cache 是页图缓存目录，不存在会自动重建。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def register_cuda_dlls() -> None:
    nv = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nv.exists():
        return
    for d in nv.rglob("bin"):
        dd = str(d.resolve())
        try:
            os.add_dll_directory(dd)
        except OSError:
            pass
        os.environ["PATH"] = dd + ";" + os.environ.get("PATH", "")


register_cuda_dlls()

OCR_MODELS = {
    "mobile": ("PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec"),
    "server": ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
}


def render(pdf_path: str, page_number: int, dpi: int, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"page-{page_number:04d}.png"
    import fitz

    if not out.exists():
        with fitz.open(pdf_path) as doc:
            page = doc[page_number - 1]
            scale = dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            pix.save(str(out))
    from PIL import Image

    with Image.open(out) as im:
        return out, im.width, im.height


def gpu_mem() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except Exception:
        return "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layout", default="res/系统解剖学/text_v3/layout.json")
    ap.add_argument("--page-dpi", type=int, default=300, help="layout 检测所用的 dpi")
    ap.add_argument("--cache", default=".cache/render300")
    ap.add_argument("--pages", default=None, help="只用这些页（默认 layout 里所有表格页）")
    ap.add_argument("--ocr", default="mobile", choices=list(OCR_MODELS))
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args()

    from paddleocr import TableRecognitionPipelineV2

    layout = json.loads(Path(args.layout).read_text(encoding="utf-8"))
    if args.pages:
        pages = [int(x) for x in args.pages.split(",") if x.strip()]
    else:
        pages = [int(p) for p in layout["table_pages"]]

    det, rec = OCR_MODELS[args.ocr]
    print(f"表格管线: TableRecognitionPipelineV2 | device={args.device} | OCR={args.ocr}", flush=True)
    pipeline = TableRecognitionPipelineV2(
        device=args.device,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        text_detection_model_name=det,
        text_recognition_model_name=rec,
    )
    print("模型加载后显存:", gpu_mem(), flush=True)

    tables_dir = args.out / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for n in pages:
        boxes = layout["pages"].get(str(n), {}).get("table_boxes", [])
        if not boxes:
            print(f"第{n}页: 无表格框，跳过", flush=True)
            continue
        img_path, iw, ih = render(str(args.pdf), n, args.page_dpi, Path(args.cache))
        # layout 坐标与 --page-dpi 一致时无需缩放；保留比例换算以防 dpi 不同
        sx, sy = iw / layout["pages"][str(n)].get("img_w", iw), ih / layout["pages"][str(n)].get("img_h", ih)
        from PIL import Image

        page_img = Image.open(img_path)
        htmls: list[str] = []
        t_page = time.time()
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = (int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(iw, x1), min(ih, y1)
            if x1 - x0 < 20 or y1 - y0 < 20:
                continue
            crop_path = Path(args.cache) / f"crop-{n:04d}-{i}.png"
            page_img.crop((x0, y0, x1, y1)).save(crop_path)
            t0 = time.time()
            results = pipeline.predict(input=str(crop_path))
            dt = time.time() - t0
            for res in results:
                j = res.json if isinstance(res.json, dict) else {}
                for tr in j.get("res", {}).get("table_res_list", []):
                    html = tr.get("pred_html", "")
                    if html:
                        htmls.append(html)
            print(f"  第{n}页 表{i}: {x1 - x0}x{y1 - y0}px  {dt:.1f}s  显存 {gpu_mem()}", flush=True)
        page_img.close()
        md = "\n\n".join(f"[表格]\n{html}" for html in htmls)
        (tables_dir / f"page-{n:04d}.md").write_text(md, encoding="utf-8")
        (tables_dir / f"page-{n:04d}.json").write_text(
            json.dumps({"page": n, "tables": htmls}, ensure_ascii=False), encoding="utf-8"
        )
        report.append({"page": n, "tables": len(htmls), "seconds": round(time.time() - t_page, 1)})
        print(f"[第{n}页] {time.time() - t_page:.1f}s  {len(htmls)} 个表格", flush=True)

    (args.out / "tables_gpu_report.json").write_text(
        json.dumps({"ocr": args.ocr, "device": args.device, "pages": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"完成：{tables_dir} | 总耗时 {sum(r['seconds'] for r in report):.0f}s")


if __name__ == "__main__":
    main()
