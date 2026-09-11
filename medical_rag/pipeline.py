"""教材处理流水线：OCR → 版面检测 → 表格还原 → 定向纠错 → 版面结构化 → 建索引。

每一步都是独立子进程，实时解析输出并把阶段进度写进 Task，供本地网站显示。
子进程使用两套虚拟环境（与 README 一致）：

  .venv-ocr    Python 3.14 + rapidocr（OCR / 纠错 / 结构化）
  .venv-ocr312 Python 3.12 + paddlepaddle-gpu（版面 / 表格）

环境变量可覆盖：MEDICAL_RAG_PYTHON、MEDICAL_RAG_OCR_PYTHON、MEDICAL_RAG_PADDLE_PYTHON。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import fitz

from .pdftext import cached_analyze, kind_label
from .tasks import Task, TaskCancelled
from .workspace import safe_dir_name

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = DEFAULT_ROOT

STAGES: list[tuple[str, str]] = [
    ("text", "文字层解析与结构化"),
    ("ocr", "OCR 文字识别"),
    ("layout", "版面/表格页检测"),
    ("tables", "表格结构还原"),
    ("fix", "OCR 定向纠错"),
    ("structure", "版面结构化"),
    ("index", "建立检索索引"),
]
STAGE_KEYS = [key for key, _ in STAGES]
STAGE_LABELS = dict(STAGES)

_OCR_PROGRESS = re.compile(r"^(\d+)/(\d+)\s+第(\d+)页")
_LAYOUT_PROGRESS = re.compile(r"^\s*(\d+)/(\d+)\s+\S")
_TABLE_PAGE_DONE = re.compile(r"^\[第(\d+)页\]")
_FIX_DONE = re.compile(r"修改文件数[:：]\s*(\d+)")
_CHAPTER_DONE = re.compile(r"^第(\d+)章[\s:：]")
_INDEX_DONE = re.compile(r"已处理《")


@dataclass
class PipelineContext:
    """一次教材处理所需的全部上下文。"""

    root: Path
    book_dir: Path
    pdf: Path
    title: str
    dpi: int = 300
    device: str = "gpu"
    reset_ocr: bool = False
    python: str = field(default_factory=lambda: sys.executable)
    ocr_python: str | None = None
    paddle_python: str | None = None
    db: Path | None = None

    @property
    def text_dir(self) -> Path:
        return self.book_dir / "text_v3"

    @property
    def processed_dir(self) -> Path:
        return self.book_dir / "processed_v3"

    @classmethod
    def from_book(
        cls,
        root: Path | str,
        book: dict[str, Any],
        dpi: int = 300,
        device: str = "gpu",
        reset_ocr: bool = False,
    ) -> "PipelineContext":
        if not book.get("pdf"):
            raise ValueError(f"《{book.get('title', '?')}》没有找到可处理的 PDF")
        return cls(
            root=Path(root).resolve(),
            book_dir=Path(book["dir"]).resolve(),
            pdf=Path(book["pdf"]).resolve(),
            title=str(book.get("index_title") or book.get("title") or Path(book["pdf"]).stem),
            dpi=dpi,
            device=device,
            reset_ocr=reset_ocr,
        )

    def resolve_interpreters(self) -> None:
        """确定每一步使用的 Python：环境变量 > 项目虚拟环境 > 当前解释器。"""
        env = os.environ
        self.python = env.get("MEDICAL_RAG_PYTHON") or sys.executable
        self.ocr_python = (
            env.get("MEDICAL_RAG_OCR_PYTHON")
            or venv_python(self.root, "ocr")
            or self.python
        )
        self.paddle_python = (
            env.get("MEDICAL_RAG_PADDLE_PYTHON")
            or venv_python(self.root, "ocr312")
            or self.ocr_python
        )


def venv_python(root: Path, name: str) -> str | None:
    base = root / f".venv-{name}"
    candidates = [
        base / "Scripts" / "python.exe",
        base / "bin" / "python3",
        base / "bin" / "python",
    ]
    return next((str(path) for path in candidates if path.exists()), None)


def _count(directory: Path, pattern: str) -> int:
    return sum(1 for _ in directory.glob(pattern)) if directory.is_dir() else 0


def count_boxes(ctx: PipelineContext) -> int:
    return _count(ctx.text_dir / "boxes", "page-*.json")


def book_pages(ctx: PipelineContext) -> int:
    """教材总页数：quality.json → manifest.json → OCR 结果页数。"""
    quality = ctx.processed_dir / "quality.json"
    if quality.exists():
        try:
            pages = json.loads(quality.read_text(encoding="utf-8")).get("pages")
            if isinstance(pages, int) and pages > 0:
                return pages
        except (OSError, json.JSONDecodeError):
            pass
    manifest = ctx.text_dir / "manifest.json"
    if manifest.exists():
        try:
            pages = json.loads(manifest.read_text(encoding="utf-8")).get("pages")
            if isinstance(pages, list) and pages:
                return len(pages)
        except (OSError, json.JSONDecodeError):
            pass
    return count_boxes(ctx)


def table_pages(ctx: PipelineContext) -> list[int]:
    path = ctx.text_dir / "layout.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return sorted(int(page) for page in data.get("table_pages", []))
    except (OSError, ValueError, TypeError):
        return []


def chapter_total(ctx: PipelineContext) -> int:
    path = ctx.book_dir / "chapters.json"
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    items = data.get("chapters", []) if isinstance(data, dict) else data
    return len(items) if isinstance(items, list) else 0


def plan_stages(status: dict[str, Any]) -> list[str]:
    """根据教材状态与工作流计算还需要运行的阶段（“自动”模式）。

    - workflow == "text"（PDF 自带文字层）：text → index，跳过 OCR；
    - 其余（图片型/混合型）：走原有 OCR 流水线。
    """
    workflow = (status or {}).get("workflow", "ocr")
    if workflow == "text":
        stages: list[str] = []
        if not status["structure"]["complete"]:
            stages.append("text")
        if stages or not status["index"]["complete"]:
            stages.append("index")
        if not stages:  # 已全部完成：文字层重建很快，默认重做一遍
            stages = ["text", "index"]
        return stages

    stages = []
    if not status["ocr"]["complete"]:
        stages.append("ocr")
    if not status["layout"]["complete"]:
        stages.append("layout")
    if not status["tables"]["complete"]:
        stages.append("tables")
    if any(stage in stages for stage in ("ocr", "tables")):
        stages.append("fix")
    if stages or not status["structure"]["complete"]:
        stages.append("structure")
    if stages or not status["index"]["complete"]:
        stages.append("index")
    if not stages:  # 已全部完成：默认做一次轻量重建（纠错 + 结构化 + 索引）
        stages = ["fix", "structure", "index"]
    return stages


def normalize_stages(stages: list[str] | str | None, status: dict[str, Any] | None = None) -> list[str]:
    """归一化阶段列表：支持 "auto"、乱序、去重，并按流水线顺序排序。"""
    if stages in (None, "auto", "all"):
        if stages == "all" or not status:
            return list(STAGE_KEYS)
        return plan_stages(status)
    wanted = set(stages) if not isinstance(stages, str) else {stages}
    unknown = wanted - set(STAGE_KEYS)
    if unknown:
        raise ValueError(f"未知阶段：{', '.join(sorted(unknown))}")
    return [key for key in STAGE_KEYS if key in wanted]


def stage_command(ctx: PipelineContext, stage: str, force: bool = False) -> list[str]:
    root = ctx.root
    if stage == "text":
        return [ctx.python, "-m", "medical_rag.pdftext", str(ctx.pdf), str(ctx.book_dir)]
    if stage == "ocr":
        command = [
            ctx.ocr_python or ctx.python,
            str(PACKAGE_ROOT / "tools_ocr_v3.py"),
            str(ctx.pdf),
            str(ctx.text_dir),
            "--dpi",
            str(ctx.dpi),
        ]
        if force or ctx.reset_ocr:
            command.append("--reset")
        return command
    if stage == "layout":
        return [
            ctx.paddle_python or ctx.python,
            str(PACKAGE_ROOT / "tools_layout_v3.py"),
            str(ctx.pdf),
            str(ctx.text_dir / "layout.json"),
            "--dpi",
            str(ctx.dpi),
            "--device",
            ctx.device,
            "--cache",
            str(root / ".cache" / f"render_layout_{safe_dir_name(ctx.book_dir.name)}"),
        ]
    if stage == "tables":
        return [
            ctx.paddle_python or ctx.python,
            str(PACKAGE_ROOT / "tools_table_v3_gpu.py"),
            str(ctx.pdf),
            str(ctx.text_dir),
            "--layout",
            str(ctx.text_dir / "layout.json"),
            "--cache",
            str(root / ".cache" / f"render300_{safe_dir_name(ctx.book_dir.name)}"),
            "--device",
            ctx.device,
        ]
    if stage == "fix":
        return [
            ctx.ocr_python or ctx.python,
            str(PACKAGE_ROOT / "tools_fix_ocr_v3.py"),
            "--target",
            str(ctx.text_dir),
        ]
    if stage == "structure":
        command = [
            ctx.python,
            str(PACKAGE_ROOT / "tools_structure_v3.py"),
            "--book-dir",
            str(ctx.book_dir),
        ]
        total = book_pages(ctx)
        if total:
            command += ["--total-pages", str(total)]
        return command
    if stage == "index":
        return [
            ctx.python,
            "-m",
            "medical_rag.cli",
            "ingest-text",
            str(ctx.processed_dir),
            "--title",
            ctx.title,
        ]
    raise ValueError(f"未知阶段：{stage}")


def _check_prerequisites(ctx: PipelineContext, stage: str) -> None:
    if stage == "text":
        if not ctx.pdf.exists():
            raise FileNotFoundError(f"PDF 不存在：{ctx.pdf}")
        info = cached_analyze(ctx.pdf, ctx.root / ".medical_rag" / "pdf_text")
        if info.get("kind") != "text":
            raise RuntimeError(
                f"该 PDF 没有可用的文字层（{kind_label(info.get('kind'))}，"
                f"{info.get('text_pages', 0)}/{info.get('pages', 0)} 页有文字），请改用 OCR 阶段"
            )
    elif stage == "ocr":
        if not ctx.pdf.exists():
            raise FileNotFoundError(f"PDF 不存在：{ctx.pdf}")
    elif stage == "layout":
        if count_boxes(ctx) == 0:
            raise RuntimeError("需要先完成 OCR：text_v3/boxes 为空")
    elif stage == "tables":
        if not (ctx.text_dir / "layout.json").exists():
            raise RuntimeError("需要先完成版面检测：text_v3/layout.json 不存在")
    elif stage == "fix":
        if count_boxes(ctx) == 0 and _count(ctx.text_dir / "tables", "page-*.md") == 0:
            raise RuntimeError("需要先完成 OCR 或表格还原：text_v3 为空")
    elif stage == "structure":
        if count_boxes(ctx) == 0:
            raise RuntimeError("需要先完成 OCR：text_v3/boxes 为空")
    elif stage == "index":
        if not (ctx.processed_dir / "structured").is_dir():
            raise RuntimeError("需要先生成结构化目录：processed_v3/structured 不存在")


def _stage_total_hint(ctx: PipelineContext, stage: str) -> int:
    if stage == "text":
        pages = book_pages(ctx)
        return pages if pages else _count(ctx.text_dir / "boxes", "page-*.json")
    if stage == "ocr":
        return max(0, book_pages(ctx) - count_boxes(ctx)) or count_boxes(ctx)
    if stage == "layout":
        return count_boxes(ctx)
    if stage == "tables":
        return len(table_pages(ctx))
    if stage == "structure":
        return chapter_total(ctx)
    if stage in ("fix", "index"):
        return 1
    return 0


def _progress_from_line(ctx: PipelineContext, stage: str, line: str, tables_total: int) -> tuple[int, int, str] | None:
    if stage in ("ocr", "text"):
        match = _OCR_PROGRESS.match(line)
        if match:
            return int(match.group(1)), int(match.group(2)), f"第 {match.group(3)} 页"
    elif stage == "layout":
        match = _LAYOUT_PROGRESS.match(line)
        if match:
            return int(match.group(1)), int(match.group(2)), f"第 {match.group(1)} 页"
    elif stage == "tables":
        match = _TABLE_PAGE_DONE.match(line)
        if match:
            page = int(match.group(1))
            pages = table_pages(ctx)
            position = pages.index(page) + 1 if page in pages else None
            if position is not None:
                return position, len(pages), f"第 {page} 页"
    elif stage == "fix":
        match = _FIX_DONE.search(line)
        if match:
            return 1, 1, f"修改 {match.group(1)} 个文件"
    elif stage == "structure":
        match = _CHAPTER_DONE.match(line)
        if match:
            total = chapter_total(ctx)
            return int(match.group(1)), total or int(match.group(1)), line.split(":")[0]
    elif stage == "index":
        if _INDEX_DONE.search(line):
            return 1, 1, "索引完成"
    return None


def run_stage(task: Task, ctx: PipelineContext, stage: str, force: bool = False) -> None:
    """执行一个阶段；失败抛异常，取消抛 TaskCancelled。"""
    _check_prerequisites(ctx, stage)
    command = stage_command(ctx, stage, force=force)
    task.log("$ " + " ".join(str(part) for part in command))
    total_hint = _stage_total_hint(ctx, stage)
    task.start_stage(stage, STAGE_LABELS[stage], total=total_hint, message="启动中…")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    # 同时加入数据根目录与包所在目录，保证临时数据目录下也能 import medical_rag
    paths = [str(ctx.root)]
    if str(PACKAGE_ROOT) not in paths:
        paths.append(str(PACKAGE_ROOT))
    existing_path = env.get("PYTHONPATH", "")
    if existing_path:
        paths.append(existing_path)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if ctx.db is not None:  # 让 ingest-text 子进程写入同一索引库
        env["MEDICAL_RAG_DB"] = str(ctx.db)

    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(ctx.root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        **kwargs,
    )
    task.attach_process(process)
    tables_total = len(table_pages(ctx))
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            if task.cancelled:
                raise TaskCancelled()
            line = raw_line.rstrip()
            if line:
                task.log(line)
            update = _progress_from_line(ctx, stage, line, tables_total)
            if update is not None:
                done, total, message = update
                task.progress(done, total, message)
        code = process.wait()
    finally:
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        if process.poll() is None:
            try:
                process.kill()
            except OSError:
                pass
    if task.cancelled:
        raise TaskCancelled()
    if code != 0:
        raise RuntimeError(f"{STAGE_LABELS[stage]}失败（退出码 {code}），详见日志")
    task.finish_stage()


def run_book_pipeline(
    task: Task,
    root: Path | str,
    book: dict[str, Any],
    stages: list[str] | str | None = None,
    force: bool = False,
    dpi: int = 300,
    device: str = "gpu",
    reset_ocr: bool = False,
    db: Path | str | None = None,
) -> list[str]:
    """按顺序运行指定阶段（stages="auto" 时按教材状态自动挑选）。"""
    root = Path(root).resolve()
    ctx = PipelineContext.from_book(root, book, dpi=dpi, device=device, reset_ocr=reset_ocr)
    if db is not None:
        ctx.db = Path(db).expanduser().resolve()
    ctx.resolve_interpreters()
    status = book.get("status") or None
    selected = normalize_stages(stages, status)
    task.set_stages([(key, STAGE_LABELS[key]) for key in selected])
    task.log(f"教材目录：{ctx.book_dir}")
    task.log(f"PDF：{ctx.pdf}（{_human_size(ctx.pdf.stat().st_size) if ctx.pdf.exists() else '缺失'}）")
    task.log(f"阶段：{' → '.join(f'{STAGE_LABELS[key]}' for key in selected)}")
    task.log(f"解释器：主 {ctx.python} | OCR {ctx.ocr_python} | Paddle {ctx.paddle_python}")
    for stage in selected:
        task.check_cancelled()
        run_stage(task, ctx, stage, force=force)
    return selected


PDF_MAGIC = b"%PDF-"
MAX_PDF_PAGES = 6000  # 单本教材页数上限，防止压缩炸弹/误传超大文件


def validate_pdf(
    path: Path | str,
    label: str = "文件",
    max_pages: int | None = MAX_PDF_PAGES,
) -> dict[str, Any]:
    """确认文件是可正常打开的 PDF，否则抛 ``ValueError``。

    校验链：非空 → 带 ``%PDF-`` 文件头 → fitz 能打开 → 未加密 → 页数 > 0
    → 页数不超上限 → 末页可解析（能暴露截断/损坏）。调用方拿到异常后
    应删除临时文件并把任务标记为 error，而不是只写日志。
    """
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"{label}不可读：{exc}") from exc
    if size == 0:
        raise ValueError(f"{label}是空文件（0 字节），不是有效的 PDF")
    try:
        with path.open("rb") as stream:
            head = stream.read(1024)
    except OSError as exc:
        raise ValueError(f"{label}不可读：{exc}") from exc
    if PDF_MAGIC not in head:
        raise ValueError(f"{label}不是 PDF（文件头缺少 %PDF-）")
    try:
        with fitz.open(path) as doc:
            if doc.needs_pass:
                raise ValueError(f"{label}已加密，需要密码才能打开")
            pages = len(doc)
            if pages <= 0:
                raise ValueError(f"{label}没有任何页面（0 页）")
            if max_pages is not None and pages > max_pages:
                raise ValueError(f"{label}共 {pages} 页，超过单本上限 {max_pages} 页")
            doc[pages - 1].get_text()  # 触发末页解析，暴露截断/损坏的 PDF
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - fitz 对损坏文件抛的异常类型不稳定
        raise ValueError(f"{label}不是有效的 PDF，或文件已损坏（{type(exc).__name__}: {exc}）") from exc
    return {"pages": pages, "size": size}


def _remove_empty(paths: list[Path]) -> None:
    """删除导入失败后残留的空目录（从内到外），已含文件的目录保留。"""
    for path in paths:
        try:
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        except OSError:
            pass


def _log_pdf_kind(task: Task, root: Path, pdf: Path) -> dict[str, Any] | None:
    """导入完成后立刻检测文字层，把结论写进任务日志。

    走到这里时 ``validate_pdf`` 已确认文件是有效 PDF，因此检测失败（例如缓存
    目录不可写）只记录告警，不影响导入结果。
    """
    try:
        info = cached_analyze(pdf, root / ".medical_rag" / "pdf_text")
    except Exception as exc:  # noqa: BLE001 - 已确认是有效 PDF，检测失败不阻断导入
        task.log(f"警告：PDF 文字层检测失败（文件本身有效）：{exc}")
        return None
    task.log(
        f"PDF 检测：{kind_label(info.get('kind'))}，"
        f"{info.get('text_pages', 0)}/{info.get('pages', 0)} 页有文字，共 {info.get('chars', 0)} 字"
    )
    if info.get("kind") == "text":
        task.log("建议：直接运行「文字层解析 + 建立索引」，无需 OCR")
    else:
        task.log("建议：使用 OCR 流水线（OCR → 版面 → 表格 → 纪错 → 结构化 → 索引）")
    return info


def import_pdf(
    task: Task,
    root: Path | str,
    source: Path | str,
    title: str | None = None,
) -> tuple[Path, Path]:
    """导入一本教材：把 PDF 复制到 ``res/<书名>/PDF/`` 下。

    返回 (book_dir, pdf_path)。同名 PDF 允许覆盖；目录里已有其他 PDF 时拒绝，
    避免同一本教材出现两个主文件。

    导入前用 :func:`validate_pdf` 校验源文件，复制完成后再次校验副本；任一失败
    都会抛错（任务标记为 error）并清理 .part 与刚建的空目录。
    """
    root = Path(root).resolve()
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"文件不存在：{source}")
    if source.is_dir():
        raise ValueError(f"这是一个目录，不是 PDF 文件：{source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError("目前只支持 PDF 文件")

    # 先校验源文件，避免无效文件走到一半才失败、留下空的 res/<书名>/ 目录
    source_info = validate_pdf(source, label=f"源文件 {source.name}")

    name = safe_dir_name(title or source.stem)
    book_dir = root / "res" / name
    pdf_dir = book_dir / "PDF"
    target = pdf_dir / source.name

    existing = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.is_dir() else []
    others = [path for path in existing if path.name != target.name]
    if others:
        raise FileExistsError(
            f"目录 res/{name} 已存在其他 PDF：{others[0].name}；请换一个书名或先删除旧目录"
        )

    task.set_progress(0, 1, f"准备导入《{name}》")
    if source == target:
        task.log(f"PDF 已在目标位置：{target}")
        _log_pdf_kind(task, root, target)
        task.set_progress(1, 1, "无需复制")
        return book_dir, target

    pdf_dir.mkdir(parents=True, exist_ok=True)
    size = source_info["size"]
    temp = target.with_name(target.name + ".part")
    temp.unlink(missing_ok=True)
    task.log(f"复制 PDF：{source} → {target}（{_human_size(size)}，{source_info['pages']} 页）")
    done = 0
    try:
        with source.open("rb") as src, temp.open("wb") as dst:
            while True:
                task.check_cancelled()
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                done += len(chunk)
                task.set_progress(done, size, f"{_human_size(done)} / {_human_size(size)}")
        # 复制完成后再校验副本：磁盘写满/中途截断都会在这里被拦下
        validate_pdf(temp, label=f"复制到 {target.name} 的文件")
        os.replace(temp, target)
    except BaseException as exc:
        temp.unlink(missing_ok=True)
        _remove_empty([pdf_dir, book_dir])
        if isinstance(exc, TaskCancelled):
            raise
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"导入失败：{type(exc).__name__}: {exc}") from exc
    task.set_progress(1, 1, "导入完成")
    _log_pdf_kind(task, root, target)
    task.log("导入完成")
    return book_dir, target


def _human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"
