"""环境自检：解释器、依赖、索引库与工作区状态。

被 ``medical-rag doctor``（CLI）与 ``GET /api/health``（网页）共用，只做只读检查。
默认不启动子解释器（快）；``--deep`` / ``deep=True`` 时会额外验证 OCR / Paddle
依赖是否真的可导入（慢，约数秒）。

注意：``/api/health`` 与其它 ``/api/*`` 一样受访问令牌保护，避免把本机路径暴露
给局域网。
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .pipeline import DEFAULT_ROOT, venv_python

MIN_PYTHON = (3, 10)
CORE_PACKAGES = ("PyMuPDF",)
OCR_PACKAGES = ("rapidocr", "onnxruntime-gpu", "opencv-python", "numpy", "pillow", "shapely", "pyclipper")
PADDLE_PACKAGES = ("paddlepaddle-gpu", "paddleocr", "paddlex")
OCR_REQUIREMENTS = "requirements-ocr.txt"
PADDLE_REQUIREMENTS = "requirements-paddle.txt"
DEEP_TIMEOUT = 180


def package_version(name: str) -> str | None:
    """已安装包的版本号；未安装返回 None。"""
    try:
        return metadata.version(name)
    except Exception:  # noqa: BLE001 - 包未安装/元数据损坏都按“未安装”处理
        return None


def interpreter_info(root: Path | str | None = DEFAULT_ROOT) -> dict[str, Any]:
    """三步流水线各阶段实际使用的 Python（环境变量 > 项目虚拟环境 > 当前解释器）。"""
    root = Path(root) if root else Path(DEFAULT_ROOT)
    ocr = os.environ.get("MEDICAL_RAG_OCR_PYTHON") or venv_python(root, "ocr") or sys.executable
    paddle = os.environ.get("MEDICAL_RAG_PADDLE_PYTHON") or venv_python(root, "ocr312") or ocr
    return {
        "python": os.environ.get("MEDICAL_RAG_PYTHON") or sys.executable,
        "ocr_python": ocr,
        "paddle_python": paddle,
        "venv_ocr": venv_python(root, "ocr") is not None,
        "venv_ocr312": venv_python(root, "ocr312") is not None,
    }


def _check(name: str, status: str, detail: str, hint: str = "") -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


# 在目标解释器里查包版本（不 import 项目代码，仅读元数据）
_VERSION_QUERY = (
    "import importlib.metadata as m, json, sys\n"
    "names = json.loads(sys.argv[1])\n"
    "out = {}\n"
    "for n in names:\n"
    "    try:\n"
    "        out[n] = m.version(n)\n"
    "    except Exception:\n"
    "        out[n] = None\n"
    "print(json.dumps(out))\n"
)


def interpreter_package_versions(python: str, packages: tuple[str, ...]) -> dict[str, str | None]:
    """查询**指定解释器**里的包版本。

    不能直接用当前进程的 importlib.metadata：doctor 跑在主环境（如 .venv-ocr）
    里，但 Paddle 依赖装在 .venv-ocr312 中，直接查会把已安装的包报成缺失。
    实测子进程查询约 88 ms/次，默认路径可承受。
    """
    if not python:
        return {name: None for name in packages}
    if os.path.normcase(str(python)) == os.path.normcase(sys.executable):
        return {name: package_version(name) for name in packages}
    try:
        done = subprocess.run(
            [python, "-c", _VERSION_QUERY, json.dumps(list(packages))],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if done.returncode == 0:
            lines = [line for line in (done.stdout or "").splitlines() if line.strip()]
            data = json.loads(lines[-1])
            return {name: data.get(name) for name in packages}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return {name: None for name in packages}


def _package_check(
    label: str,
    packages: tuple[str, ...],
    requirements: str,
    install_hint: str,
    python: str | None = None,
) -> dict[str, str]:
    found = interpreter_package_versions(python or sys.executable, packages)
    missing = [name for name, version in found.items() if version is None]
    detail = "、".join(f"{name} {version}" for name, version in found.items() if version)
    if missing:
        return _check(
            label,
            "warn",
            f"{detail or '无'}；缺少 {'、'.join(missing)}",
            f"{install_hint}：pip install -r {requirements}",
        )
    return _check(label, "ok", detail or "已安装")


def _probe_import(python: str, modules: tuple[str, ...]) -> tuple[bool, str]:
    """在指定解释器里真实 import 一次（deep 模式用）。"""
    code = (
        "import importlib,sys\n"
        f"mods={list(modules)!r}\n"
        "bad=[m for m in mods if importlib.util.find_spec(m) is None]\n"
        "print('MISSING:'+','.join(bad) if bad else 'OK')\n"
    )
    try:
        done = subprocess.run(
            [python, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=DEEP_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"无法运行 {python}：{exc}"
    output = (done.stdout or "").strip().splitlines()
    last = output[-1] if output else ""
    if done.returncode != 0:
        return False, f"退出码 {done.returncode}：{(done.stderr or '').strip()[:200]}"
    if last == "OK":
        return True, "可导入"
    return False, last[8:] if last.startswith("MISSING:") else last


def _database_report(db_path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    if not db_path.exists():
        return (
            {"path": str(db_path), "exists": False},
            [_check("索引库", "warn", f"尚未建立：{db_path}", "先导入并处理一本教材")],
        )
    from .library import Library

    try:
        library = Library(db_path)
    except Exception as exc:  # noqa: BLE001 - 打不开就是 error
        return (
            {"path": str(db_path), "exists": True, "error": str(exc)},
            [_check("索引库", "error", f"打不开：{exc}")],
        )
    try:
        books = library.list_books()
        chunks = int(library.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        integrity = library.fts_integrity_ok()
        stale = library.stale_books()
        report = {
            "path": str(db_path),
            "exists": True,
            "books": len(books),
            "chunks": chunks,
            "fts_integrity": integrity,
            "embedding_backend": getattr(library, "embedding", None) and library.embedding.name,
            "embedding_warning": getattr(library, "embedding_warning", ""),
            "stale_books": [
                {"title": item.get("title"), "reason": item.get("reason")} for item in stale
            ],
            "titles": [book["title"] for book in books],
        }
    finally:
        library.close()
    checks = [
        _check(
            "索引库",
            "ok",
            f"{len(books)} 本教材 / {chunks} 个证据块（{db_path.name}）",
        ),
        _check(
            "FTS 索引一致性",
            "ok" if integrity else "error",
            "一致" if integrity else "chunks 与 chunks_fts 不一致（历史版本遗留）",
            "" if integrity else "运行 medical-rag doctor --repair 或重建索引修复",
        ),
        _check(
            "向量后端",
            "warn" if library.embedding_warning else "ok",
            library.embedding.name
            + (f"（{library.embedding_warning}）" if library.embedding_warning else ""),
            ""
            if not library.embedding_warning
            else "语义后端不可用属正常：检索会自动退回词法后端，不影响可用性",
        ),
        _check(
            "索引时效性",
            "warn" if stale else "ok",
            f"{len(stale)} 本教材的索引可能过期" if stale else "索引与源 PDF 一致",
            "；".join(f"{item.get('title')}：{item.get('reason')}" for item in stale[:3]) if stale else "",
        ),
    ]
    return report, checks


def _workspace_report(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    from .workspace import find_pdf

    res_dir = root / "res"
    book_dirs = sorted(path for path in res_dir.iterdir() if path.is_dir()) if res_dir.is_dir() else []
    with_pdf = [path for path in book_dirs if find_pdf(path) is not None]
    report = {
        "root": str(root),
        "res_exists": res_dir.is_dir(),
        "book_dirs": len(book_dirs),
        "book_dirs_with_pdf": len(with_pdf),
    }
    if not res_dir.is_dir():
        return report, [_check("工作区", "warn", f"没有 {res_dir}，尚未导入任何教材")]
    return report, [
        _check("工作区", "ok", f"{len(book_dirs)} 个教材目录，其中 {len(with_pdf)} 个含 PDF")
    ]


def _resolve_paths(root: Path | str | None, db: Path | str | None) -> tuple[Path, Path]:
    """统一处理 None（CLI 不传 --root/--db 时），并解析成绝对路径。"""
    base = Path(root).expanduser().resolve() if root else Path(DEFAULT_ROOT).resolve()
    db_path = Path(db).expanduser().resolve() if db else base / ".medical_rag" / "library.sqlite3"
    return base, db_path


def check_environment(
    root: Path | str | None = DEFAULT_ROOT,
    db: Path | str | None = None,
    deep: bool = False,
) -> dict[str, Any]:
    """汇总环境检查；``ok`` 为 False 表示存在必须处理的 error 项。"""
    root, db_path = _resolve_paths(root, db)
    checks: list[dict[str, str]] = []

    version = sys.version_info
    checks.append(
        _check(
            "Python 版本",
            "ok" if version >= MIN_PYTHON else "error",
            f"{sys.version.split()[0]}（需要 >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}）",
        )
    )
    checks.append(_package_check("核心依赖", CORE_PACKAGES, "requirements-core.txt", "核心依赖缺失"))
    interpreters = interpreter_info(root)
    if interpreters["venv_ocr"]:
        checks.append(_check("OCR 解释器", "ok", interpreters["ocr_python"]))
    else:
        checks.append(
            _check(
                "OCR 解释器",
                "warn",
                f"未找到 .venv-ocr，回退到 {interpreters['ocr_python']}",
                f"OCR 流水线需要：python -m venv .venv-ocr && pip install -r {OCR_REQUIREMENTS}",
            )
        )
    if interpreters["venv_ocr312"]:
        checks.append(_check("Paddle 解释器", "ok", interpreters["paddle_python"]))
    else:
        checks.append(
            _check(
                "Paddle 解释器",
                "warn",
                f"未找到 .venv-ocr312，回退到 {interpreters['paddle_python']}",
                f"版面/表格需要：python -m venv .venv-ocr312 && pip install -r {PADDLE_REQUIREMENTS}",
            )
        )

    # 依赖必须查"目标解释器"：检查环境跑在主环境里，而 Paddle 依赖装在 .venv-ocr312
    if interpreters["venv_ocr"]:
        checks.append(
            _package_check(
                "OCR 依赖",
                OCR_PACKAGES,
                OCR_REQUIREMENTS,
                "OCR 依赖缺失",
                python=interpreters["ocr_python"],
            )
        )
    if interpreters["venv_ocr312"]:
        checks.append(
            _package_check(
                "Paddle 依赖",
                PADDLE_PACKAGES,
                PADDLE_REQUIREMENTS,
                "Paddle 依赖缺失",
                python=interpreters["paddle_python"],
            )
        )

    if deep:
        ocr_ok, ocr_detail = _probe_import(interpreters["ocr_python"], ("rapidocr", "onnxruntime", "cv2"))
        checks.append(
            _check("OCR 依赖可导入", "ok" if ocr_ok else "error", ocr_detail, "" if ocr_ok else f"见 {OCR_REQUIREMENTS}")
        )
        if interpreters["venv_ocr312"]:
            pad_ok, pad_detail = _probe_import(interpreters["paddle_python"], ("paddle", "paddleocr", "paddlex"))
            checks.append(
                _check(
                    "Paddle 依赖可导入",
                    "ok" if pad_ok else "error",
                    pad_detail,
                    "" if pad_ok else f"见 {PADDLE_REQUIREMENTS}",
                )
            )

    database, db_checks = _database_report(db_path)
    workspace, ws_checks = _workspace_report(root)
    checks.extend(db_checks)
    checks.extend(ws_checks)

    errors = sum(1 for item in checks if item["status"] == "error")
    warnings = sum(1 for item in checks if item["status"] == "warn")
    return {
        "ok": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "root": str(root),
        "db": str(db_path),
        "deep": bool(deep),
        "interpreters": interpreters,
        "database": database,
        "workspace": workspace,
        "checks": checks,
    }


def format_report(report: dict[str, Any]) -> str:
    """把检查报告渲染成给终端看的多行文本。"""
    icons = {"ok": "ok  ", "warn": "warn", "error": "错误"}
    lines = ["环境自检"]
    for item in report["checks"]:
        lines.append(f"  [{icons.get(item['status'], item['status'])}] {item['name']}：{item['detail']}")
        if item["hint"]:
            lines.append(f"           → {item['hint']}")
    lines.append(
        f"结论：{report['errors']} 项错误，{report['warnings']} 项警告"
        + ("（可以正常使用）" if report["ok"] else "（请先处理错误项）")
    )
    return "\n".join(lines)


def repair(root: Path | str | None = DEFAULT_ROOT, db: Path | str | None = None) -> dict[str, Any]:
    """重建 FTS 索引，修复 chunks / chunks_fts 不一致（历史版本遗留）。"""
    from .library import Library

    _, db_path = _resolve_paths(root, db)
    if not db_path.exists():
        return {"repaired": False, "reason": f"索引库不存在：{db_path}"}
    library = Library(db_path)
    try:
        before = library.fts_integrity_ok()
        chunks = library.rebuild_fts_index()
        after = library.fts_integrity_ok()
    finally:
        library.close()
    return {"repaired": bool(after), "before_ok": before, "after_ok": after, "chunks": chunks, "db": str(db_path)}
