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
OCR_REQUIREMENTS = "依赖-OCR识别_requirements-ocr.txt"
PADDLE_REQUIREMENTS = "依赖-版面表格_requirements-paddle.txt"
CORE_REQUIREMENTS = "依赖-核心_requirements-core.txt"
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
        "import sys\n"
        f"mods={list(modules)!r}\n"
        "bad=[]\n"
        "for m in mods:\n"
        "    try:\n"
        "        __import__(m)\n"
        "    except Exception as exc:\n"
        "        bad.append(f\"{m} ({type(exc).__name__}: {exc})\")\n"
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


def _stale_reason(item: dict[str, Any], base: Path) -> str:
    """索引是否落后于源 PDF：与 ``Library.book_freshness`` 同一套 mtime+size 判据。"""
    from .paths import absolute_path

    if item.get("content_hash") is None:
        return "索引没有来源指纹（旧版本建立），建议重建索引"
    source = absolute_path(base, item.get("source_path"))
    if source is None or not source.exists():
        return f"源 PDF 不存在：{item.get('source_path')}"
    try:
        stat = source.stat()
    except OSError as exc:
        return f"源 PDF 不可读：{exc}"
    mtime_changed = item.get("source_mtime") is None or abs(
        float(item["source_mtime"]) - stat.st_mtime
    ) > 1e-6
    size_changed = item.get("source_size") is None or int(item["source_size"]) != stat.st_size
    if mtime_changed or size_changed:
        return "源 PDF 已变化（mtime/大小不同），索引可能过期"
    return ""


def _database_report(
    db_path: Path, root: Path | str | None = None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """只用一条**只读**连接读出索引库报告。

    ``Library()`` 的构造器会执行 DDL 与 ``PRAGMA user_version``，只读介质/受限
    ACL 下直接失败，会把健康库判成 error（CLI 退出码 1、``/api/health`` 503）。
    这里改成只读 URI 连接（``immutable=1`` 回退），深检做不了只降级成 warn。
    """
    if not db_path.exists():
        return (
            {"path": str(db_path), "exists": False},
            [_check("索引库", "warn", f"尚未建立：{db_path}", "先导入并处理一本教材")],
        )
    import sqlite3

    from .library import _normalise
    from .workspace import open_readonly

    base = Path(root).expanduser().resolve() if root else db_path.parent.parent
    try:
        connection = open_readonly(db_path)
    except sqlite3.Error as exc:
        # 打不开只读库是「读不了」而不是「库坏了」：warning，避免健康检查假报错
        return (
            {"path": str(db_path), "exists": True, "error": str(exc)},
            [
                _check(
                    "索引库",
                    "warn",
                    f"只读打开失败（库可能被占用或权限受限）：{exc}",
                    "确认库文件权限；只读介质上可先复制一份再自检",
                )
            ],
        )
    try:
        book_rows = [dict(row) for row in connection.execute("SELECT * FROM books ORDER BY title")]
        chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    except sqlite3.Error as exc:
        connection.close()
        return (
            {"path": str(db_path), "exists": True, "error": str(exc)},
            [
                _check(
                    "索引库",
                    "warn",
                    f"只读读取失败（库可能被占用或权限受限）：{exc}",
                    "确认库文件权限；只读介质上可先复制一份再自检",
                )
            ],
        )
    # 文件级完整性：PRAGMA quick_check 只读即可执行，能抓出 b-tree/页级损坏。
    # 深检只比对索引行数，抓不到"库被写坏但行数还对"的情况；旧实现靠
    # chunks_fts 的 integrity-check，改成只读连接后那条路径不可用，所以补这一层。
    try:
        quick = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    except sqlite3.Error as exc:
        quick = f"quick_check 执行失败：{exc}"
    db_healthy = quick.strip().lower() == "ok"
    deep_error = ""
    try:
        # 深检：三套索引与 chunks 是否一致（只读可做，不触发任何写语句）
        indexed = int(connection.execute("SELECT COUNT(*) FROM chunks_fts_ngram").fetchone()[0])
        drift: list[dict[str, Any]] = []
        for row in connection.execute(
            "SELECT c.rowid AS rowid, c.text AS text, n.norm_text AS norm FROM chunks c "
            "JOIN chunks_fts_ngram n ON n.rowid = c.rowid"
        ):
            if _normalise(row["text"]) != row["norm"]:
                drift.append({"rowid": row["rowid"]})
                if len(drift) >= 3:
                    break
        drift_checked = True
    except sqlite3.Error as exc:
        indexed, drift, drift_checked = -1, [], False
        if "no such table" in str(exc).lower():
            # 老库可能没有二元组索引表：只读深检做不了，降级成 warn（不算错误）
            deep_error = "老库缺少二元组索引表"
        else:
            # 读失败 ≠ 缺表：可能是库损坏，必须记 error 而不是悄悄变成 warn
            deep_error = f"深检读失败：{exc}"
            db_healthy = False
    embedding_backend = ""
    embedding_warning = ""
    try:
        from .embeddings import create_backend_cached

        backend, embedding_warning = create_backend_cached(os.environ.get("MEDICAL_RAG_EMBEDDING"))
        embedding_backend = backend.name
        vectors = int(
            connection.execute(
                "SELECT COUNT(*) FROM chunk_vectors WHERE backend = ?", (embedding_backend,)
            ).fetchone()[0]
        )
    except Exception:  # noqa: BLE001 - 后端不可用属正常（检索会退回词法）
        vectors = int(connection.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0])
    connection.close()

    raw_paths = [row["path"] for row in book_rows]
    relative_paths = sum(1 for value in raw_paths if value and not Path(value).is_absolute())
    from .paths import absolute_path

    missing_books = []
    for row in book_rows:
        raw = row.get("path")
        if not raw:
            continue
        resolved = absolute_path(base, raw)
        if resolved is None or not resolved.exists():
            missing_books.append(row["title"])
    stale = [
        {"title": row["title"], "reason": reason}
        for row in book_rows
        if (reason := _stale_reason(row, base))
    ]
    fts_ok = db_healthy and drift_checked and indexed == chunks and not drift and vectors == chunks
    report = {
        "path": str(db_path),
        "exists": True,
        "books": len(book_rows),
        "chunks": chunks,
        "integrity_check": quick,
        "fts_integrity": fts_ok,
        "relative_paths": relative_paths,
        "missing_books": missing_books,
        "embedding_backend": embedding_backend,
        "embedding_warning": embedding_warning,
        "stale_books": [{"title": item["title"], "reason": item["reason"]} for item in stale],
        "titles": [row["title"] for row in book_rows],
        "readonly": True,
    }
    if not raw_paths:
        path_status, path_detail, path_hint = "warn", "索引库还没有教材记录", "先导入并处理一本教材"
    elif missing_books:
        path_status = "warn"
        path_detail = f"{len(missing_books)} 本教材的索引路径找不到：" + "、".join(missing_books[:3])
        path_hint = "确认 res/ 下教材目录还在；目录改过名就重新建立索引"
    else:
        path_status = "ok"
        path_detail = f"{relative_paths}/{len(raw_paths)} 本用相对路径保存，可随目录整体搬走"
        path_hint = ""
    if not db_healthy:
        integrity_check = _check(
            "索引库完整性",
            "error",
            f"PRAGMA quick_check 未通过：{quick[:120]}",
            "库文件可能损坏：先备份，再运行 medical-rag doctor --repair 或重建索引",
        )
    elif drift_checked:
        integrity_check = _check(
            "FTS 索引一致性",
            "ok" if fts_ok else "error",
            "一致"
            if fts_ok
            else "chunks 与 FTS/向量索引不一致（历史版本遗留）",
            "" if fts_ok else "运行 medical-rag doctor --repair 或重建索引修复",
        )
    else:
        integrity_check = _check(
            "FTS 索引一致性",
            "warn",
            f"只读模式下无法深检：{deep_error}",
            "需要完整校验时运行 medical-rag doctor --repair",
        )
    checks = [
        _check(
            "索引库",
            "ok",
            f"{len(book_rows)} 本教材 / {chunks} 个证据块（{db_path.name}）",
        ),
        integrity_check,
        _check(
            "向量后端",
            "warn" if embedding_warning else "ok",
            (embedding_backend or "未知后端")
            + (f"（{embedding_warning}）" if embedding_warning else ""),
            ""
            if not embedding_warning
            else "语义后端不可用属正常：检索会自动退回词法后端，不影响可用性",
        ),
        _check(
            "索引时效性",
            "warn" if stale else "ok",
            f"{len(stale)} 本教材的索引可能过期" if stale else "索引与源 PDF 一致",
            "；".join(f"{item['title']}：{item['reason']}" for item in stale[:3]) if stale else "",
        ),
        _check("索引路径", path_status, path_detail, path_hint),
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


def _output_access_check(root: Path) -> list[dict[str, str]]:
    """检查各教材 ``processed_v3`` 下的产物当前账号能不能动。

    Windows 上被别的安全上下文（其它账号、沙箱/Agent 环境）写过的产物会留下
    文件名但拒绝一切访问（读、改名、删除全部 ERROR_ACCESS_DENIED）。这种状态下
    生成流水线会在“把旧产物挪进备份”第一步就失败，且关掉资源管理器/预览器也
    没用；提前在自检里报出来，比等跑到一半失败好。
    """
    from .outputs import blocked_outputs

    res_dir = root / "res"
    if not res_dir.is_dir():
        return []
    problems: list[str] = []
    for book_dir in sorted(path for path in res_dir.iterdir() if path.is_dir()):
        target = book_dir / "processed_v3"
        if not target.is_dir():
            continue
        blocked = blocked_outputs(target)
        if blocked:
            problems.append(f"{book_dir.name}：{'、'.join(blocked)}")
    if not problems:
        return [_check("产物权限", "ok", "processed_v3 下的产物读写正常")]
    return [
        _check(
            "产物权限",
            "warn",
            f"{'；'.join(problems)} 拒绝访问（名字还在，但读/改名/删除都被拒绝）",
            "这些产物只能换掉不能覆盖：先试管理员 takeown /f \"<processed_v3>\" /r /d y 再 "
            "icacls \"<processed_v3>\" /reset /t /c /q；不需要旧产物的话，直接把 processed_v3 "
            "改名搁置（例如 processed_v3.broken），再重新生成即可",
        )
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
    checks.append(_package_check("核心依赖", CORE_PACKAGES, CORE_REQUIREMENTS, "核心依赖缺失"))
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

    # 目录被移动/换电脑后，venv 里的 python.exe 只是引导器：它按 pyvenv.cfg 里
    # 写死的基础 Python（如 C:\Python314）启动。这里真跑一次，比只看文件存在可靠。
    for flag, key, label, folder in (
        ("venv_ocr", "ocr_python", "OCR 解释器可运行", ".venv-ocr"),
        ("venv_ocr312", "paddle_python", "Paddle 解释器可运行", ".venv-ocr312"),
    ):
        if not interpreters[flag]:
            continue
        runnable, detail = _probe_import(interpreters[key], ())
        checks.append(
            _check(
                label,
                "ok" if runnable else "error",
                interpreters[key] if runnable else f"{folder} 无法启动：{detail}",
                ""
                if runnable
                else f"{folder} 依赖写死的基础 Python，换电脑/换用户后必须重建（见 README 或 scripts\\重建虚拟环境_Rebuild-Venvs.bat）",
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

    database, db_checks = _database_report(db_path, root)
    workspace, ws_checks = _workspace_report(root)
    checks.extend(db_checks)
    checks.extend(ws_checks)
    checks.extend(_output_access_check(root))

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
    library = Library(db_path, root=_resolve_paths(root, db)[0])
    try:
        before = library.fts_integrity_ok()
        chunks = library.rebuild_fts_index()
        after = library.fts_integrity_ok()
    finally:
        library.close()
    return {"repaired": bool(after), "before_ok": before, "after_ok": after, "chunks": chunks, "db": str(db_path)}


def compact(root: Path | str | None = DEFAULT_ROOT, db: Path | str | None = None) -> dict[str, Any]:
    """压缩索引库：FTS5 ``optimize``（合并段）+ ``VACUUM``（回收空闲页）。

    重建索引会反复删除/插入整本书的块，FTS5 段与 SQLite 空闲页都会累积。
    实测重建后的库 22.6 MB，经 optimize + VACUUM 回到 15.5 MB；注意**只跑
    optimize 反而会先变大**（合并段时临时扩张），必须再做 VACUUM 才回收。

    VACUUM 需要临时占用与库等量的磁盘空间，不要在磁盘将满时执行。
    """
    from .library import Library

    _, db_path = _resolve_paths(root, db)
    if not db_path.exists():
        return {"compacted": False, "reason": f"索引库不存在：{db_path}"}
    before = db_path.stat().st_size
    library = Library(db_path, root=_resolve_paths(root, db)[0])
    try:
        # WAL 里的内容要先落盘，VACUUM 看到的才是最终大小
        library.cx.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        library.cx.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
        library.cx.execute("INSERT INTO chunks_fts_ngram(chunks_fts_ngram) VALUES('optimize')")
        library.cx.commit()
        library.cx.execute("VACUUM")
        library.cx.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        chunks = int(library.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        integrity = library.fts_integrity_ok()
    finally:
        library.close()
    after = db_path.stat().st_size
    return {
        "compacted": True,
        "db": str(db_path),
        "before_bytes": int(before),
        "after_bytes": int(after),
        "saved_bytes": max(0, int(before - after)),
        "chunks": chunks,
        "fts_integrity": integrity,
    }
