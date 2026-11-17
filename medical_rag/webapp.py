"""本地医学教材工作台网站（标准库 HTTP 服务，仅依赖 PyMuPDF）。

启动：
    python -m medical_rag.webapp                     # 默认 http://127.0.0.1:17173
    python -m medical_rag.webapp --port 18080        # 指定端口
    python -m medical_rag.webapp --no-browser        # 不自动打开浏览器

页面功能：
    - 显示教材列表与处理状态（PDF / OCR / 版面 / 表格 / 结构化 / 索引）；
    - 封面取 PDF 第一页，逐页预览支持输入页码跳转，并可查看该页识别文本；
    - 快捷导入教材（本地上传或直接填写路径）；
    - 一键处理教材，实时显示各阶段进度与日志。

API 概览：
    GET  /api/overview                 书库、任务、环境信息
    GET  /api/health                   环境自检（同 medical-rag doctor，异常时 HTTP 503）
    GET  /api/books                    教材列表
    GET  /api/books/<id>               教材详情（含章节、页范围）
    GET  /api/books/<id>/cover.png     封面（PDF 第一页）
    GET  /api/books/<id>/page/<n>.png  页面预览图（磁盘缓存）
    GET  /api/books/<id>/text/<n>      该页识别文本
    GET  /api/books/<id>/pdf           原版 PDF（支持 Range，可用 #page=N 跳页）
    POST /api/import                   按本地路径导入
    POST /api/import/begin             上传第一步：登记文件名，返回 task_id
    PUT  /api/import/upload/<task_id>  上传第二步：请求体即文件字节流
    POST /api/inspect                  检测 PDF 是否带文字层（导入前预览）
    POST /api/process                  处理教材（异步任务）
    POST /api/reindex                  只重建索引
    POST /api/search                   检索教材证据
    GET  /api/tasks                    任务列表
    GET  /api/tasks/<id>?since=N       任务快照 + 增量日志
    POST /api/tasks/<id>/cancel        取消任务
"""

from __future__ import annotations

import argparse
import functools
import hmac
import ipaddress
import json
import mimetypes
import os
import re
import secrets
import socket
import sqlite3
import sys
import threading
import time
import urllib.parse
import weakref
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .doctor import check_environment, interpreter_info
from .library import Library
from .pdftext import cached_analyze, kind_label, recommendation
from .pipeline import PACKAGE_ROOT, run_book_pipeline, validate_pdf
from .tasks import Task, TaskManager
from .workspace import WINDOWS_RESERVED_NAMES, safe_dir_name, scan_books

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 17173  # 冷门端口，避开 8000/8080/5000 等常用端口
WEBUI_DIR = Path(__file__).resolve().parent / "webui"
DEFAULT_ROOT = PACKAGE_ROOT
AUTH_COOKIE = "medrag_token"
AUTH_HINT = "缺少或错误的访问令牌；请使用启动时打印的带 ?token= 的地址"
MAX_JSON_BODY = 8 * 1024 * 1024
# 单次上传的字节上限（可用 MEDICAL_RAG_MAX_UPLOAD 覆盖，单位字节）
MAX_UPLOAD_SIZE = int(os.environ.get("MEDICAL_RAG_MAX_UPLOAD") or 2 * 1024 * 1024 * 1024)
COPY_CHUNK = 1024 * 1024
PDF_CHUNK = 1024 * 1024
# 预览图宽度档位：缓存键只能用这几档，否则每个窗口宽度都会生成一张新 PNG。
# 前端（webui/app.js）必须使用同一组数字，改这里要同步改前端。
RENDER_WIDTH_BUCKETS = (900, 1200, 1600, 2000, 2600)
# 预览图缓存上限（默认 256 MB，可用 MEDICAL_RAG_WEB_CACHE_MAX_MB 覆盖）
MAX_CACHE_BYTES = int(os.environ.get("MEDICAL_RAG_WEB_CACHE_MAX_MB") or 256) * 1024 * 1024
# 每渲染这么多次预览图检查一次缓存总量
CACHE_PRUNE_EVERY = 50
# 本机回环主机名：Host/Origin 校验永远接受它们
LOOPBACK_HOST_NAMES = frozenset({"127.0.0.1", "::1", "localhost"})
# JSON 请求体允许的 Content-Type（含 +json 后缀）；表单/纯文本属"简单请求"，必须拒绝
_JSON_CONTENT_TYPES = frozenset({"application/json", "text/json", "application/x-json"})
_WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::", "[::]", "*"})


@functools.lru_cache(maxsize=1)
def _local_host_names() -> frozenset[str]:
    """本机可用主机名（Host 头白名单用），失败时返回空集合。

    ``--allow-remote`` 绑定具体网卡时，浏览器多半用 ``http://<机器名>:<端口>``
    访问；只认绑定 IP 会把正常访问 403 掉。这里收集机器名/FQDN/解析出的别名，
    通配绑定不走这条路（枚举不出来，靠令牌鉴权）。
    """
    names: set[str] = set(LOOPBACK_HOST_NAMES)
    try:
        names.add(socket.gethostname().strip().lower().rstrip("."))
        names.add(socket.getfqdn().strip().lower().rstrip("."))
        _, aliases, addresses = socket.gethostbyname_ex(socket.gethostname())
        names.update(alias.strip().lower().rstrip(".") for alias in aliases)
        names.update(str(address) for address in addresses)
    except (OSError, UnicodeError):
        pass
    return frozenset(name for name in names if name)


class UploadTooLarge(ValueError):
    """上传体积超过 MAX_UPLOAD_SIZE（与控制流里的普通 ValueError 区分状态码）。"""


class PayloadTooLarge(ValueError):
    """JSON 请求体超过 MAX_JSON_BODY（应回 413，而不是 500）。"""


class UnsupportedMediaType(ValueError):
    """变更类接口的 Content-Type 不是 application/json（应回 415，挡住 CSRF 简单请求）。"""


MAX_FILE_NAME = 120


def safe_file_name(name: str, fallback: str = "教材.pdf") -> str:
    """把上传文件名转成安全的单文件名，并保证 .pdf 后缀。

    三个必须避开的坑：
      1. 去掉路径与非法字符，避免写到目标目录之外；
      2. 截断时先保留扩展名、只截主体——否则超长文件名会被截成没有 .pdf 的
         字符串，后续按后缀判断的逻辑全部失效；
      3. Windows 保留设备名即使带扩展名（CON.pdf / NUL.pdf / COM1.pdf）也
         无法创建，需要加下划线。
    """
    raw = Path(str(name)).name
    raw = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw)
    raw = re.sub(r"\s+", " ", raw).strip().strip(".")
    if not raw:
        return fallback

    body = raw[:-4] if raw.lower().endswith(".pdf") else raw
    body = body.strip().strip(".")
    limit = max(1, MAX_FILE_NAME - len(".pdf"))
    body = body[:limit].strip().rstrip(". ") or Path(fallback).stem or "教材"
    if body.upper() in WINDOWS_RESERVED_NAMES:
        body = f"{body}_"
    return f"{body}.pdf"


def is_loopback(host: str) -> bool:
    """host 是否只监听本机（127.0.0.1 / ::1 / localhost）。"""
    if str(host).strip().lower() in ("localhost", "localhost."):
        return True
    try:
        return ipaddress.ip_address(str(host).strip()).is_loopback
    except ValueError:
        return False


def book_lock_key(directory: Path | str) -> str:
    """教材目录的规范化锁键（Windows 下忽略大小写，路径分隔符统一）。

    同一本教材的处理 / 重建索引 / 导入共用一个键，避免多个任务同时写
    text_v3、processed_v3、SQLite 索引与缓存 PNG。
    """
    return os.path.normcase(str(Path(directory).resolve()))


def bucket_width(width: int | str) -> int:
    """把请求宽度量化到 ``RENDER_WIDTH_BUCKETS`` 的档位（取 ≥ 请求值的最小档）。

    缓存键只能是有限几档：否则窗口宽 × zoomFactor × dpr 算出的任意整数都会
    生成一张独立 PNG（实测 175 张 / 71.5 MB / 26 种宽度）。按"向上取档"量化，
    请求值永远拿到不小于它的图，不会变糊。
    """
    try:
        value = max(1, int(width))
    except (TypeError, ValueError):
        value = RENDER_WIDTH_BUCKETS[0]
    for bucket in RENDER_WIDTH_BUCKETS:
        if value <= bucket:
            return bucket
    return RENDER_WIDTH_BUCKETS[-1]


def _prune_cache(app: "WebApp", limit_bytes: int = MAX_CACHE_BYTES) -> int:
    """按 mtime 从旧到新删预览图，直到缓存总量回到 ``limit_bytes`` 以内。

    返回删除的文件数。只删 ``*.png``，不动 ``.medical_rag`` 下的其它产物；
    删除失败（被占用）的项直接跳过，不做重试。
    """
    if limit_bytes <= 0:
        return 0
    entries: list[tuple[float, int, Path]] = []
    total = 0
    try:
        for path in app.cache_dir.rglob("*.png"):
            try:
                info = path.stat()
            except OSError:
                continue
            entries.append((info.st_mtime, info.st_size, path))
            total += info.st_size
    except OSError:
        return 0
    if total <= limit_bytes:
        return 0
    removed = 0
    for _mtime, size, path in sorted(entries):
        if total <= limit_bytes:
            break
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        removed += 1
    return removed


def render_page_png(app: "WebApp", book: dict[str, Any], page_no: int, width: int) -> Path:
    """把 PDF 某页渲染成 PNG（磁盘缓存，按 PDF mtime 失效，宽度按档位量化）。"""
    pdf = Path(book["pdf"])
    if not pdf.exists():
        raise FileNotFoundError(f"PDF 不存在：{pdf}")
    page_no = max(1, int(page_no))
    width = bucket_width(width)
    out_dir = app.cache_dir / str(book["id"])
    out = out_dir / f"page-{page_no:04d}-w{width}.png"
    mtime = pdf.stat().st_mtime
    if out.exists() and out.stat().st_mtime >= mtime:
        return out
    with app.render_lock:
        if out.exists() and out.stat().st_mtime >= mtime:
            return out
        import fitz

        with fitz.open(pdf) as doc:
            if page_no > len(doc):
                raise IndexError(f"页码超出范围：{page_no} > {len(doc)}")
            page = doc[page_no - 1]
            scale = width / page.rect.width
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
        out_dir.mkdir(parents=True, exist_ok=True)
        temp = out.with_name(out.stem + ".tmp.png")
        pix.save(str(temp))
        os.replace(temp, out)
    app.note_render()
    return out


class WebApp:
    """站点状态：教材扫描缓存、任务管理器、渲染锁。"""

    def __init__(
        self,
        root: Path | str = DEFAULT_ROOT,
        db: Path | str | None = None,
        token: str = "",
        allow_local_paths: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        self.db = Path(db).expanduser().resolve() if db else self.root / ".medical_rag" / "library.sqlite3"
        self.cache_dir = self.root / ".medical_rag" / "web_cache"
        # 非空时启用访问令牌（非本机监听必须启用；本机监听默认为空 = 不鉴权）
        self.token = str(token or "")
        # hmac.compare_digest 的 str 版本只接受 ASCII，中文口令会直接抛 TypeError
        self.token_bytes = self.token.encode("utf-8")
        self.allow_local_paths = bool(allow_local_paths)
        self.asset_cache = "private, max-age=86400" if self.token else "public, max-age=86400"
        self.tasks = TaskManager()
        self.render_lock = threading.Lock()
        self._books_lock = threading.Lock()
        self._books_cache: tuple[float, list[dict[str, Any]]] = (0.0, [])
        # 常驻连接：ThreadingHTTPServer 每个连接一个线程，所以按线程各持一条
        # Library（sqlite 连接不能跨线程共享），请求结束不关闭。
        self._local = threading.local()
        self._lib_lock = threading.Lock()
        self._libraries: "weakref.WeakSet[Library]" = weakref.WeakSet()
        # sqlite 连接只能由创建它的线程关闭。close_libraries() 在收尾线程里遇到
        # 别的线程的连接时不能真的 close()，只能登记“请关闭”（见 Library.request_close），
        # 记账在这里，避免用 except Exception: pass 假装收尾成功。
        self._pending_close: "weakref.WeakSet[Library]" = weakref.WeakSet()
        self._close_errors: list[str] = []
        # 预览图缓存清理：渲染计数 + 独立锁（不占用 render_lock）
        self._cache_lock = threading.Lock()
        self._render_count = 0

    # ------------------------------------------------------------------ 书库
    def books(self, max_age: float = 1.5) -> list[dict[str, Any]]:
        with self._books_lock:
            now = time.time()
            if now - self._books_cache[0] < max_age:
                return self._books_cache[1]
            books = scan_books(self.root, self.db)
            self._books_cache = (now, books)
            return books

    def invalidate_books(self) -> None:
        with self._books_lock:
            self._books_cache = (0.0, [])

    def book(self, book_id: str) -> dict[str, Any] | None:
        return next((book for book in self.books() if book["id"] == book_id), None)

    def library(self) -> Library:
        """返回**本线程**的常驻 Library（连接复用，请求结束不关闭）。

        开一条 ``Library`` 要跑一遍全量 DDL + 两轮 backfill（实测 186 ms），
        而真正要跑的查询只要 0.1 ms 级；每次请求新建连接是纯浪费，而且
        ``_init_schema`` 里的 ``PRAGMA user_version`` 写会让读接口也去抢写锁
        （重建索引期间整站读接口会卡到 busy_timeout 30 s 后 500）。
        """
        library = getattr(self._local, "library", None)
        if library is not None:
            return library
        with self._lib_lock:
            library = getattr(self._local, "library", None)
            if library is None:
                library = Library(self.db, root=self.root)
                self._local.library = library
                self._libraries.add(library)
        return library

    def warm_library(self) -> None:
        """启动时预热本线程的常驻连接，把首查的开关库开销挪到启动阶段。

        索引库还不存在时直接跳过：``Library()`` 会建库，而"只读接口不建库"
        是 ``tests/test_webapp.py::HealthTests`` 明确守护的契约。
        """
        try:
            if not Path(self.db).exists():
                return
        except OSError:
            return
        try:
            self.library()
        except Exception:  # noqa: BLE001 - 预热失败不影响服务，首次请求会重试
            pass

    def invalidate_library(self) -> None:
        """写入口完成后失效本线程的常驻连接（其它线程靠 WAL 自动看到新提交）。"""
        library = getattr(self._local, "library", None)
        if library is None:
            return
        self._local.library = None
        self._close_library(library)

    def release_thread_library(self) -> None:
        """连接关闭时释放本线程的常驻连接。

        ``ThreadingHTTPServer`` 每个连接一个线程：keep-alive 连接在处理完所有
        请求后才走到这里，所以连接复用不受影响；而 ``Connection: close`` 的
        客户端（curl / 脚本 / urllib）每次请求都换线程，必须在这里回收，否则
        每条连接都会留下一条再没人用的 sqlite 连接。
        """
        library = getattr(self._local, "library", None)
        if library is None:
            return
        self._local.library = None
        self._close_library(library)

    def close_libraries(self) -> None:
        """收尾：关闭所有线程的常驻连接（进程退出 / server_close 时调用）。

        诚实的合同：sqlite 连接**只能由创建它的线程关闭**（``check_same_thread``），
        所以这里只能真正关闭“调用线程自己的”连接；别的线程的连接一律
        ``Library.request_close()`` 登记，由 owner 线程在下一次使用
        （``Library._ensure_open``）或 ``release_thread_library()`` 时关闭。

        以前这里用 ``except Exception: pass`` 吞掉跨线程的 ``ProgrammingError``，
        结果 ``_libraries`` 账清零了、连接却还活着：收尾看起来成功，其实没有。
        """
        current = threading.get_ident()
        with self._lib_lock:
            # 之前登记过、但 owner 线程已经关掉的连接随手清掉，账本才能收敛
            libraries = [item for item in self._libraries if not item.closed]
            libraries += [item for item in self._pending_close if not item.closed]
            self._libraries.clear()
            self._pending_close.clear()
        local = getattr(self._local, "library", None)
        self._local.library = None
        if local is not None and not local.closed and local not in libraries:
            libraries.append(local)
        for item in libraries:
            if item.owner_thread == current:
                self._close_library(item)
                continue
            # 其它线程创建的连接：不能在这里 close()，登记后由其 owner 线程收尾
            item.request_close()
            with self._lib_lock:
                self._pending_close.add(item)

    def _close_library(self, library: Library) -> None:
        """在 owner 线程里关闭一条常驻连接。

        跨线程调用会抛 ``sqlite3.ProgrammingError``：不静默吞掉，记进
        ``_close_errors`` 并转成“登记关闭”，由 owner 线程补关。
        """
        with self._lib_lock:
            self._libraries.discard(library)
            self._pending_close.discard(library)
        try:
            library.close()
        except sqlite3.Error as exc:
            with self._lib_lock:
                self._close_errors.append(f"{type(exc).__name__}: {exc}")
                self._pending_close.add(library)
            library.request_close()

    def note_render(self) -> None:
        """登记一次预览图渲染；每 ``CACHE_PRUNE_EVERY`` 次检查一次缓存总量。"""
        self._render_count += 1
        if self._render_count % CACHE_PRUNE_EVERY:
            return
        with self._cache_lock:
            try:
                _prune_cache(self)
            except Exception:  # noqa: BLE001 - 清缓存失败不能影响出图
                pass

    def overview(self, port: int | None = None) -> dict[str, Any]:
        from .embeddings import create_backend_cached

        backend, embedding_warning = create_backend_cached(os.environ.get("MEDICAL_RAG_EMBEDDING"))
        books = self.books()
        indexed = [book for book in books if book["status"]["index"]["complete"]]
        return {
            "root": str(self.root),
            "db": str(self.db),
            "port": port,
            "time": time.time(),
            "interpreters": interpreter_info(self.root),
            "embedding": {"backend": backend.name, "warning": embedding_warning},
            "books": books,
            "tasks": self.tasks.list(),
            "totals": {
                "books": len(books),
                "indexed": len(indexed),
                "chunks": sum(book["status"]["index"]["chunks"] for book in books),
            },
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "MedicalRAGWeb/0.1"
    protocol_version = "HTTP/1.1"
    _auth_cookie = ""

    @property
    def app(self) -> WebApp:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def finish(self) -> None:
        """连接收尾：释放本线程的常驻 Library（keep-alive 连接处理完所有请求才到这里）。"""
        try:
            self.app.release_thread_library()
        except Exception:  # noqa: BLE001 - 收尾阶段不再抛
            pass
        super().finish()

    # --------------------------------------------------------------- 鉴权
    def _cookie_token(self) -> str:
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, value = part.strip().partition("=")
            if name == AUTH_COOKIE:
                return urllib.parse.unquote(value)
        return ""

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        """令牌可来自 X-Auth-Token 头、?token= 查询参数或 Cookie。

        查询参数/ Cookie 是必需的：封面、预览图和 PDF 都用 <img src> / 链接加载，
        无法附加自定义请求头。
        """
        if not self.app.token_bytes:
            return True
        candidates = (
            self.headers.get("X-Auth-Token") or "",
            (query.get("token") or [""])[0],
            self._cookie_token(),
        )
        # 统一编码成 bytes 再比较：hmac.compare_digest 的 str 版本只接受 ASCII，
        # 中文口令会抛 TypeError 让每个请求都 500。
        return any(
            value and hmac.compare_digest(str(value).encode("utf-8"), self.app.token_bytes)
            for value in candidates
        )

    # ------------------------------------------------- 跨站 / 重绑定防护
    def _host_allowed(self) -> bool:
        """Host 头必须是本机名或本站实际绑定的地址（防 DNS rebinding）。

        DNS rebinding 下浏览器会带 ``Host: evil.example.com`` 来访问 127.0.0.1，
        只认回环名与 ``--host`` 传入的绑定地址即可挡住。
        """
        raw = (self.headers.get("Host") or "").strip()
        if not raw:
            return True  # HTTP/1.0 客户端可能不带 Host
        try:
            hostname = (urllib.parse.urlsplit("//" + raw).hostname or "").strip().lower().rstrip(".")
        except ValueError:
            return False
        if not hostname:
            return False
        if hostname in LOOPBACK_HOST_NAMES:
            return True
        allowed = getattr(self.server, "allowed_hosts", None)
        if allowed is None:  # 通配绑定（0.0.0.0 / ::）：枚举不出来，只靠令牌鉴权
            return True
        return hostname in allowed

    def _origin_allowed(self) -> bool:
        """变更类请求的跨站防护：有 Origin / Sec-Fetch-Site 时必须同源。

        缺失时放行——curl、脚本、现有测试都不带这些头；浏览器跨站请求一定带
        Origin（或 Sec-Fetch-Site: cross-site），所以宽松实现同样能挡住 CSRF。
        """
        if (self.headers.get("Sec-Fetch-Site") or "").strip().lower() == "cross-site":
            return False
        origin = (self.headers.get("Origin") or "").strip()
        if not origin:
            return True
        try:
            parsed = urllib.parse.urlsplit(origin)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https"):
            return False  # 含 Origin: null（file:// / 沙箱 iframe）
        hostname = (parsed.hostname or "").strip().lower().rstrip(".")
        if not hostname:
            return False
        # 同源比较必须带上端口：只比主机名会让本机其它端口（另一个 dev server、
        # Jupyter、本项目的第二个实例）发起的请求也被判成同源，CSRF 防护形同虚设。
        try:
            request = urllib.parse.urlsplit("//" + (self.headers.get("Host") or "").strip())
        except ValueError:
            return False
        request_host = (request.hostname or "").strip().lower().rstrip(".")
        if not request_host:
            return False
        expected_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        actual_port = request.port or (443 if parsed.scheme == "https" else 80)
        if expected_port != actual_port:
            return False
        if hostname == request_host:
            return True
        # Host 可能写机器名、Origin 写回环 IP（或反之）：回环名之间互认，但端口
        # 必须一致（上面已验证）。
        if hostname in LOOPBACK_HOST_NAMES and request_host in LOOPBACK_HOST_NAMES:
            return True
        allowed = getattr(self.server, "allowed_hosts", None)
        if allowed is None:
            return False
        return hostname in allowed

    def _guard_request(self, mutating: bool = False) -> bool:
        """入口统一防护；返回 False 表示已经回过 403。"""
        if not self._host_allowed():
            self._reject_cross_site(403, "Host 不在允许列表内（本站只接受本机或绑定的地址）")
            return False
        if mutating and not self._origin_allowed():
            self._reject_cross_site(403, "拒绝跨站请求：Origin 与本站不同源")
            return False
        return True

    def _reject_cross_site(self, status: int, message: str) -> None:
        """回 403 前先把请求体读掉，避免 Windows 上直接关连接变成 RST。"""
        try:
            if self.command in ("POST", "PUT", "PATCH"):
                self._drain_body(COPY_CHUNK)
        except Exception:  # noqa: BLE001 - 读体失败不影响回 403
            self.close_connection = True
        self._send_error_json(status, message)

    def _remember_token(self, query: dict[str, list[str]]) -> None:
        """带 ?token= 访问时种下 Cookie，后续静态资源/图片/PDF 自动携带。"""
        if self.app.token and (query.get("token") or [""])[0]:
            value = urllib.parse.quote(self.app.token)
            # SameSite=Strict 让跨站请求不携带该 Cookie，作为变更类接口的 CSRF 防护
            self._auth_cookie = f"{AUTH_COOKIE}={value}; Path=/; HttpOnly; SameSite=Strict"

    # ------------------------------------------------------------- 响应工具
    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store", extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        if self._auth_cookie:
            self.send_header("Set-Cookie", self._auth_cookie)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _send_data(self, data: Any, status: int = 200) -> None:
        body = json.dumps({"ok": status < 400, "data": data}, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: int, message: str) -> None:
        body = json.dumps({"ok": False, "error": message}, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_conflict(self, task: Task, message: str) -> None:
        """409：这本教材已有任务在运行，同时把已有 task_id 返回给调用方。"""
        body = json.dumps(
            {
                "ok": False,
                "error": message,
                "data": {"task_id": task.id, "label": task.label, "status": task.status},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        # 请求体可能还没读（超大上传），关闭连接避免 HTTP/1.1 流水线错位
        self.close_connection = True
        self._send(409, body, "application/json; charset=utf-8")

    def _send_file(self, path: Path, content_type: str, cache: str) -> None:
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_error_json(500, f"读取文件失败：{exc}")
            return
        self._send(200, body, content_type, cache=cache)

    def _json_content_type(self) -> bool:
        """变更类接口的 Content-Type 是否可接受（容忍 ``; charset=`` 与 ``+json``）。"""
        raw = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not raw:
            return True  # curl / 脚本常常不带 Content-Type
        return raw in _JSON_CONTENT_TYPES or raw.endswith("+json")

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_JSON_BODY:
            raise PayloadTooLarge("请求体过大")
        if not self._json_content_type():
            raise UnsupportedMediaType(
                "变更类接口只接受 application/json；text/plain 等简单请求属于 CSRF 载体，已拒绝"
            )
        raw = self.rfile.read(length)
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    @staticmethod
    def _trim(data: Any) -> Any:
        """去掉内部字段，保持 API 输出干净。"""
        if isinstance(data, dict):
            return {key: Handler._trim(value) for key, value in data.items() if not key.startswith("_")}
        if isinstance(data, list):
            return [Handler._trim(item) for item in data]
        return data

    # ------------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        try:
            if not self._guard_request():
                return
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            query = urllib.parse.parse_qs(parsed.query)

            # 静态资源不包含数据，先放行，避免首次访问时 Cookie 还未种下
            if path.startswith("/static/"):
                return self._serve_static(path[len("/static/"):])
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")

            if not self._authorized(query):
                return self._send_error_json(401, AUTH_HINT)
            self._remember_token(query)

            if path in ("/", "/index.html"):
                return self._serve_static("index.html")

            if path == "/api/overview":
                port = getattr(self.server, "server_port", None)
                return self._send_data(self._trim(self.app.overview(port)))
            if path == "/api/health":
                report = check_environment(self.app.root, self.app.db)
                return self._send_data(self._trim(report), status=200 if report["ok"] else 503)
            if path == "/api/books":
                return self._send_data(self._trim(self.app.books()))
            if path == "/api/tasks":
                return self._send_data(self.app.tasks.list())

            parts = [segment for segment in path.split("/") if segment]
            if parts[:2] == ["api", "books"] and len(parts) >= 3:
                return self._book_get(parts[2], parts[3:], query)
            if parts[:2] == ["api", "tasks"] and len(parts) == 3:
                return self._task_get(parts[2], query)
            return self._send_error_json(404, "没有这个接口")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _serve_static(self, name: str) -> None:
        base = WEBUI_DIR.resolve()
        target = (base / name).resolve()
        if not target.is_file() or base not in target.parents:
            return self._send_error_json(404, "静态资源不存在")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if target.suffix in (".html", ".css", ".js"):
            content_type = {"html": "text/html", "css": "text/css", "js": "text/javascript"}[target.suffix[1:]] + "; charset=utf-8"
        self._send_file(target, content_type, cache="no-cache")

    def _book_get(self, book_id: str, rest: list[str], query: dict[str, list[str]]) -> None:
        book = self.app.book(book_id)
        if book is None:
            return self._send_error_json(404, "没有这本教材")
        if not rest:
            data = dict(book)
            index = book["status"]["index"]
            if index.get("book_id"):
                # 常驻连接：请求结束不关闭（见 WebApp.library）
                library = self.app.library()
                library_row = library.get_book(index["book_id"])
                data["index_book"] = library_row
                page_range = library.page_range(index["book_id"])
                data["page_range"] = {"min": page_range[0], "max": page_range[1]} if page_range else None
            return self._send_data(self._trim(data))

        if not book.get("pdf"):
            return self._send_error_json(404, "这本教材没有 PDF，无法预览")
        try:
            if rest == ["cover.png"] or rest == ["cover"]:
                width = _int_arg(query, "w", 480)
                path = render_page_png(self.app, book, 1, width)
                return self._send_file(path, "image/png", cache=self.app.asset_cache)
            if rest == ["pdf"]:
                return self._serve_pdf(Path(book["pdf"]))
            if len(rest) == 2 and rest[0] == "page":
                page_no = int(rest[1].split(".")[0])
                width = _int_arg(query, "w", 1500)
                path = render_page_png(self.app, book, page_no, width)
                return self._send_file(path, "image/png", cache=self.app.asset_cache)
            if len(rest) == 2 and rest[0] == "text":
                page_no = int(rest[1].split(".")[0])
                return self._send_data(self._page_text(book, page_no))
        except (ValueError, IndexError) as exc:
            return self._send_error_json(400, str(exc))
        return self._send_error_json(404, "没有这个接口")

    def _page_text(self, book: dict[str, Any], page_no: int) -> dict[str, Any]:
        index = book["status"]["index"]
        if not index.get("book_id"):
            return {"page": page_no, "indexed": False, "chunks": []}
        chunks = self.app.library().chunks_for_page(index["book_id"], page_no)
        return {
            "page": page_no,
            "indexed": True,
            "chunks": [
                {"chunk_id": item["chunk_id"], "section": item["section"], "text": item["text"]}
                for item in chunks
            ],
        }

    def _serve_pdf(self, pdf: Path) -> None:
        if not pdf.exists():
            return self._send_error_json(404, "PDF 文件不存在")
        size = pdf.stat().st_size
        start, end = 0, size - 1
        status = 200
        range_header = self.headers.get("Range")
        if range_header and size:
            match = re.match(r"bytes=(\d*)-(\d*)", range_header.strip())
            if match:
                first, last = match.group(1), match.group(2)
                if first and last:            # bytes=a-b
                    start, end = int(first), min(int(last), size - 1)
                elif first:                   # bytes=a-
                    start, end = int(first), size - 1
                elif last:                    # bytes=-N：末尾 N 字节（PDF 阅读器读尾部交叉引用表要用）
                    start, end = max(0, size - int(last)), size - 1
                else:                         # bytes=-：等同整个文件
                    start, end = 0, size - 1
                if start >= size or start > end:
                    return self._send(416, b"", "application/pdf", extra={"Content-Range": f"bytes */{size}"})
                status = 206
        self.send_response(status)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        filename = urllib.parse.quote(pdf.name)
        self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{filename}")
        self.send_header("Cache-Control", "private, no-store" if self.app.token else "no-store")
        self.end_headers()
        if self.command == "HEAD":
            return
        remaining = end - start + 1
        try:
            with pdf.open("rb") as stream:
                stream.seek(start)
                while remaining > 0:
                    chunk = stream.read(min(PDF_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _task_get(self, task_id: str, query: dict[str, list[str]]) -> None:
        since = _int_arg(query, "since", 0)
        snapshot = self.app.tasks.snapshot(task_id, since=since)
        if snapshot is None:
            return self._send_error_json(404, "没有这个任务")
        self._send_data(snapshot)

    # ------------------------------------------------------------------ POST
    def do_POST(self) -> None:  # noqa: N802
        try:
            if not self._guard_request(mutating=True):
                return
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            parts = [segment for segment in path.split("/") if segment]
            if not self._authorized(urllib.parse.parse_qs(parsed.query)):
                # 完整读掉 JSON 体再回复，避免客户端看到连接被重置（WinError 10053）
                self._drain_body(MAX_JSON_BODY)
                return self._send_error_json(401, AUTH_HINT)
            if path == "/api/import":
                return self._post_import()
            if path == "/api/import/begin":
                return self._post_import_begin()
            if path == "/api/inspect":
                return self._post_inspect()
            if path == "/api/process":
                return self._post_process()
            if path == "/api/reindex":
                return self._post_reindex()
            if path == "/api/search":
                return self._post_search()
            if parts[:2] == ["api", "tasks"] and len(parts) == 4 and parts[3] == "cancel":
                return self._post_task_cancel(parts[2])
            self._drain_body()
            return self._send_error_json(404, "没有这个接口")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except PayloadTooLarge as exc:
            # 与 401/415 分支一致：先把已经到达的请求体读掉再回 413。带未读数据
            # 直接关连接，Windows 会发 RST，客户端拿到 WinError 10053 而不是 413
            # （_drain_body 的 docstring 已说明；实测 200 次里 18 次丢响应）。
            self._drain_body()
            self.close_connection = True
            self._send_error_json(413, str(exc))
        except UnsupportedMediaType as exc:
            self._drain_body(COPY_CHUNK)
            self._send_error_json(415, str(exc))
        except (ValueError, TypeError) as exc:
            # 客户端错误（坏 JSON / 非法数值 / 非对象体）不该回 500
            self._send_error_json(400, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _drain_body(self, limit: int = COPY_CHUNK) -> bool:
        """尽力读掉请求体，返回是否已读完。

        拒绝请求（如 401）时如果不读走请求体就直接关连接，Windows 会因为接收
        缓冲区里还有未读数据而发 RST，客户端拿到的是连接被中止而不是我们发的
        状态码。所以体量小就完整读掉，体量超过 ``limit`` 才标记关闭连接。
        """
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return True
        if length > limit:
            self.close_connection = True
            return False
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(COPY_CHUNK, remaining))
            if not chunk:
                self.close_connection = True
                return False
            remaining -= len(chunk)
        return True

    def _post_import(self) -> None:
        body = self._read_json()
        if not self.app.allow_local_paths:
            return self._send_error_json(403, "远程模式只允许上传 PDF，不允许读取服务端本地路径")
        source = str(body.get("path") or "").strip()
        if not source:
            return self._send_error_json(400, "请填写 PDF 路径")
        title = str(body.get("title") or "").strip() or None
        path = Path(source).expanduser()
        if not path.exists():
            return self._send_error_json(400, f"文件不存在：{path}")
        if path.is_dir():
            return self._send_error_json(400, "请填写 PDF 文件路径，而不是目录")
        if path.suffix.lower() != ".pdf":
            return self._send_error_json(400, "目前只支持 PDF 文件")
        resolved = path.resolve()
        if self._pdf_conflict(title or resolved.stem, resolved.name):
            return self._send_error_json(400, "目标目录已存在其他 PDF，请换一个书名")

        from .pipeline import import_pdf  # 延迟导入，避免循环引用

        app = self.app
        book_name = title or resolved.stem
        task, created = app.tasks.create_unique(
            "import",
            f"导入《{book_name}》",
            lambda current: import_pdf(current, app.root, resolved, title=title),
            meta={"kind": "path", "path": str(resolved)},
            lock_key=book_lock_key(app.root / "res" / safe_dir_name(book_name)),
        )
        if not created:
            return self._send_conflict(task, f"《{book_name}》已有任务在运行，请等它结束")
        self.app.invalidate_library()
        self._send_data({"task_id": task.id})

    def _pdf_conflict(self, title: str, filename: str) -> bool:
        """目标目录里已有其它 PDF 时不允许导入，避免一本教材两个主文件。"""
        book_dir = self.app.root / "res" / safe_dir_name(title)
        pdf_dir = book_dir / "PDF"
        if not pdf_dir.is_dir():
            return False
        return any(path.name != filename for path in pdf_dir.glob("*.pdf"))

    def _post_import_begin(self) -> None:
        body = self._read_json()
        filename = safe_file_name(str(body.get("filename") or ""))
        title = str(body.get("title") or "").strip() or Path(filename).stem
        if not filename:
            return self._send_error_json(400, "缺少文件名")
        # 先判运行中的任务：这是比“目录已有其它 PDF”更根本的拒绝理由
        lock_key = book_lock_key(self.app.root / "res" / safe_dir_name(title))
        running = self.app.tasks.active_for(lock_key)
        if running is not None:
            return self._send_conflict(running, f"《{title}》已有任务在运行，请等它结束")
        if self._pdf_conflict(title, filename):
            return self._send_error_json(400, "同名目录已存在其他 PDF，请换一个书名")
        task = self.app.tasks.create_manual(
            "import",
            f"导入《{title}》",
            # 注意用 book_key 而不是 lock_key：pending 状态不算已持锁，否则用户
            # 放弃上传后残留的任务会永久阻塞这本教材
            meta={"kind": "upload", "filename": filename, "title": title, "book_key": lock_key},
        )
        self._send_data(
            {
                "task_id": task.id,
                "upload_url": f"/api/import/upload/{task.id}",
                "book_dir": str(self.app.root / "res" / safe_dir_name(title)),
            }
        )

    def _post_inspect(self) -> None:
        """导入前检测 PDF 是否带文字层（供导入弹窗使用）。"""
        body = self._read_json()
        if not self.app.allow_local_paths:
            return self._send_error_json(403, "远程模式只允许上传 PDF，不允许读取服务端本地路径")
        path_text = str(body.get("path") or "").strip()
        if not path_text:
            return self._send_error_json(400, "请填写 PDF 路径")
        pdf = Path(path_text).expanduser()
        if not pdf.exists():
            return self._send_error_json(400, f"文件不存在：{pdf}")
        if pdf.is_dir() or pdf.suffix.lower() != ".pdf":
            return self._send_error_json(400, "请填写 PDF 文件路径")
        info = cached_analyze(pdf, self.app.root / ".medical_rag" / "pdf_text", root=self.app.root)
        if info.get("error"):
            return self._send_error_json(400, str(info["error"]))
        text_kind = info.get("kind")
        self._send_data(
            {
                **info,
                "label": kind_label(text_kind),
                "recommendation": recommendation(info),
                "recommended_stages": ["text", "index"]
                if text_kind == "text"
                else ["ocr", "layout", "tables", "fix", "structure", "index"],
            }
        )

    def _post_process(self) -> None:
        body = self._read_json()
        book_id = str(body.get("book_id") or "")
        book = self.app.book(book_id)
        if book is None:
            return self._send_error_json(404, "没有这本教材")
        if not book.get("pdf"):
            return self._send_error_json(400, "这本教材没有 PDF，无法处理")
        stages = body.get("stages", "auto")
        if not isinstance(stages, (list, str)):
            return self._send_error_json(400, "stages 必须是数组或 auto")
        force = bool(body.get("force", False))
        dpi = _body_int(body, "dpi", 300, 72, 600)
        device = str(body.get("device") or "gpu")
        reset_ocr = bool(body.get("reset_ocr", False))

        app = self.app
        stages_label = "自动" if stages == "auto" or stages is None else "/".join(stages)
        task, created = app.tasks.create_unique(
            "process",
            f"处理《{book['title']}》（{stages_label}）",
            lambda current: run_book_pipeline(
                current, app.root, book, stages=stages, force=force,
                dpi=dpi, device=device, reset_ocr=reset_ocr, db=app.db,
            ),
            meta={"book_id": book_id, "kind": "process", "stages": stages},
            lock_key=book_lock_key(book["dir"]),
        )
        if not created:
            return self._send_conflict(task, f"《{book['title']}》已有任务在运行，请等它结束")
        self.app.invalidate_library()
        self._send_data({"task_id": task.id})

    def _post_reindex(self) -> None:
        body = self._read_json()
        book_id = str(body.get("book_id") or "")
        book = self.app.book(book_id)
        if book is None:
            return self._send_error_json(404, "没有这本教材")
        if not book["status"]["structure"]["complete"]:
            return self._send_error_json(400, "还没有结构化结果（processed_v3/structured），请先处理教材")
        app = self.app
        task, created = app.tasks.create_unique(
            "index",
            f"重建索引《{book['title']}》",
            lambda current: run_book_pipeline(current, app.root, book, stages=["index"], db=app.db),
            meta={"book_id": book_id, "kind": "reindex"},
            lock_key=book_lock_key(book["dir"]),
        )
        if not created:
            return self._send_conflict(task, f"《{book['title']}》已有任务在运行，请等它结束")
        self.app.invalidate_library()
        self._send_data({"task_id": task.id})

    def _post_search(self) -> None:
        body = self._read_json()
        query = str(body.get("query") or "").strip()
        if not query:
            return self._send_error_json(400, "检索词不能为空")
        limit = _body_int(body, "limit", 8, 1, 20)
        book_id = body.get("book_id")
        scope = None
        library = self.app.library()
        if book_id not in (None, "", "all"):
            index_id = self._resolve_index_book_id(book_id, library)
            if index_id is None:
                return self._send_error_json(400, "没有这本教材，或该教材尚未建立索引")
            row = library.get_book(index_id)
            scope = {"id": index_id, "title": row["title"] if row else str(book_id)}
            results = library.search(query, limit, book=index_id)
        else:
            results = library.search(query, limit)
        self._send_data({"query": query, "results": results, "scope": scope})

    def _resolve_index_book_id(self, book_id: Any, library: Library) -> int | None:
        """兼容两种范围标识：网站书卡的 workspace id（hash）与索引库的整数 id。"""
        workspace = self.app.book(str(book_id))
        if workspace is not None:
            index_id = workspace["status"]["index"].get("book_id")
            if index_id is not None:
                return int(index_id)
        if str(book_id).isdigit():
            row = library.get_book(int(book_id))
            return int(row["id"]) if row else None
        return None

    def _post_task_cancel(self, task_id: str) -> None:
        self._drain_body()
        if not self.app.tasks.cancel(task_id):
            return self._send_error_json(400, "任务不存在或已经结束")
        self._send_data({"task_id": task_id, "cancelled": True})

    # ------------------------------------------------------------------- PUT
    def do_PUT(self) -> None:  # noqa: N802
        try:
            if not self._guard_request(mutating=True):
                return
            parsed = urllib.parse.urlparse(self.path)
            path = urllib.parse.unquote(parsed.path)
            parts = [segment for segment in path.split("/") if segment]
            if not self._authorized(urllib.parse.parse_qs(parsed.query)):
                # 上传体可能极大，只读出小量；超过上限则标记关闭连接
                self._drain_body(COPY_CHUNK)
                return self._send_error_json(401, AUTH_HINT)
            if parts[:3] == ["api", "import", "upload"] and len(parts) == 4:
                return self._put_upload(parts[3])
            self._drain_body()
            return self._send_error_json(404, "没有这个接口")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (PayloadTooLarge, UploadTooLarge) as exc:
            self.close_connection = True
            self._send_error_json(413, str(exc))
        except (ValueError, TypeError) as exc:
            self._send_error_json(400, f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(500, f"{type(exc).__name__}: {exc}")

    def _put_upload(self, task_id: str) -> None:
        task = self.app.tasks.get(task_id)
        if task is None:
            return self._send_error_json(404, "没有这个上传任务")
        if task.status != "pending":
            self._drain_body()
            return self._send_error_json(409, "该上传任务已经开始或结束")
        # 先卡体积再占锁/读流：超大文件不进落盘路径，也不占用教材锁
        if self.headers.get("Transfer-Encoding", "").lower().strip() == "chunked":
            task.mark_error("不支持 chunked 传输编码")
            self.close_connection = True
            return self._send_error_json(411, "请提供 Content-Length（暂不支持 chunked 传输编码）")
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._send_error_json(411, "缺少 Content-Length 或文件为空")
        if length > MAX_UPLOAD_SIZE:
            task.mark_error(f"文件 {length} 字节超过上限 {MAX_UPLOAD_SIZE} 字节")
            # 先把已到达的字节读掉再回 413，否则带未读数据关连接会被 Windows 发 RST，
            # 客户端拿到 WinError 10053 而不是 413（同 _drain_body 的 docstring）
            self._drain_body()
            self.close_connection = True
            return self._send_error_json(
                413, f"文件超过上限（{MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB）"
            )
        # 真正开始写书库前才占锁，避免放弃上传后残留 pending 任务永久阻塞这本教材
        lock_key = str(task.meta.get("book_key") or "")
        if not self.app.tasks.acquire_lock(task, lock_key):
            return self._send_conflict(
                self.app.tasks.active_for(lock_key) or task,
                "这本教材已有任务在运行，请等它结束",
            )
        filename = str(task.meta.get("filename") or "教材.pdf")
        title = str(task.meta.get("title") or Path(filename).stem)
        book_dir = self.app.root / "res" / safe_dir_name(title)
        pdf_dir = book_dir / "PDF"
        other = [path for path in pdf_dir.glob("*.pdf") if path.name != filename] if pdf_dir.is_dir() else []
        if other:
            task.mark_error("目标目录已存在其他 PDF")
            return self._send_error_json(400, "同名目录已存在其他 PDF，请换一个书名")

        pdf_dir.mkdir(parents=True, exist_ok=True)
        target = pdf_dir / filename
        temp = pdf_dir / (filename + ".part")
        task.mark_running()
        task.log(f"接收上传：{filename}（{length / 1024 / 1024:.1f} MB）→ {target}")
        written = 0
        try:
            with temp.open("wb") as stream:
                while written < length:
                    if task.cancelled:
                        raise InterruptedError("已取消")
                    chunk = self.rfile.read(min(COPY_CHUNK, length - written))
                    if not chunk:
                        raise IOError("上传中断：客户端提前断开")
                    written += len(chunk)
                    # 累计上限（防御 chunked/长度头不一致的情况）
                    if written > MAX_UPLOAD_SIZE:
                        raise UploadTooLarge(
                            f"上传超过上限（{MAX_UPLOAD_SIZE / 1024 / 1024:.0f} MB）"
                        )
                    stream.write(chunk)
                    task.set_progress(written, length, f"{written / 1024 / 1024:.1f} / {length / 1024 / 1024:.1f} MB")
            # 字节收齐后先校验，再原子落盘：无效/截断的文件不能进书库
            upload_info = validate_pdf(temp, label=f"上传的 {filename}")
            if upload_info.get("repaired"):
                task.log(
                    "警告：上传的 PDF 已损坏，是经 PyMuPDF 修复后打开的；"
                    "文字/页码可能不完整，建议核对后再处理"
                )
            os.replace(temp, target)
        except InterruptedError:
            temp.unlink(missing_ok=True)
            task.mark_cancelled()
            return self._send_error_json(400, "上传已取消")
        except UploadTooLarge as exc:
            temp.unlink(missing_ok=True)
            task.mark_error(str(exc))
            self.close_connection = True
            return self._send_error_json(413, str(exc))
        except ValueError as exc:
            temp.unlink(missing_ok=True)
            task.mark_error(str(exc))
            return self._send_error_json(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            temp.unlink(missing_ok=True)
            task.mark_error(str(exc))
            raise
        try:
            info = cached_analyze(target, self.app.root / ".medical_rag" / "pdf_text", root=self.app.root)
            task.log(
                f"PDF 检测：{kind_label(info.get('kind'))}，"
                f"{info.get('text_pages', 0)}/{info.get('pages', 0)} 页有文字，共 {info.get('chars', 0)} 字"
            )
            task.meta["pdf_kind"] = info.get("kind")
            task.meta["pdf_text_pages"] = info.get("text_pages")
            if info.get("kind") == "text":
                task.log("建议：直接运行「文字层解析 + 建立索引」，无需 OCR")
            else:
                task.log("建议：使用 OCR 流水线（OCR → 版面 → 表格 → 纪错 → 结构化 → 索引）")
        except Exception as exc:  # noqa: BLE001 - 已确认是有效 PDF，检测失败不影响上传结果
            task.log(f"警告：PDF 文字层检测失败（文件本身有效）：{exc}")
        task.mark_done(f"上传完成：{target}")
        self.app.invalidate_books()
        self.app.invalidate_library()
        self._send_data({"task_id": task.id, "path": str(target), "book_dir": str(book_dir)})

    # ------------------------------------------------------------------ HEAD
    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()


class MedicalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], app: WebApp, verbose: bool = False):
        super().__init__(address, Handler)
        self.app = app
        self.verbose = verbose
        self.bind_host = str(address[0])
        # Host 头白名单：通配绑定（0.0.0.0 / ::）枚举不出合法主机名，只能用 None
        # 表示"不校验"，那种情况下必须靠令牌鉴权（--allow-remote 会强制生成令牌）。
        name = self.bind_host.strip().lower().rstrip(".")
        if name in _WILDCARD_HOSTS:
            self.allowed_hosts = None
        else:
            hosts = {name}
            # 本机机器名与 localhost 也接受：--allow-remote 绑定具体网卡时，浏览器
            # 通常用 http://<机器名>:<端口> 访问，只认绑定 IP 会把正常请求 403 掉。
            for candidate in _local_host_names():
                hosts.add(candidate)
            self.allowed_hosts: frozenset[str] | None = frozenset(hosts)

    def server_close(self) -> None:
        """关闭监听后统一收尾常驻 Library 连接（WAL 收尾 checkpoint 落在这里）。"""
        super().server_close()
        try:
            self.app.close_libraries()
        except Exception:  # noqa: BLE001 - 收尾阶段不再抛
            pass


def _int_arg(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        return int(query.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def _body_int(body: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    """取 JSON 体里的数值字段：非法值回落默认值，再夹到 ``[low, high]``。

    裸 ``int(body.get("limit"))`` 遇到 ``"abc"`` 会抛 ValueError，最后变成 500
    + Python 类名，前端只能看到一个错误框。
    """
    try:
        value = int(body.get(key) or default)
    except (TypeError, ValueError):
        value = default
    return max(low, min(value, high))


def create_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    root: Path | str = DEFAULT_ROOT,
    db: Path | str | None = None,
    verbose: bool = False,
    port_attempts: int = 20,
    token: str = "",
) -> MedicalHTTPServer:
    app = WebApp(root, db, token=token, allow_local_paths=is_loopback(host))
    app.warm_library()  # 把首查的开关库开销挪到启动阶段
    last_error: OSError | None = None
    for offset in range(port_attempts):
        try:
            server = MedicalHTTPServer((host, port + offset), app, verbose=verbose)
            if offset:
                print(f"[信息] 端口 {port} 被占用，改用 {port + offset}", flush=True)
            return server
        except OSError as exc:
            last_error = exc
    raise OSError(f"无法绑定端口 {port}-{port + port_attempts - 1}：{last_error}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="本地医学教材工作台网站")
    parser.add_argument("--host", default=DEFAULT_HOST, help="监听地址（默认 127.0.0.1；非本机地址需 --allow-remote）")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="允许监听非本机地址（危险：局域网可读全部 PDF/教材文本并触发 OCR），自动启用访问令牌",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="访问令牌（默认取 MEDICAL_RAG_TOKEN；监听非本机地址时未提供则自动生成并打印）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MEDICAL_RAG_WEB_PORT") or DEFAULT_PORT),
        help=f"监听端口（默认 {DEFAULT_PORT}）",
    )
    parser.add_argument("--root", type=Path, default=None, help="项目根目录（默认自动定位）")
    parser.add_argument("--db", type=Path, default=None, help="library.sqlite3 路径")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    parser.add_argument("--verbose", action="store_true", help="输出每个 HTTP 请求日志")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    loopback = is_loopback(args.host)
    if not loopback and not args.allow_remote:
        parser.error(
            f"--host {args.host} 会监听非本机地址，局域网内任何人都能读取全部 PDF 与教材文本、"
            "上传文件并触发 OCR/GPU 任务。\n"
            "如确实需要，请显式加上 --allow-remote（将自动启用访问令牌）。"
        )

    token = str(args.token or os.environ.get("MEDICAL_RAG_TOKEN") or "").strip()
    generated = False
    if not token and not loopback:
        token = secrets.token_urlsafe(24)
        generated = True

    server = create_server(
        args.host, args.port, args.root or DEFAULT_ROOT, args.db, verbose=args.verbose, token=token
    )
    host, port = server.server_address[0], server.server_address[1]
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    suffix = f"?token={urllib.parse.quote(token)}" if token else ""
    url = f"http://{display_host}:{port}/{suffix}"
    info = interpreter_info(Path(args.root or DEFAULT_ROOT).resolve())

    print("=" * 62)
    print("  本地医学教材工作台")
    print(f"  地址：{url}")
    if token:
        print(f"  访问令牌：{token}" + ("（自动生成，本次启动有效）" if generated else ""))
        if not loopback:
            print("  ⚠ 已监听非本机地址：局域网内需凭令牌访问，请勿对外网暴露")
    print(f"  项目：{server.app.root}")
    print(f"  索引库：{server.app.db}")
    print(f"  Python：{info['python']}")
    print(f"  OCR 环境：{info['ocr_python']}" + ("" if info["venv_ocr"] else "（未找到 .venv-ocr，使用主环境）"))
    print(f"  Paddle 环境：{info['paddle_python']}" + ("" if info["venv_ocr312"] else "（未找到 .venv-ocr312，使用主环境）"))
    print("=" * 62)
    print("  Ctrl+C 停止服务")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止…")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
