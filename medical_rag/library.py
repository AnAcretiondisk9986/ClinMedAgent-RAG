from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fitz

from .embeddings import create_backend_cached, serialize
from .outputs import is_internal_output
from .qa import (
    CONFIDENCE_ORDER,
    SCOPE_MIN_CONFIDENCE,
    load_book_aliases,
    match_book_ex,
    plan_question,
    strip_book_mention,
)
from .rerank import candidates_for_rerank, create_reranker

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / ".medical_rag" / "library.sqlite3"

# books 表的来源指纹：用于判断"PDF 已替换但索引还是旧的"。
# 旧版本建的库没有这些列，_migrate_books() 会幂等补齐。
SCHEMA_VERSION = 4
BOOK_FINGERPRINT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("source_path", "TEXT"),
    ("source_sha256", "TEXT"),
    ("source_mtime", "REAL"),
    ("source_size", "INTEGER"),
    ("content_hash", "TEXT"),
    ("pipeline_version", "TEXT"),
    ("indexed_at", "TEXT"),
    ("page_offset", "INTEGER"),
    ("quality_path", "TEXT"),
)
FINGERPRINT_FIELDS: tuple[str, ...] = tuple(name for name, _ in BOOK_FINGERPRINT_COLUMNS)


def _tokens(value: str) -> list[str]:
    value = value.lower()
    # Latin words stay whole; Chinese text uses single characters so medical
    # terms can still be found without a Chinese FTS tokenizer.
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", value)


def _normalise(value: str) -> str:
    return re.sub(r"\s+", "", _correct_terms(value).lower())

COMMON_OCR_FIXES = {
    "系统解部学": "系统解剖学", "解部学": "解剖学", "骨膜含有丰富的血管神经和淋巴管": "骨膜含有丰富的血管、神经和淋巴管",
    "肱骨头": "肱骨头", "胸锁乳突肌": "胸锁乳突肌", "迷走神经": "迷走神经",
}

def _correct_terms(value: str) -> str:
    for wrong, right in COMMON_OCR_FIXES.items():
        value = value.replace(wrong, right)
    return value


# ---------------------------------------------------------------- 排序信号
# 与 pdftext.RE_TOC_DOTS / tools_structure_v3.RE_TOC_DOTS 保持一致：
# 中文目录引导符是……（两个 U+2026），不是三个点
TOC_DOTS = re.compile(r"[…]{2,}|[.．·]{3,}")
FIGURE_MARK = "[图注/图例]"
SENTENCE_PUNCT = re.compile(r"[。；：！？]")


def _quality_penalty(text: str) -> tuple[float, list[str]]:
    """对目录页 / 页眉 / 图注等低价值块降权。

    这些块同样堆满医学名词，字面重合度很高，但不是教材正文结论；不降权会把
    真正的定义与正文挤下去。返回 (系数, 原因列表)。
    """
    penalty = 1.0
    reasons: list[str] = []
    if TOC_DOTS.search(text):
        penalty *= 0.55
        reasons.append("疑似目录页（点线引导）")
    if text.lstrip().startswith(FIGURE_MARK):
        penalty *= 0.75
        reasons.append("图注/图例")
    if len(text) < 30 and not SENTENCE_PUNCT.search(text):
        penalty *= 0.7
        reasons.append("文本过短（疑似页眉或标签）")
    return penalty, reasons


def _proximity(text: str, matched: list[str]) -> float:
    """命中概念在文本中的集中程度（0–1）。

    概念散在全页各处通常只是关键词堆砌；集中在一小段里才更可能是定义/结论。
    """
    if len(matched) < 2:
        return 0.0
    spans = [(index, index + len(term)) for term in matched if (index := text.find(term)) >= 0]
    if len(spans) < 2:
        return 0.0
    window = max(end for _, end in spans) - min(start for start, _ in spans)
    if window <= 60:
        return 1.0
    if window <= 160:
        return 0.6
    if window <= 400:
        return 0.3
    return 0.0


def _match_reasons(
    exact: bool,
    phrase_hits: list[str],
    proximity: float,
    section_hits: int,
    heading_hit: bool,
    matched: list[str],
    penalties: list[str],
) -> list[str]:
    """生成给 Agent 看的命中原因（为什么这条证据被选中）。"""
    reasons: list[str] = []
    if exact:
        reasons.append("完整题干原样出现")
    if phrase_hits:
        reasons.append("相邻概念命中：" + "、".join(phrase_hits[:2]))
    if proximity >= 0.6:
        reasons.append("概念集中出现")
    if section_hits:
        reasons.append("章节标题匹配")
    if heading_hit:
        reasons.append("段落开头命中")
    if matched:
        reasons.append("命中概念：" + "、".join(matched[:4]))
    reasons.extend(penalties)
    return reasons



@dataclass
class IngestReport:
    title: str
    path: str
    pages: int
    extractable_pages: int
    image_only_pages: int
    chunks: int
    database: str

    def __str__(self) -> str:
        warning = "；注意：该 PDF 没有可提取文字层，当前未建立可检索文本索引" if self.image_only_pages else ""
        return (
            f"已处理《{self.title}》：总页数 {self.pages}，可提取文字页 {self.extractable_pages}，"
            f"图片页 {self.image_only_pages}，文本块 {self.chunks}。数据库：{self.database}{warning}"
        )


class Library:
    """Local multi-book evidence library."""

    def __init__(self, db: Path | str | None = None):
        self.db = Path(db or os.environ.get("MEDICAL_RAG_DB") or DEFAULT_DB).expanduser().resolve()
        self.db.parent.mkdir(parents=True, exist_ok=True)
        # 教材缩写白名单（内置 + 可选的 aliases.json）
        self.aliases = load_book_aliases(self.db.parent)
        # 向量后端：默认零依赖词法后端；语义后端不可用时自动回退（见 embeddings.py）
        self.embedding, self.embedding_warning = create_backend_cached(
            os.environ.get("MEDICAL_RAG_EMBEDDING")
        )
        # 重排器：默认特征式线性重排（见 rerank.py）
        self.reranker, self.reranker_warning = create_reranker(
            os.environ.get("MEDICAL_RAG_RERANKER")
        )
        self.cx = sqlite3.connect(self.db, timeout=30.0)
        self.cx.row_factory = sqlite3.Row
        # WAL：重建索引的大写事务进行期间读者不被阻塞；busy_timeout：并发写自动串行化
        self.cx.execute("PRAGMA busy_timeout = 30000")
        self.cx.execute("PRAGMA journal_mode = WAL")
        self.cx.execute("PRAGMA synchronous = NORMAL")
        self._init_schema()

    def close(self) -> None:
        try:
            if self.cx.in_transaction:  # 未提交的写事务回滚，避免留下半套索引
                self.cx.rollback()
        except sqlite3.Error:
            pass
        finally:
            self.cx.close()

    def _init_schema(self) -> None:
        extra_columns = "".join(
            f",\n                {name} {kind}" for name, kind in BOOK_FINGERPRINT_COLUMNS
        )
        self.cx.executescript(
            f"""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS books (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                pages INTEGER NOT NULL,
                extractable_pages INTEGER NOT NULL DEFAULT 0,
                image_only_pages INTEGER NOT NULL DEFAULT 0,
                added_at TEXT NOT NULL{extra_columns}
            );
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT NOT NULL UNIQUE,
                book_id INTEGER NOT NULL REFERENCES books(id) ON DELETE CASCADE,
                page INTEGER NOT NULL,
                section TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_book_page ON chunks(book_id, page);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, section, content='chunks', content_rowid='rowid'
            );
            -- 中文二元组索引：unicode61 会把一整段连续中文当成一个 token，
            -- 所以 MATCH '"上皮组织"' 永远不会命中；切成字符二元组后每个二元组
            -- 成为独立 token，中文子串检索才可用。norm_text 順便缓存规范化文本，
            -- 避免每次检索都对全部候选重算 _normalise()。
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts_ngram USING fts5(
                bigrams, norm_text
            );
            -- 向量通道：稀疏向量（bucket:weight），backend 用于后端切换后失效重建
            CREATE TABLE IF NOT EXISTS chunk_vectors (
                rowid INTEGER PRIMARY KEY,
                backend TEXT NOT NULL,
                vector TEXT NOT NULL
            );
            """
        )
        self._migrate_books()
        self._backfill_ngram_index()
        self._backfill_vector_index()
        self.cx.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.cx.commit()

    def _migrate_books(self) -> None:
        """老库幂等补齐来源指纹列（不重建表，不丢数据）。"""
        existing = {row["name"] for row in self.cx.execute("PRAGMA table_info(books)")}
        for name, kind in BOOK_FINGERPRINT_COLUMNS:
            if name not in existing:
                self.cx.execute(f"ALTER TABLE books ADD COLUMN {name} {kind}")

    @staticmethod
    def _bigram_text(normalized: str) -> str:
        """把规范化文本切成字符二元组（空格分隔），供 FTS5 索引。"""
        if len(normalized) < 2:
            return normalized
        return " ".join(normalized[i:i + 2] for i in range(len(normalized) - 1))

    def _ngram_values(self, text: str) -> tuple[str, str]:
        """返回 (二元组字符串, 规范化文本)。"""
        normalized = _normalise(text)
        return self._bigram_text(normalized), normalized

    def _insert_ngram(self, rowid: int, text: str) -> None:
        bigrams, normalized = self._ngram_values(text)
        self.cx.execute(
            "INSERT INTO chunks_fts_ngram(rowid, bigrams, norm_text) VALUES(?,?,?)",
            (int(rowid), bigrams, normalized),
        )

    def _backfill_ngram_index(self) -> None:
        """老库一次性回填二元组索引（幂等）；已齐则只做两次 COUNT。"""
        chunks = int(self.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if not chunks:
            return
        indexed = int(self.cx.execute("SELECT COUNT(*) FROM chunks_fts_ngram").fetchone()[0])
        if indexed >= chunks:
            return
        self.cx.execute("DELETE FROM chunks_fts_ngram")
        for row in self.cx.execute("SELECT rowid, text FROM chunks").fetchall():
            self._insert_ngram(row["rowid"], row["text"])

    def _insert_vector(self, rowid: int, text: str) -> None:
        vector = self.embedding.embed([_normalise(text)])[0]
        self.cx.execute(
            "INSERT INTO chunk_vectors(rowid, backend, vector) VALUES(?,?,?)",
            (int(rowid), self.embedding.name, serialize(vector)),
        )

    def _vector_count(self) -> int:
        return int(
            self.cx.execute(
                "SELECT COUNT(*) FROM chunk_vectors WHERE backend = ?", (self.embedding.name,)
            ).fetchone()[0]
        )

    def _backfill_vector_index(self) -> None:
        """老库一次性回填向量索引（幂等）；换后端后也会自动重建。"""
        chunks = int(self.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if not chunks:
            return
        if self._vector_count() >= chunks:
            return
        self.cx.execute("DELETE FROM chunk_vectors")
        for row in self.cx.execute("SELECT rowid, text FROM chunks").fetchall():
            self._insert_vector(row["rowid"], row["text"])

    def rebuild_vector_index(self) -> int:
        """重建向量索引（换嵌入后端或修复后调用）。"""
        self.cx.execute("DELETE FROM chunk_vectors")
        for row in self.cx.execute("SELECT rowid, text FROM chunks").fetchall():
            self._insert_vector(row["rowid"], row["text"])
        self.cx.commit()
        return self._vector_count()

    def _upsert_book(
        self,
        book_id: int | None,
        title: str,
        content_path: str,
        pages: int,
        extractable: int,
        image_only: int,
        fingerprint: dict[str, Any],
        now: str,
    ) -> int:
        """写入 / 更新 books 行，并一并写入来源指纹。"""
        # indexed_at 不在 fingerprint 里（由本次写入时间统一决定），必须显式补上，
        # 否则 FINGERPRINT_FIELDS 里的 indexed_at 会写进 NULL
        values = {**fingerprint, "indexed_at": now}
        fingerprint_values = [values.get(name) for name in FINGERPRINT_FIELDS]
        if book_id is not None:
            assignments = ", ".join(
                ["title=?", "pages=?", "extractable_pages=?", "image_only_pages=?", "added_at=?"]
                + [f"{name}=?" for name in FINGERPRINT_FIELDS]
            )
            self.cx.execute(
                f"UPDATE books SET {assignments} WHERE id=?",
                (title, pages, extractable, image_only, now, *fingerprint_values, book_id),
            )
            return int(book_id)
        columns = [
            "title", "path", "pages", "extractable_pages", "image_only_pages", "added_at",
            *FINGERPRINT_FIELDS,
        ]
        self.cx.execute(
            f"INSERT INTO books({', '.join(columns)}) VALUES({', '.join('?' for _ in columns)})",
            (title, content_path, pages, extractable, image_only, now, *fingerprint_values),
        )
        return int(self.cx.execute("SELECT last_insert_rowid()").fetchone()[0])

    @staticmethod
    def _file_sha256(path: Path, block: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while True:
                chunk = stream.read(block)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _source_fingerprint(self, text_root: Path, files: list[Path]) -> dict[str, Any]:
        """来源指纹：PDF 的路径/hash/mtime/大小 + 结构化内容 hash + 质量元数据。

        只在**建索引时**计算（包含一次 PDF 全文件 sha256），扫描/查询路径只用
        mtime+size 做廉价比对，避免每次列书库都重算几百 MB 的哈希。
        """
        from .workspace import find_pdf  # 延迟导入：workspace 反向依赖 library

        fingerprint: dict[str, Any] = {
            "source_path": None,
            "source_sha256": None,
            "source_mtime": None,
            "source_size": None,
            "content_hash": None,
            "pipeline_version": None,
            "page_offset": None,
            "quality_path": None,
        }
        book_dir = text_root.parent
        pdf = find_pdf(book_dir) if book_dir.is_dir() else None
        if pdf is not None:
            fingerprint["source_path"] = str(pdf)
            try:
                stat = pdf.stat()
                fingerprint["source_mtime"] = float(stat.st_mtime)
                fingerprint["source_size"] = int(stat.st_size)
                fingerprint["source_sha256"] = self._file_sha256(pdf)
            except OSError:
                pass

        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\x00")
            try:
                digest.update(path.read_bytes())
            except OSError:
                digest.update(b"<unreadable>")
            digest.update(b"\x01")
        fingerprint["content_hash"] = digest.hexdigest()

        quality_path = text_root / "quality.json"
        if quality_path.exists():
            fingerprint["quality_path"] = str(quality_path)
            try:
                quality = json.loads(quality_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                quality = None
            if isinstance(quality, dict):
                if quality.get("pipeline"):
                    fingerprint["pipeline_version"] = str(quality["pipeline"])[:200]
                offset = quality.get("page_offset")
                if isinstance(offset, int):
                    fingerprint["page_offset"] = offset
        return fingerprint

    def _delete_book_chunks(self, book_id: int) -> None:
        """删除某本书的全部块，并同步维护三套索引。

        ``chunks_fts`` 是 external-content FTS5 表：``'delete'`` 命令必须带上
        原始列值（text/section），否则旧词条不会被移除。``chunks.rowid`` 会被
        复用，残留词条会让检索命中已经不存在的内容（幽灵结果），同时
        ``integrity-check`` 会报 ``database disk image is malformed``。

        ``chunks_fts_ngram`` 与 ``chunk_vectors`` 都靠 ``rowid IN (SELECT ...
        FROM chunks ...)`` 定位，**必须在删除 chunks 之前执行**，否则子查询为
        空、旧行残留，重新索引时会撞 UNIQUE 约束或产生重复条目。
        """
        self.cx.execute(
            "INSERT INTO chunks_fts(chunks_fts, rowid, text, section) "
            "SELECT 'delete', rowid, text, section FROM chunks WHERE book_id = ?",
            (book_id,),
        )
        self.cx.execute(
            "DELETE FROM chunks_fts_ngram WHERE rowid IN (SELECT rowid FROM chunks WHERE book_id = ?)",
            (book_id,),
        )
        self.cx.execute(
            "DELETE FROM chunk_vectors WHERE rowid IN (SELECT rowid FROM chunks WHERE book_id = ?)",
            (book_id,),
        )
        self.cx.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))

    def fts_integrity_ok(self) -> bool:
        """检查三套索引与 chunks 是否一致。

        1) chunks_fts（external content）用 rank=1 逐行与 content 表核对；
        2) chunks_fts_ngram（独立 FTS5）比对行数；
        3) chunk_vectors（普通表）比对行数，少行意味着有块参与不了向量重排。
        """
        try:
            self.cx.execute("INSERT INTO chunks_fts(chunks_fts, rank) VALUES('integrity-check', 1)")
        except sqlite3.DatabaseError:
            return False
        chunks = int(self.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        indexed = int(self.cx.execute("SELECT COUNT(*) FROM chunks_fts_ngram").fetchone()[0])
        return indexed == chunks and self._vector_count() == chunks

    def rebuild_fts_index(self) -> int:
        """重建三套索引（修复历史遗留的 chunks/chunks_fts 不一致）。"""
        self.cx.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self.cx.execute("DELETE FROM chunks_fts_ngram")
        self.cx.execute("DELETE FROM chunk_vectors")
        for row in self.cx.execute("SELECT rowid, text FROM chunks").fetchall():
            self._insert_ngram(row["rowid"], row["text"])
            self._insert_vector(row["rowid"], row["text"])
        self.cx.commit()
        return int(self.cx.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @staticmethod
    def _chunk_page(text: str, max_chars: int = 1200) -> Iterable[str]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        buffer = ""
        for para in paragraphs:
            parts = [para]
            if len(para) > max_chars:
                parts = [p.strip() for p in re.split(r"(?<=[。！？；.!?;])", para) if p.strip()]
            for part in parts:
                if buffer and len(buffer) + len(part) + 1 > max_chars:
                    yield buffer
                    buffer = ""
                buffer = f"{buffer} {part}".strip()
        if buffer:
            yield buffer

    def ingest(self, pdf: Path | str, title: str | None = None) -> IngestReport:
        path = Path(pdf).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".pdf":
            raise ValueError("目前只支持 PDF 文件")

        doc = fitz.open(path)
        pages = len(doc)
        title = title or path.stem
        extractable = 0
        image_only = 0
        page_chunks: list[tuple[int, str, str]] = []
        for page_number, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if not text:
                image_only += 1
                continue
            extractable += 1
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            section = lines[0][:160] if lines else f"第{page_number}页"
            for chunk in self._chunk_page(text):
                page_chunks.append((page_number, section, chunk))
        doc.close()

        existing = self.cx.execute("SELECT id FROM books WHERE path = ?", (str(path),)).fetchone()
        book_id = existing[0] if existing else None
        if book_id is not None:
            self._delete_book_chunks(book_id)
        now = datetime.now(timezone.utc).isoformat()
        try:
            stat = path.stat()
            source_mtime, source_size = float(stat.st_mtime), int(stat.st_size)
        except OSError:
            source_mtime, source_size = None, None
        source_sha256 = self._file_sha256(path)
        fingerprint = {
            "source_path": str(path),
            "source_sha256": source_sha256,
            "source_mtime": source_mtime,
            "source_size": source_size,
            # 直接从 PDF 抽文本时，内容 = 源文件，两者哈希等同
            "content_hash": source_sha256,
            "pipeline_version": "pdf-text（PyMuPDF 逐页抽取）",
            "page_offset": None,
            "quality_path": None,
        }
        book_id = self._upsert_book(
            book_id, title, str(path), pages, extractable, image_only, fingerprint, now
        )

        for ordinal, (page, section, text) in enumerate(page_chunks):
            digest = hashlib.sha1(f"{path}:{page}:{ordinal}:{text}".encode("utf-8")).hexdigest()[:20]
            cur = self.cx.execute(
                "INSERT INTO chunks(chunk_id,book_id,page,section,text) VALUES(?,?,?,?,?)",
                (digest, book_id, page, section, text),
            )
            self.cx.execute(
                "INSERT INTO chunks_fts(rowid,text,section) VALUES(?,?,?)",
                (cur.lastrowid, text, section),
            )
            self._insert_ngram(cur.lastrowid, text)
            self._insert_vector(cur.lastrowid, text)
        self.cx.commit()
        return IngestReport(title, str(path), pages, extractable, image_only, len(page_chunks), str(self.db))

    def ingest_markdown_tree(self, text_dir: Path | str, title: str | None = None) -> IngestReport:
        """Index processed Markdown while preserving chapter/page citations."""
        root = Path(text_dir).expanduser().resolve()
        files = sorted((root / "structured").glob("*.md")) if (root / "structured").exists() else sorted(root.rglob("*.md"))
        files = [f for f in files if f.name not in {"README.md", "book.md"} and not is_internal_output(f)]
        if not files:
            raise FileNotFoundError(f"未找到可索引 Markdown：{root}")
        title = title or root.parent.name
        # 整本重建在一个写事务里完成：中途失败不会留下半套索引，并发重建在此串行化
        self.cx.execute("BEGIN IMMEDIATE")
        existing = self.cx.execute("SELECT id FROM books WHERE path = ?", (str(root),)).fetchone()
        book_id = existing[0] if existing else None
        if book_id is not None:
            self._delete_book_chunks(book_id)
        now = datetime.now(timezone.utc).isoformat()
        fingerprint = self._source_fingerprint(root, files)
        book_id = self._upsert_book(
            book_id, title, str(root), len(files), len(files), 0, fingerprint, now
        )
        count = 0
        max_source_page = 0
        for file in files:
            content = file.read_text(encoding="utf-8", errors="ignore")
            chapter = file.stem
            page = 0
            heading = ""
            buffer = []
            def flush():
                nonlocal count, buffer, page
                text = " ".join(x.strip() for x in buffer if x.strip()).strip()
                if not text: return
                section = f"{chapter} · {heading}" if heading else chapter
                digest = hashlib.sha1(f"{root}:{file}:{page}:{count}:{text}".encode("utf-8")).hexdigest()[:20]
                cur2 = self.cx.execute("INSERT INTO chunks(chunk_id,book_id,page,section,text) VALUES(?,?,?,?,?)", (digest, book_id, page, section, text))
                self.cx.execute("INSERT INTO chunks_fts(rowid,text,section) VALUES(?,?,?)", (cur2.lastrowid, text, section))
                self._insert_ngram(cur2.lastrowid, text)
                self._insert_vector(cur2.lastrowid, text)
                count += 1; buffer = []
            for line in content.splitlines():
                m = re.match(r"^## 原书第\s*(\d+)\s*页", line)
                if m:
                    flush(); page = int(m.group(1)); max_source_page = max(max_source_page, page); heading = ""; continue
                hm = re.match(r"^#{3,6}\s+(.+?)\s*$", line)
                if hm:
                    # 节/小节标题：开始新块，并作为后续证据的 section 元数据
                    flush(); heading = hm.group(1); buffer.append(heading); continue
                if line.startswith("# ") or line.startswith("> ") or not line.strip():
                    continue
                if line.startswith("[图注/图例]"):
                    buffer.append("[图注/图例] " + line[len("[图注/图例]"):].strip())
                else:
                    buffer.append(line)
                if sum(len(x) for x in buffer) >= 1200: flush()
            flush()
        self.cx.commit()
        page_count = max_source_page or len(files)
        self.cx.execute("UPDATE books SET pages=?, extractable_pages=?, image_only_pages=0 WHERE id=?", (page_count, page_count, book_id))
        self.cx.commit()
        return IngestReport(title, str(root), page_count, page_count, 0, count, str(self.db))

    def book_freshness(self, book_id: int) -> dict[str, Any]:
        """判断索引是否落后于源 PDF（识别“PDF 已替换但仍显示旧索引”）。

        刻意只比对 mtime + size：对几百 MB 的扫描件每次列书库都重算 sha256 太慢。
        需要强校验时用 :meth:`verify_source_hash`。
        """
        row = self.cx.execute("SELECT * FROM books WHERE id = ?", (int(book_id),)).fetchone()
        if row is None:
            return {"known": False, "book_id": int(book_id)}
        data = dict(row)
        info: dict[str, Any] = {
            "known": True,
            "book_id": int(book_id),
            "title": data.get("title"),
            "source_path": data.get("source_path"),
            "source_mtime": data.get("source_mtime"),
            "source_size": data.get("source_size"),
            "content_hash": data.get("content_hash"),
            "pipeline_version": data.get("pipeline_version"),
            "indexed_at": data.get("indexed_at"),
            "page_offset": data.get("page_offset"),
            "quality_path": data.get("quality_path"),
            "has_fingerprint": data.get("content_hash") is not None,
            "stale": False,
            "reason": "",
        }
        if not info["has_fingerprint"]:
            info.update(stale=True, reason="索引没有来源指纹（旧版本建立），建议重建索引")
            return info
        source = data.get("source_path")
        if not source or not Path(source).exists():
            info.update(stale=True, reason=f"源 PDF 不存在：{source}")
            return info
        try:
            stat = Path(source).stat()
        except OSError as exc:
            info.update(stale=True, reason=f"源 PDF 不可读：{exc}")
            return info
        mtime_changed = data.get("source_mtime") is None or abs(
            float(data["source_mtime"]) - stat.st_mtime
        ) > 1e-6
        size_changed = data.get("source_size") is None or int(data["source_size"]) != stat.st_size
        if mtime_changed or size_changed:
            info.update(stale=True, reason="源 PDF 已变化（mtime/大小不同），索引可能过期")
        return info

    def verify_source_hash(self, book_id: int) -> dict[str, Any]:
        """强校验：重算源 PDF 的 sha256 并与索引记录比对（慢，按需调用）。"""
        info = self.book_freshness(book_id)
        if not info.get("known") or not info.get("source_path"):
            return {**info, "hash_matches": False}
        stored = info.get("source_sha256") or self.cx.execute(
            "SELECT source_sha256 FROM books WHERE id = ?", (int(book_id),)
        ).fetchone()[0]
        try:
            current = self._file_sha256(Path(info["source_path"]))
        except OSError as exc:
            return {**info, "hash_matches": False, "reason": f"无法读取源 PDF：{exc}"}
        matches = bool(stored) and stored == current
        return {
            **info,
            "hash_matches": matches,
            "source_sha256": current,
            "stale": info["stale"] or not matches,
            "reason": info["reason"] or ("" if matches else "源 PDF 内容已变化（sha256 不同）"),
        }

    def stale_books(self) -> list[dict[str, Any]]:
        """返回索引可能过期的教材（供 doctor / 网页提示）。"""
        return [
            info
            for info in (self.book_freshness(row["id"]) for row in self.list_books())
            if info.get("stale")
        ]

    def list_books(self) -> list[dict]:
        return [dict(row) for row in self.cx.execute("SELECT * FROM books ORDER BY title")]

    def get_book(self, book_id: int) -> dict | None:
        row = self.cx.execute("SELECT * FROM books WHERE id=?", (int(book_id),)).fetchone()
        return dict(row) if row else None

    def chunks_for_page(self, book_id: int, page: int) -> list[dict]:
        """按书 + 原书页码返回该页的全部证据块（用于网页预览正文）。"""
        rows = self.cx.execute(
            "SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book "
            "FROM chunks c JOIN books b ON b.id=c.book_id "
            "WHERE c.book_id=? AND c.page=? ORDER BY c.rowid",
            (int(book_id), int(page)),
        ).fetchall()
        return [dict(row) for row in rows]

    def page_range(self, book_id: int) -> tuple[int, int] | None:
        row = self.cx.execute(
            "SELECT MIN(page) AS lo, MAX(page) AS hi FROM chunks WHERE book_id=?",
            (int(book_id),),
        ).fetchone()
        if row is None or row["lo"] is None:
            return None
        return int(row["lo"]), int(row["hi"])

    def _candidate_rows(
        self, q: str, tokens: list[str], book_ids: set[int] | None = None
    ) -> list[sqlite3.Row]:
        """取候选证据块。

        优先用二元组 FTS5 索引定位候选：既避免全表扫描，又顺带取出缓存的
        ``norm_text``（不再对每一块重算 _normalise）。
        出现单字词时无法用二元组表达，回退全表扫描以保证召回不降级。
        """
        bigrams: set[str] = set()
        if tokens and all(len(token) >= 2 for token in tokens):
            bigrams.update(q[i:i + 2] for i in range(max(0, len(q) - 1)))
            for token in tokens:
                bigrams.update(token[i:i + 2] for i in range(len(token) - 1))
        bigrams = {gram for gram in bigrams if gram.strip()}

        if bigrams:
            match = " OR ".join(f'"{gram}"' for gram in sorted(bigrams))
            sql = (
                "SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book,b.path,"
                "n.norm_text AS norm_text, v.vector AS vector "
                "FROM chunks_fts_ngram n JOIN chunks c ON c.rowid = n.rowid "
                "JOIN books b ON b.id = c.book_id "
                "LEFT JOIN chunk_vectors v ON v.rowid = c.rowid "
                "WHERE n.bigrams MATCH ?"
            )
            params: list[Any] = [match]
            if book_ids:
                placeholders = ",".join("?" for _ in book_ids)
                sql += f" AND c.book_id IN ({placeholders})"
                params.extend(sorted(book_ids))
            return self.cx.execute(sql, params).fetchall()

        sql = (
            "SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book,b.path,"
            "NULL AS norm_text, v.vector AS vector "
            "FROM chunks c JOIN books b ON b.id=c.book_id "
            "LEFT JOIN chunk_vectors v ON v.rowid = c.rowid"
        )
        if book_ids:
            placeholders = ",".join("?" for _ in book_ids)
            return self.cx.execute(
                f"{sql} WHERE c.book_id IN ({placeholders})", tuple(sorted(book_ids))
            ).fetchall()
        return self.cx.execute(sql).fetchall()

    def _python_search(self, query: str, limit: int, book_ids: set[int] | None = None) -> list[dict]:
        q = _normalise(query)
        # Prefer multi-character medical concepts over single Chinese
        # characters. The latter are retained as a fallback for short OCR
        # fragments and Latin terms.
        from .qa import extract_concepts

        concepts = [_normalise(term) for term in extract_concepts(query)]
        tokens = [_normalise(t) for t in concepts if _normalise(t)]
        if not tokens:
            tokens = [_normalise(t) for t in _tokens(query) if t.strip()]
        if not q or not tokens:
            return []
        rows = self._candidate_rows(q, tokens, book_ids)
        scored: list[tuple[float, sqlite3.Row]] = []
        # Character n-grams help recall OCR variants; concept coverage and
        # metadata boosts prevent generic words such as “位置/特点” from
        # dominating the ranking.
        qgrams = {q[i:i + 2] for i in range(max(0, len(q) - 1))} or {q}
        # 相邻概念连写（如“肩关节组成”），在定义式行文里会真正出现
        phrases = [f"{left}{right}" for left, right in zip(tokens, tokens[1:])]
        for row in rows:
            # norm_text 来自二元组索引的缓存；回退全表扫描时为 NULL，现场规范化
            cached = row["norm_text"]
            raw_text = row["text"]
            text = cached if cached is not None else _normalise(raw_text)
            section = _normalise(row["section"])
            book = _normalise(row["book"])
            grams = {text[i:i + 2] for i in range(max(0, len(text) - 1))}
            overlap = len(qgrams & grams) / max(len(qgrams), 1)
            matched = [term for term in tokens if term in text]
            # Require at least one meaningful concept match. This avoids
            # returning a page only because a generic question phrase overlaps.
            if not matched:
                continue
            coverage = len(matched) / max(len(tokens), 1)
            hits = sum(text.count(term) for term in matched)
            exact = q in text
            phrase_hits = [phrase for phrase in phrases if phrase in text]
            proximity = _proximity(text, matched)
            section_hits = sum(1 for term in tokens if term in section)
            book_boost = sum(0.25 for term in tokens if term in book)
            concept_boost = sum(min(text.count(term), 3) * 0.55 for term in matched)
            heading_hit = any(term in text[:40] for term in matched)
            penalty, penalties = _quality_penalty(raw_text)
            score = (
                overlap * 1.4
                + coverage * 4.0
                + hits * 0.08
                + (4.0 if exact else 0.0)
                + concept_boost
                + section_hits * 0.7
                + book_boost
                + len(phrase_hits) * 1.6
                + proximity * 1.2
                + (0.8 if heading_hit else 0.0)
            ) * penalty
            if score <= 0:
                continue
            item = dict(row)
            item.pop("norm_text", None)  # 内部缓存字段，不对外暴露
            # 注意：vector 必须留到重排阶段再用，不能在这里 pop
            item["score"] = round(score, 4)
            item["match_reason"] = _match_reasons(
                exact, phrase_hits, proximity, section_hits, heading_hit, matched, penalties
            )
            scored.append((score, item))
        scored.sort(key=lambda x: (-x[0], x[1]["page"], x[1]["chunk_id"]))
        # 向量通道只对词法候选做有界重排（见 rerank.py：不独立全库召回）
        ranked = candidates_for_rerank(scored)
        self.reranker.rerank(ranked, self.embedding.embed([q])[0])
        return ranked[:limit]

    def _resolve_book_ids(self, book: int | str | None, books: list[dict] | None = None) -> set[int] | None:
        """把 book 参数解析成书籍 id 集合：支持 id、数字字符串、书名/缩写（如“组胚”）。

        None 表示不限定（全部教材）；显式给出但找不到时返回空集合。
        """
        if book is None:
            return None
        if isinstance(book, (set, frozenset, list, tuple)):  # 内部已解析好的 id 集合
            return {int(item) for item in book}
        rows = books if books is not None else self.list_books()
        if isinstance(book, int) or (isinstance(book, str) and book.strip().isdigit()):
            target = int(book)
            return {target} if any(row["id"] == target for row in rows) else set()
        text = str(book).strip()
        if not text:
            return None
        matched = match_book_ex(text, rows, self.aliases)
        if matched is not None and (
            CONFIDENCE_ORDER.get(matched.confidence, 0) >= CONFIDENCE_ORDER[SCOPE_MIN_CONFIDENCE]
        ):
            return {matched.book["id"]}
        return {row["id"] for row in rows if text in row["title"]} or set()

    def search(self, query: str, limit: int = 5, book: int | str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 50))
        book_ids = self._resolve_book_ids(book)
        if book_ids is not None and not book_ids:
            return []
        results = self._python_search(query, limit, book_ids)
        if results:
            return results
        # FTS5 remains useful for Latin terminology even when the local
        # Chinese scorer cannot find a meaningful multi-character concept.
        terms = [t for t in _tokens(query) if re.fullmatch(r"[a-z0-9_]+", t)]
        if not terms:
            return []
        match = " OR ".join(f'"{t.replace(chr(34), "")}"' for t in terms)
        if book_ids:
            placeholders = ",".join("?" for _ in book_ids)
            rows = self.cx.execute(
                f"""SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book,b.path
                   FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid
                   JOIN books b ON b.id=c.book_id
                   WHERE chunks_fts MATCH ? AND c.book_id IN ({placeholders})
                   ORDER BY bm25(chunks_fts) LIMIT ?""",
                (match, *sorted(book_ids), limit),
            ).fetchall()
        else:
            rows = self.cx.execute(
                """SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book,b.path
                   FROM chunks_fts f JOIN chunks c ON c.rowid=f.rowid
                   JOIN books b ON b.id=c.book_id
                   WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?""",
                (match, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_context(self, chunk_id: str, radius: int = 1, limit: int = 5) -> list[dict]:
        """Return nearby chunks from the same book for cross-page answers."""
        row = self.cx.execute("SELECT book_id,page FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if not row:
            return []
        radius = max(0, min(int(radius), 3))
        limit = max(1, min(int(limit), 10))
        rows = self.cx.execute(
            """SELECT c.chunk_id,c.page,c.section,c.text,b.title AS book,b.path
               FROM chunks c JOIN books b ON b.id=c.book_id
               WHERE c.book_id=? AND c.page BETWEEN ? AND ?
               ORDER BY c.page,c.rowid LIMIT ?""",
            (row["book_id"], row["page"] - radius, row["page"] + radius, limit),
        ).fetchall()
        return [dict(item) for item in rows]

    def answer_question(self, question: str, limit: int = 6, book: int | str | None = None) -> dict:
        """Create a citation-first evidence pack for an exam question.

        This method intentionally does not invent a final medical answer. In
        MCP mode the calling agent uses this evidence pack to write the answer
        and cite the returned pages.

        ``book`` 可显式限定教材（id / 书名 / 缩写）；不传时会自动识别题干里点名
        的教材（如“组胚里…”），并把它从题干中剥离后再抽取概念。
        """
        books = self.list_books()
        scoped_question = question
        book_ids = self._resolve_book_ids(book, books) if book is not None else None
        if book_ids is not None and not book_ids:
            return {"status": "invalid_book", "message": f"未找到教材：{book}"}
        scope = None
        mention = match_book_ex(question, books, self.aliases)
        if (
            mention is not None
            and CONFIDENCE_ORDER.get(mention.confidence, 0)
            >= CONFIDENCE_ORDER[SCOPE_MIN_CONFIDENCE]
            and (book_ids is None or mention.book["id"] in book_ids)
        ):
            scoped_question = strip_book_mention(question, mention.mention)
            if book_ids is None:
                book_ids = {mention.book["id"]}
            scope = {
                "source": "explicit" if book is not None else "question",
                "book_id": mention.book["id"],
                "title": mention.book["title"],
                "mention": mention.mention,
                # 置信度与命中原因：低置信度不会走到这里（SCOPE_MIN_CONFIDENCE），
                # 调用方可据此判断自动限域是否可靠。
                "confidence": mention.confidence,
                "reason": mention.reason,
            }
        elif book_ids is not None:
            row = next((item for item in books if item["id"] in book_ids), None)
            scope = {
                "source": "explicit",
                "book_id": next(iter(book_ids)) if len(book_ids) == 1 else None,
                "title": row["title"] if row else str(book),
                "mention": None,
            }

        plan = plan_question(scoped_question)
        if not plan.question:
            return {"status": "invalid_question", "message": "题目不能为空"}

        limit = max(1, min(int(limit), 20))
        merged: dict[str, dict] = {}
        for query_index, query in enumerate(plan.queries):
            for rank, result in enumerate(self.search(query, max(limit * 2, 8), book=book_ids), start=1):
                item = dict(result)
                # Earlier passes use the full question and should dominate;
                # single-concept passes are only a recall fallback and must not
                # outrank a result matching the whole question.
                pass_weights = [1.20, 0.95, 0.38, 0.28]
                pass_weight = pass_weights[query_index] if query_index < len(pass_weights) else 0.22
                score = float(item.get("score", 0.0)) * pass_weight + max(0.0, 0.18 - rank * 0.01)
                item["retrieval_score"] = round(score, 4)
                old = merged.get(item["chunk_id"])
                if old is None or score > old["retrieval_score"]:
                    merged[item["chunk_id"]] = item

        ranked = sorted(
            merged.values(),
            key=lambda item: (-item["retrieval_score"], item["page"], item["chunk_id"]),
        )
        evidence = self._dedupe_evidence(ranked, limit)
        self._annotate_adjacent_pages(evidence)
        for item in evidence:
            item["context"] = self.get_context(item["chunk_id"], radius=1, limit=5)

        if plan.question_type == "choice":
            guidance = "逐项核对题干和选项，不能仅凭相似词判断；答案必须引用教材证据。"
        elif plan.question_type == "compare":
            guidance = "按比较维度组织答案，例如组成、位置、结构、功能和临床意义。"
        elif plan.question_type == "definition":
            guidance = "先给出教材定义，再补充位置、组成、特点或作用。"
        elif plan.question_type == "why":
            guidance = "按‘结构/机制 → 结果’解释原因，并区分教材原文与推断。"
        else:
            guidance = "围绕题干逐点作答；如果证据不足，应明确说明知识库未检索到依据。"
        if scope:
            guidance = f"{guidance} 检索范围已限定为《{scope['title']}》。"

        result = {
            "status": "evidence_found" if evidence else "no_evidence",
            "plan": plan.as_dict(),
            "evidence": evidence,
            "answer_guidance": guidance,
            "citation_rule": "引用《书名》、原书页码、章节和 chunk_id；不得把未检索到的内容说成教材结论。",
        }
        if scope:
            result["book_scope"] = scope
        return result

    @staticmethod
    def _dedupe_evidence(items: list[dict], limit: int, max_per_page: int = 2) -> list[dict]:
        """去掉重复或被包含的证据，并限制同一页最多几条。

        同一段正文可能因切块边界重叠、相邻页重叠而被多次召回。重复证据只是
        占掉上下文预算，还会让 Agent 误以为存在多处独立依据。

        相邻页不做“合并成一条”：证据必须保留具体页码才能被引用校验，而且
        ``get_context(radius=1)`` 已经会带回前后页内容；合并反而会弱化页码定位。
        这里改为在 :meth:`_annotate_adjacent_pages` 里标注相邻页。
        """
        result: list[dict] = []
        chosen: list[str] = []
        per_page: dict[tuple, int] = {}
        for item in items:
            normalized = _normalise(item.get("text") or "")
            if not normalized:
                continue
            if any(
                normalized == other or normalized in other or other in normalized
                for other in chosen
            ):
                continue
            key = (item.get("book"), item.get("page"))
            if per_page.get(key, 0) >= max_per_page:
                continue
            per_page[key] = per_page.get(key, 0) + 1
            chosen.append(normalized)
            result.append(item)
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _annotate_adjacent_pages(evidence: list[dict]) -> None:
        """标注“同一本书里相邻页也入选”，让 Agent 知道证据在后续页延续。"""
        pages_by_book: dict[Any, set[int]] = {}
        for item in evidence:
            pages_by_book.setdefault(item.get("book"), set()).add(item.get("page"))
        for item in evidence:
            neighbours = sorted(
                page
                for page in pages_by_book.get(item.get("book"), set())
                if isinstance(page, int)
                and isinstance(item.get("page"), int)
                and abs(page - item["page"]) == 1
            )
            if neighbours:
                item["adjacent_pages"] = neighbours
                item["match_reason"] = list(item.get("match_reason") or []) + [
                    "相邻页也有证据（第 " + "、".join(str(page) for page in neighbours) + " 页）"
                ]

    def get_chunk(self, chunk_id: str) -> dict | None:
        row = self.cx.execute(
            "SELECT c.*,b.title AS book,b.path FROM chunks c JOIN books b ON b.id=c.book_id WHERE c.chunk_id=?",
            (chunk_id,),
        ).fetchone()
        return dict(row) if row else None

