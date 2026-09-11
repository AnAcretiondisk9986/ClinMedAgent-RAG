"""向量后端：可插拔的嵌入接口 + 零依赖的词法后端。

设计目标：把「向量」这一层做成**可替换的后端**，这样今天不装任何东西也能用，
将来接入真正的语义模型（本地 ONNX / Ollama / 远端 API）时只需换后端，检索
流程与数据库结构都不用改。

内置 :class:`LexicalEmbedding` 是**词法级**表示（字符二元组 hashing + 次线性
TF + L2 归一化），它能提供一路与集合式 overlap 不同的相似度通道（长度归一化、
词频饱和），但**不做同义词泛化**（"心梗" 与 "心肌梗死" 仍不相似）。这是明确
的取舍，不要把它当成语义检索。

``embed()`` 的输入契约：texts 应为**已规范化**文本（library._normalise 处理过：
小写、去空白、已做 OCR 定向纠错）。中文检索不需要保留空格，因此语义后端接
规范化文本同样可用。
"""

from __future__ import annotations

import json
import math
import re
import urllib.error
import urllib.request
import zlib
from typing import Any, Protocol, Sequence

VECTOR_DIM = 1 << 17      # 131072 个哈希桶
VECTOR_TOP_K = 96         # 每块只保留权重最高的若干桶（稀疏剪枝，控存储）
EMBEDDING_MAX_CHARS = 4000  # 单块参与向量化的最大长度
OLLAMA_TIMEOUT = 30

# 向量通道只对词法候选做**有界重排**（不独立全库召回）。
# 理由：没有 ANN 索引（sqlite-vec / faiss 属新依赖）时，向量若独立扫描全库会
# 让检索延迟回到线性增长，反而抵消掉二元组索引带来的可扩展性。
VECTOR_RERANK_SIZE = 200  # 参与向量重排的候选上限
VECTOR_WEIGHT = 1.6       # 向量相似度对最终分数的贡献权重
VECTOR_REASON_THRESHOLD = 0.25  # 相似度高于此值才在 match_reason 里说明

_WHITESPACE = re.compile(r"\s+")
DEFAULT_BACKEND = "lexical"


class EmbeddingBackend(Protocol):
    """向量后端协议：接一个语义模型只需实现 ``embed``。"""

    name: str

    def embed(self, texts: Sequence[str]) -> list[dict[int, float]]:
        """把已规范化文本转成稀疏向量（bucket → weight）。"""
        ...


def bucket_of(gram: str) -> int:
    """稳定的二元组哈希桶。

    不能用内置 ``hash()``：CPython 对 str 的哈希带随机盐，跨进程不稳定，
    存进数据库后就对不上查询向量了。
    """
    return zlib.crc32(gram.encode("utf-8")) % VECTOR_DIM


class LexicalEmbedding:
    """零依赖词法后端：字符二元组 + 次线性 TF + L2 归一化。"""

    name = "lexical-bigram"

    @staticmethod
    def _vectorise(normalized: str) -> dict[int, float]:
        text = _WHITESPACE.sub("", normalized).lower()[:EMBEDDING_MAX_CHARS]
        if len(text) < 2:
            return {}
        counts: dict[int, int] = {}
        for index in range(len(text) - 1):
            bucket = bucket_of(text[index:index + 2])
            counts[bucket] = counts.get(bucket, 0) + 1
        # 次线性 TF：高频词不因为重复出现就线性压过其他词
        weighted = {bucket: 1.0 + math.log(count) for bucket, count in counts.items()}
        top = sorted(weighted.items(), key=lambda item: (-item[1], item[0]))[:VECTOR_TOP_K]
        norm = math.sqrt(sum(weight * weight for _, weight in top)) or 1.0
        return {bucket: weight / norm for bucket, weight in top}

    def embed(self, texts: Sequence[str]) -> list[dict[int, float]]:
        return [self._vectorise(text) for text in texts]


class OllamaEmbedding:
    """可选语义后端：调用本机 Ollama 的 embedding 接口。

    需要两件事同时满足，否则 :meth:`available` 为 False，调用方应回退到词法后端：
      1. Ollama 以 ``--embeddings`` 启动（否则接口直接报错）；
      2. 已 ``ollama pull`` 一个 embedding 模型（如 bge-m3 / nomic-embed-text）。

    注意：本仓库当前环境（Ollama 未开 --embeddings、且无 embedding 模型）下
    这条路径**无法端到端验证**，因此默认不启用，且所有失败都会回退到词法后端。
    """

    def __init__(self, model: str = "bge-m3", host: str = "http://127.0.0.1:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"
        self._dim = 0

    def available(self) -> tuple[bool, str]:
        """探测 Ollama embedding 是否真的可用，返回 (可用, 说明)。"""
        try:
            vectors = self.embed(["探测"])
        except Exception as exc:  # noqa: BLE001 - 任何失败都视为不可用
            return False, f"{type(exc).__name__}: {exc}"
        if not vectors or not vectors[0]:
            return False, "Ollama 返回了空向量"
        return True, f"Ollama {self.model}（{len(vectors[0])} 维）"

    def embed(self, texts: Sequence[str]) -> list[dict[int, float]]:
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
            body = json.loads(response.read().decode("utf-8"))
        embeddings = body.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise ValueError("Ollama 未返回预期的 embeddings 字段")
        result: list[dict[int, float]] = []
        for vector in embeddings:
            dense = [float(value) for value in vector]
            norm = math.sqrt(sum(value * value for value in dense)) or 1.0
            # 语义向量是稠密的，这里转为稀疏字典以复用同一套存储与打分
            result.append({index: value / norm for index, value in enumerate(dense) if value})
        return result


_BACKEND_CACHE: dict[str, tuple[EmbeddingBackend, str]] = {}


def create_backend_cached(spec: str | None = None) -> tuple[EmbeddingBackend, str]:
    """带进程级缓存的 :func:`create_backend`。

    探测语义后端可能要走网络；webapp 每个请求都会新建 Library，因此必须缓存，
    不能每次构造都去探测。
    """
    key = (spec or "").strip() or DEFAULT_BACKEND
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = create_backend(key)
    return _BACKEND_CACHE[key]


def create_backend(spec: str | None = None) -> tuple[EmbeddingBackend, str]:
    """按配置创建后端，返回 (后端, 警告说明)。

    配置来自 ``MEDICAL_RAG_EMBEDDING``：``lexical``（默认）或
    ``ollama:<model>``。语义后端不可用时**自动回退词法后端**并返回警告，
    保证检索永远可用（Ollama 是可选增强，不是硬依赖）。
    """
    """按配置创建后端，返回 (后端, 警告说明)。

    配置来自 ``MEDICAL_RAG_EMBEDDING``：``lexical``（默认）或
    ``ollama:<model>``。语义后端不可用时**自动回退词法后端**并返回警告，
    保证检索永远可用（Ollama 是可选增强，不是硬依赖）。
    """
    spec = (spec or "").strip()
    if not spec or spec == DEFAULT_BACKEND:
        return LexicalEmbedding(), ""
    if spec.startswith("ollama:"):
        model = spec.split(":", 1)[1].strip() or "bge-m3"
        backend = OllamaEmbedding(model=model)
        ok, detail = backend.available()
        if ok:
            return backend, ""
        return (
            LexicalEmbedding(),
            f"配置的语义后端不可用（{detail}），已回退词法后端 lexical-bigram",
        )
    return LexicalEmbedding(), f"未知的向量后端配置 {spec!r}，已回退词法后端"


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    """余弦相似度。两侧均已 L2 归一化，因此只需点积。"""
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(bucket, 0.0) for bucket, weight in left.items())


def serialize(vector: dict[int, float]) -> str:
    """稀疏向量序列化为紧凑字符串（bucket:weight，空格分隔）。"""
    return " ".join(f"{bucket}:{weight:.4f}" for bucket, weight in sorted(vector.items()))


def deserialize(blob: str | None) -> dict[int, float]:
    result: dict[int, float] = {}
    for part in (blob or "").split():
        bucket, _, weight = part.partition(":")
        try:
            result[int(bucket)] = float(weight)
        except ValueError:
            continue
    return result
