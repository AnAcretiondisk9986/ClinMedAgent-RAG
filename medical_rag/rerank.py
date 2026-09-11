"""候选证据重排（reranker）。

检索分两步：先由 FTS/二元组 + 词法打分选出**有界候选**，再由 reranker 重排。
默认实现 :class:`FeatureReranker` 是**特征式线性重排**：零依赖、可解释、可单测，
并把重排权重集中成显式常量，便于按回归集调参。

留出 :class:`Reranker` 协议，将来要换成交叉编码器或 LLM 重排（逐个候选打分）
时只需实现 ``rerank``，检索流程与数据库都不用改。

两个明确的取舍：

1. **不做独立召回**。没有 ANN 索引（sqlite-vec / faiss 属新依赖）时，向量若
   独立扫描全库会让检索延迟回到线性增长，抵消掉二元组索引带来的可扩展性，
   因此向量只参与对词法候选的重排。
2. **不内置 LLM 重排实现**。逐个候选调用本地大模型会让一次检索从几十毫秒变成
   数秒，而且本仓库当前环境无 embedding 模型可用、无法端到端验证——与其发一个
   未经验证、会拖慢检索的默认路径，不如把接口留出来（见 :func:`create_reranker`）。
   需要时按协议实现即可，``FeatureReranker`` 仍作为不可用时的回退。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .embeddings import cosine, deserialize

VECTOR_RERANK_SIZE = 200        # 参与重排的候选上限（有界，不随语料线性增长）
VECTOR_WEIGHT = 1.6             # 向量相似度对最终分数的贡献权重
VECTOR_REASON_THRESHOLD = 0.25  # 相似度高于此值才在 match_reason 里说明
DEFAULT_RERANKER = "feature"


class Reranker(Protocol):
    """重排协议：接一个交叉编码器 / LLM 重排只需实现 ``rerank``。"""

    name: str

    def rerank(
        self, items: list[dict], query_vector: Mapping[int, float]
    ) -> list[dict]:
        """就地对候选重排并返回；应保证返回顺序即最终顺序。"""
        ...


class FeatureReranker:
    """线性特征重排：词法分 + 向量相似度，并回写可解释字段。

    词法分数（含 #14 的短语/集中度/章节/降权等信号）已由 ``_python_search``
    算入 ``item["score"]``，这里只在它之上叠加向量通道，避免重复计权。
    """

    name = "feature-linear"
    VECTOR_WEIGHT = VECTOR_WEIGHT
    REASON_THRESHOLD = VECTOR_REASON_THRESHOLD

    def rerank(
        self, items: list[dict], query_vector: Mapping[int, float]
    ) -> list[dict]:
        if not items:
            return items
        # 先统一摘掉内部字段（无论能否算出相似度，都不能泄漏到 API 输出）
        vectors = [deserialize(item.pop("vector", None)) for item in items]
        if not query_vector:
            for item in items:
                item["vector_similarity"] = 0.0
            return self._sort(items)
        for item, vector in zip(items, vectors):
            similarity = cosine(dict(query_vector), vector)
            item["vector_similarity"] = round(similarity, 4)
            if similarity <= 0:
                continue
            item["score"] = round(float(item["score"]) + self.VECTOR_WEIGHT * similarity, 4)
            if similarity >= self.REASON_THRESHOLD:
                item["match_reason"] = list(item.get("match_reason") or []) + [
                    f"向量相似度 {similarity:.2f}"
                ]
        return self._sort(items)

    @staticmethod
    def _sort(items: list[dict]) -> list[dict]:
        items.sort(key=lambda item: (-item["score"], item["page"], item["chunk_id"]))
        return items


def create_reranker(spec: str | None = None) -> tuple[Reranker, str]:
    """按配置创建重排器，返回 (重排器, 警告说明)。

    ``MEDICAL_RAG_RERANKER``：``feature``（默认）。其他取值目前一律回退到
    ``feature`` 并给出警告——重排是可选增强，不能因为配置写错就让检索不可用。
    """
    spec = (spec or "").strip() or DEFAULT_RERANKER
    if spec == DEFAULT_RERANKER:
        return FeatureReranker(), ""
    return FeatureReranker(), f"未知的重排器配置 {spec!r}，已回退特征式重排 feature-linear"


def candidates_for_rerank(scored: Sequence[tuple[float, dict]]) -> list[dict]:
    """从 (score, item) 列表里取出参与重排的有界候选。"""
    return [item for _, item in scored[:VECTOR_RERANK_SIZE]]
