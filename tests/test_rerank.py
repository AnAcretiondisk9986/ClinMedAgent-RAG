from __future__ import annotations

import unittest

from medical_rag.embeddings import LexicalEmbedding, serialize
from medical_rag.rerank import (
    VECTOR_RERANK_SIZE,
    FeatureReranker,
    candidates_for_rerank,
    create_reranker,
)


def item(chunk_id: str, page: int, score: float, vector=None) -> dict:
    data = {"chunk_id": chunk_id, "page": page, "score": score, "match_reason": ["命中概念：骨膜"]}
    if vector is not None:
        data["vector"] = serialize(vector)
    return data


class FeatureRerankerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LexicalEmbedding()
        self.reranker = FeatureReranker()
        self.query = self.backend.embed(["骨膜含有丰富的血管"])[0]
        self.near = self.backend.embed(["骨膜含有丰富的血管和神经。"])[0]
        self.far = self.backend.embed(["胚胎发育分为三个胚层。"])[0]

    def test_vector_similarity_is_written_for_every_item(self) -> None:
        items = [item("a", 1, 10.0, self.near), item("b", 2, 9.0)]
        self.reranker.rerank(items, self.query)
        for entry in items:
            self.assertIn("vector_similarity", entry)
        self.assertEqual(items[1]["vector_similarity"], 0.0)  # 没有向量 → 0

    def test_internal_vector_field_is_always_removed(self) -> None:
        """即使算不出相似度（查询向量为空）也不能把内部字段泄漏出去。"""
        items = [item("a", 1, 10.0, self.near), item("b", 2, 9.0, self.far)]
        self.reranker.rerank(items, {})
        for entry in items:
            self.assertNotIn("vector", entry)
            self.assertEqual(entry["vector_similarity"], 0.0)

    def test_similar_item_overtakes_equal_lexical_score(self) -> None:
        items = [item("b", 2, 10.0, self.far), item("a", 1, 10.0, self.near)]
        self.reranker.rerank(items, self.query)
        self.assertEqual(items[0]["chunk_id"], "a")
        self.assertGreater(items[0]["score"], items[1]["score"])

    def test_high_similarity_is_explained_in_match_reason(self) -> None:
        items = [item("a", 1, 10.0, self.near)]
        self.reranker.rerank(items, self.query)
        self.assertTrue(
            any("向量相似度" in reason for reason in items[0]["match_reason"])
        )

    def test_low_similarity_does_not_add_reason_noise(self) -> None:
        items = [item("a", 1, 10.0, self.far)]
        self.reranker.rerank(items, self.query)
        self.assertFalse(
            any("向量相似度" in reason for reason in items[0]["match_reason"])
        )

    def test_lexical_score_is_preserved_when_no_vector(self) -> None:
        items = [item("a", 1, 10.0)]
        self.reranker.rerank(items, self.query)
        self.assertEqual(items[0]["score"], 10.0)

    def test_sorting_is_deterministic(self) -> None:
        items = [item("c", 2, 5.0), item("a", 1, 5.0), item("b", 1, 5.0)]
        self.reranker.rerank(items, {})
        self.assertEqual([entry["chunk_id"] for entry in items], ["a", "b", "c"])

    def test_empty_input_is_handled(self) -> None:
        self.assertEqual(self.reranker.rerank([], self.query), [])


class RerankerSelectionTests(unittest.TestCase):
    def test_default_is_feature_reranker(self) -> None:
        reranker, warning = create_reranker(None)
        self.assertIsInstance(reranker, FeatureReranker)
        self.assertEqual(warning, "")
        self.assertEqual(reranker.name, "feature-linear")

    def test_unknown_spec_falls_back_with_warning(self) -> None:
        """重排是可选增强：配置写错也必须退回可用实现，而不是让检索失败。"""
        reranker, warning = create_reranker("cross-encoder:whatever")
        self.assertIsInstance(reranker, FeatureReranker)
        self.assertIn("已回退特征式重排", warning)


class CandidateSelectionTests(unittest.TestCase):
    def test_candidates_are_capped_and_keep_order(self) -> None:
        scored = [(float(1000 - index), {"chunk_id": f"c{index}", "page": index, "score": 1.0})
                  for index in range(VECTOR_RERANK_SIZE + 50)]
        picked = candidates_for_rerank(scored)
        self.assertEqual(len(picked), VECTOR_RERANK_SIZE)
        self.assertEqual(picked[0]["chunk_id"], "c0")
        self.assertEqual(picked[-1]["chunk_id"], f"c{VECTOR_RERANK_SIZE - 1}")

    def test_short_input_is_returned_as_is(self) -> None:
        scored = [(1.0, {"chunk_id": "only", "page": 1, "score": 1.0})]
        self.assertEqual([entry["chunk_id"] for entry in candidates_for_rerank(scored)], ["only"])


if __name__ == "__main__":
    unittest.main()
