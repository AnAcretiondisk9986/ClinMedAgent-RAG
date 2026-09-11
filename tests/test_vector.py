from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from medical_rag import embeddings
from medical_rag.embeddings import (
    LexicalEmbedding,
    OllamaEmbedding,
    bucket_of,
    cosine,
    create_backend,
    create_backend_cached,
    deserialize,
    serialize,
)
from medical_rag.library import Library


def make_library(root: Path, texts: dict[int, str]) -> Library:
    structured = root / "book" / "processed_v3" / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    body = "".join(f"## 原书第 {page} 页\n\n{text}\n\n" for page, text in texts.items())
    (structured / "01-章节.md").write_text(f"# 第一章\n\n{body}", encoding="utf-8")
    library = Library(root / "library.sqlite3")
    library.ingest_markdown_tree(root / "book" / "processed_v3", "测试教材")
    return library


class EmbeddingBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LexicalEmbedding()

    def test_identical_text_has_cosine_one(self) -> None:
        vector = self.backend.embed(["骨膜含有丰富的血管和神经。"])[0]
        self.assertAlmostEqual(cosine(vector, vector), 1.0, places=6)

    def test_unrelated_text_has_low_similarity(self) -> None:
        left = self.backend.embed(["骨膜含有丰富的血管和神经。"])[0]
        right = self.backend.embed(["胚胎发育分为三个胚层。"])[0]
        self.assertLess(cosine(left, right), 0.35)

    def test_similar_text_scores_higher_than_unrelated(self) -> None:
        query = self.backend.embed(["骨膜含有丰富的血管"])[0]
        near = self.backend.embed(["骨膜含有丰富的血管和神经。"])[0]
        far = self.backend.embed(["胚胎发育分为三个胚层。"])[0]
        self.assertGreater(cosine(query, near), cosine(query, far))

    def test_vectors_are_sparse_and_top_k_bounded(self) -> None:
        vector = self.backend.embed(["骨" * 5000])[0]
        self.assertLessEqual(len(vector), embeddings.VECTOR_TOP_K)

    def test_empty_and_single_char_text_yield_no_vector(self) -> None:
        self.assertEqual(self.backend.embed([""])[0], {})
        self.assertEqual(self.backend.embed(["骨"])[0], {})

    def test_bucket_hash_is_stable_across_processes(self) -> None:
        """不能用内置 hash()：它对 str 带随机盐，跨进程会让存好的向量对不上查询。"""
        self.assertEqual(bucket_of("骨膜"), bucket_of("骨膜"))
        self.assertLess(bucket_of("骨膜"), embeddings.VECTOR_DIM)
        self.assertNotEqual(bucket_of("骨膜"), bucket_of("血管"))

        import subprocess
        import sys

        code = "from medical_rag.embeddings import bucket_of; print(bucket_of('骨膜'))"
        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(int(done.stdout.strip()), bucket_of("骨膜"))

    def test_serialize_round_trip(self) -> None:
        vector = self.backend.embed(["骨膜含有丰富的血管和神经。"])[0]
        restored = deserialize(serialize(vector))
        # 序列化会保留 4 位小数，因此不能断言完全相等
        self.assertEqual(set(restored), set(vector))
        for bucket, weight in vector.items():
            self.assertAlmostEqual(restored[bucket], weight, places=4)

    def test_deserialize_tolerates_broken_input(self) -> None:
        self.assertEqual(deserialize(None), {})
        self.assertEqual(deserialize(""), {})
        self.assertEqual(deserialize("garbage ###"), {})
        self.assertEqual(deserialize("7:0.5 broken 9:0.25"), {7: 0.5, 9: 0.25})


class BackendSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        embeddings._BACKEND_CACHE.clear()

    def tearDown(self) -> None:
        embeddings._BACKEND_CACHE.clear()

    def test_default_is_zero_dependency_lexical(self) -> None:
        backend, warning = create_backend(None)
        self.assertIsInstance(backend, LexicalEmbedding)
        self.assertEqual(warning, "")
        self.assertEqual(backend.name, "lexical-bigram")

    def test_unknown_spec_falls_back_with_warning(self) -> None:
        backend, warning = create_backend("nope:model")
        self.assertIsInstance(backend, LexicalEmbedding)
        self.assertIn("未知的向量后端", warning)

    def test_unavailable_semantic_backend_falls_back(self) -> None:
        """Ollama 不可用时必须自动退回词法后端，而不是让检索报错。"""
        with mock.patch.object(OllamaEmbedding, "available", return_value=(False, "服务未开启 embeddings")):
            backend, warning = create_backend("ollama:bge-m3")
        self.assertIsInstance(backend, LexicalEmbedding)
        self.assertIn("已回退词法后端", warning)

    def test_available_semantic_backend_is_used(self) -> None:
        with mock.patch.object(OllamaEmbedding, "available", return_value=(True, "ok")):
            backend, warning = create_backend("ollama:bge-m3")
        self.assertEqual(backend.name, "ollama:bge-m3")
        self.assertEqual(warning, "")

    def test_cache_returns_same_instance(self) -> None:
        first = create_backend_cached("lexical")
        second = create_backend_cached("lexical")
        self.assertIs(first[0], second[0])

    def test_ollama_availability_never_raises(self) -> None:
        """探测失败只能返回 False，不能抛异常（Ollama 是可选增强）。"""
        ok, detail = OllamaEmbedding(model="definitely-not-a-model").available()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(detail, str)
        self.assertTrue(detail)


class VectorIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="medrag_vector_"))
        self.library = make_library(
            self.tmp,
            {
                10: "骨膜含有丰富的血管、神经和淋巴管。",
                11: "胚胎发育分为内胚层、中胚层和外胚层。",
            },
        )
        self.db = self.library.db

    def tearDown(self) -> None:
        self.library.close()

    def _count(self, sql: str, params: tuple = ()) -> int:
        return int(self.library.cx.execute(sql, params).fetchone()[0])

    def test_vectors_are_written_for_every_chunk(self) -> None:
        chunks = self._count("SELECT COUNT(*) FROM chunks")
        vectors = self._count("SELECT COUNT(*) FROM chunk_vectors")
        self.assertEqual(chunks, vectors)
        self.assertTrue(self.library.fts_integrity_ok())

    def test_backend_name_is_recorded(self) -> None:
        names = {row[0] for row in self.library.cx.execute("SELECT DISTINCT backend FROM chunk_vectors")}
        self.assertEqual(names, {self.library.embedding.name})

    def test_search_exposes_similarity_but_not_raw_vector(self) -> None:
        for item in self.library.search("骨膜含有丰富的血管", 3):
            self.assertIn("vector_similarity", item)
            self.assertNotIn("vector", item)
            self.assertNotIn("norm_text", item)

    def test_vector_channel_increases_relevant_scores(self) -> None:
        query = "骨膜含有丰富的血管、神经和淋巴管。"
        with_vectors = {item["chunk_id"]: item for item in self.library.search(query, 3)}
        self.assertTrue(any(item["vector_similarity"] > 0.5 for item in with_vectors.values()))

        self.library.cx.execute("DELETE FROM chunk_vectors")
        self.library.cx.commit()
        without_vectors = {item["chunk_id"]: item for item in self.library.search(query, 3)}

        self.assertEqual(set(with_vectors), set(without_vectors))
        for chunk_id, item in with_vectors.items():
            self.assertGreaterEqual(item["score"], without_vectors[chunk_id]["score"])
            self.assertEqual(without_vectors[chunk_id]["vector_similarity"], 0.0)
        self.assertTrue(
            any(
                with_vectors[key]["score"] > without_vectors[key]["score"]
                for key in with_vectors
            )
        )

    def test_reindex_does_not_duplicate_vectors(self) -> None:
        before = self._count("SELECT COUNT(*) FROM chunk_vectors")
        for _ in range(3):
            self.library.ingest_markdown_tree(self.tmp / "book" / "processed_v3", "测试教材")
        self.assertEqual(self._count("SELECT COUNT(*) FROM chunk_vectors"), before)
        self.assertTrue(self.library.fts_integrity_ok())

    def test_integrity_detects_missing_vectors(self) -> None:
        self.library.cx.execute(
            "DELETE FROM chunk_vectors WHERE rowid = (SELECT MIN(rowid) FROM chunks)"
        )
        self.library.cx.commit()
        self.assertFalse(self.library.fts_integrity_ok())

    def test_rebuild_restores_vectors(self) -> None:
        self.library.cx.execute("DELETE FROM chunk_vectors")
        self.library.cx.commit()
        self.assertFalse(self.library.fts_integrity_ok())

        self.library.rebuild_fts_index()

        self.assertTrue(self.library.fts_integrity_ok())
        self.assertTrue(self.library.search("骨膜", 3))

    def test_rebuild_vector_index_returns_row_count(self) -> None:
        chunks = self._count("SELECT COUNT(*) FROM chunks")
        self.assertEqual(self.library.rebuild_vector_index(), chunks)

    def test_legacy_database_backfills_vectors(self) -> None:
        chunks = self._count("SELECT COUNT(*) FROM chunks")
        self.library.cx.execute("DELETE FROM chunk_vectors")
        self.library.cx.commit()
        self.library.close()

        reopened = Library(self.db)
        try:
            self.assertEqual(
                int(reopened.cx.execute("SELECT COUNT(*) FROM chunk_vectors").fetchone()[0]), chunks
            )
            self.assertTrue(reopened.fts_integrity_ok())
        finally:
            reopened.close()

    def test_switching_backend_invalidates_and_rebuilds_vectors(self) -> None:
        """换嵌入后端后旧向量不可用，打开库时应按新后端重算。"""
        self.library.cx.execute("UPDATE chunk_vectors SET backend = 'other-backend'")
        self.library.cx.commit()
        self.assertFalse(self.library.fts_integrity_ok())
        self.library.close()

        reopened = Library(self.db)
        try:
            self.assertEqual(
                {row[0] for row in reopened.cx.execute("SELECT DISTINCT backend FROM chunk_vectors")},
                {reopened.embedding.name},
            )
            self.assertTrue(reopened.fts_integrity_ok())
        finally:
            reopened.close()

    def test_backfill_runs_at_most_once_when_index_is_complete(self) -> None:
        """已完整时不应反复回填：用 DELETE 计数变化来观察。"""
        self.library.close()
        reopened = Library(self.db)
        try:
            with mock.patch.object(Library, "_insert_vector", wraps=Library._insert_vector) as spy:
                # 触发一次 _init_schema 之外的显式回填调用
                reopened._backfill_vector_index()
                self.assertEqual(spy.call_count, 0)
        finally:
            reopened.close()


class EnvironmentConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        embeddings._BACKEND_CACHE.clear()

    def tearDown(self) -> None:
        embeddings._BACKEND_CACHE.clear()

    def test_env_var_selects_backend_for_library(self) -> None:
        tmp = Path(tempfile.mkdtemp(prefix="medrag_envvec_"))
        with mock.patch.dict(os.environ, {"MEDICAL_RAG_EMBEDDING": "lexical"}):
            library = Library(tmp / "library.sqlite3")
            try:
                self.assertEqual(library.embedding.name, "lexical-bigram")
                self.assertEqual(library.embedding_warning, "")
            finally:
                library.close()


if __name__ == "__main__":
    unittest.main()
