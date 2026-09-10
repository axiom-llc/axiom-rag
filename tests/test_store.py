"""Tests for rag.store — uses a real in-memory ChromaDB instance, no Gemini."""
import pytest
import tempfile
import os

from rag.config import load_config
from rag import store


@pytest.fixture()
def cfg(tmp_path):
    """Config pointing to a temp ChromaDB directory."""
    store._get_client.cache_clear()
    yield load_config(
        embedding_dimension=8,
        gemini_api_key="test-key",
        chroma_path=str(tmp_path / "chroma"),
        collection_name="test_col",
        embedding_model="gemini-embedding-2",
    )
    store._get_client.cache_clear()


def _fake_embedding(dim: int = 8) -> list[float]:
    import random
    v = [random.random() for _ in range(dim)]
    norm = sum(x**2 for x in v) ** 0.5
    return [x / norm for x in v]


def _upsert_doc(cfg, doc_id: str, n_chunks: int = 3):
    chunks = [f"chunk {i} of {doc_id}" for i in range(n_chunks)]
    embeddings = [_fake_embedding() for _ in range(n_chunks)]
    store.upsert(chunks, embeddings, doc_id, cfg)
    return chunks, embeddings


class TestUpsertAndList:
    def test_upsert_appears_in_list(self, cfg):
        _upsert_doc(cfg, "doc_a")
        assert "doc_a" in store.list_documents(cfg)

    def test_multiple_docs_listed(self, cfg):
        _upsert_doc(cfg, "doc_a")
        _upsert_doc(cfg, "doc_b")
        docs = store.list_documents(cfg)
        assert "doc_a" in docs and "doc_b" in docs

    def test_list_returns_unique_ids(self, cfg):
        _upsert_doc(cfg, "dup")
        _upsert_doc(cfg, "dup")  # re-upsert same doc
        assert store.list_documents(cfg).count("dup") == 1

    def test_upsert_idempotent(self, cfg):
        _upsert_doc(cfg, "doc_a", n_chunks=3)
        _upsert_doc(cfg, "doc_a", n_chunks=3)
        stats = store.collection_stats(cfg)
        assert stats["total_chunks"] == 3  # not 6


class TestDelete:
    def test_delete_removes_chunks(self, cfg):
        _upsert_doc(cfg, "to_delete", n_chunks=4)
        n = store.delete_document("to_delete", cfg)
        assert n == 4
        assert "to_delete" not in store.list_documents(cfg)

    def test_delete_nonexistent_returns_zero(self, cfg):
        assert store.delete_document("ghost", cfg) == 0

    def test_delete_only_targets_doc(self, cfg):
        _upsert_doc(cfg, "keep")
        _upsert_doc(cfg, "remove")
        store.delete_document("remove", cfg)
        assert "keep" in store.list_documents(cfg)


class TestQuery:
    def test_score_threshold_filters_low_scores(self, cfg):
        # Insert one chunk then query with a very high threshold
        chunks = ["relevant content about the topic"]
        embeddings = [_fake_embedding()]
        store.upsert(chunks, embeddings, "doc_q", cfg)
        # High threshold should filter everything (random embeddings ≠ query)
        high_cfg = load_config(
            embedding_dimension=8,
            gemini_api_key="test-key",
            chroma_path=str(cfg.chroma_path),
            collection_name=cfg.collection_name,
            embedding_model=cfg.embedding_model,
            score_threshold=0.9999,
            top_k=5,
        )
        results = store.query(_fake_embedding(), high_cfg)
        assert results == []

    def test_results_sorted_by_score_desc(self, cfg):
        # Insert multiple chunks; even with random embeddings, sort order must hold
        for i in range(5):
            store.upsert(
                [f"doc content {i}"],
                [_fake_embedding()],
                f"doc_{i}",
                cfg,
            )
        low_threshold_cfg = load_config(
            embedding_dimension=8,
            gemini_api_key="test-key",
            chroma_path=str(cfg.chroma_path),
            collection_name=cfg.collection_name,
            embedding_model=cfg.embedding_model,
            score_threshold=0.0,
            top_k=5,
        )
        results = store.query(_fake_embedding(), low_threshold_cfg)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_collection_returns_empty(self, cfg):
        results = store.query(_fake_embedding(), cfg)
        assert results == []


class TestStats:
    def test_stats_shape(self, cfg):
        _upsert_doc(cfg, "a", n_chunks=2)
        _upsert_doc(cfg, "b", n_chunks=3)
        s = store.collection_stats(cfg)
        assert s["total_chunks"] == 5
        assert set(s["documents"]) == {"a", "b"}


def test_failed_replacement_preserves_existing_document(cfg):
    store.upsert(["old one", "old two"], [[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6], "keep", cfg)
    before = store._get_collection(cfg).get(where={"doc_id": "keep"})
    with pytest.raises(Exception, match="dimension"):
        store.upsert(["new"], [[1.0, 0.0, 0.0]], "keep", cfg)
    assert store._get_collection(cfg).get(where={"doc_id": "keep"}) == before


def test_mismatched_replacement_lengths_preserve_document(cfg):
    store.upsert(["old"], [[1.0] + [0.0] * 7], "keep", cfg)
    with pytest.raises(ValueError, match="same length"):
        store.upsert(["new"], [], "keep", cfg)
    assert store._get_collection(cfg).get(where={"doc_id": "keep"})["documents"] == ["old"]


def test_concurrent_replacements_do_not_mix_document_revisions(cfg):
    from concurrent.futures import ThreadPoolExecutor
    store.upsert(["original"] * 3, [[1.0] + [0.0] * 7] * 3, "shared", cfg)

    def replace_document(revision):
        store.upsert([f"revision-{revision}"] * revision, [[1.0] + [0.0] * 7] * revision, "shared", cfg)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(replace_document, [6, 2, 7, 1, 5, 3, 8, 4]))
    documents = store._get_collection(cfg).get(where={"doc_id": "shared"})["documents"]
    assert len(set(documents)) == 1
    assert len(documents) == int(documents[0].split("-")[-1])
