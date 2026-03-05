"""ChromaDB vector store abstraction.

All ChromaDB access is contained here.  To swap in pgvector: implement the
same public functions (upsert, query, delete_document, list_documents,
collection_stats) in a new store_pg.py and update the import in pipeline.py.
Nothing else changes.

Client caching
    _get_client() is LRU-cached on the resolved path string.  This avoids
    per-call client construction (overhead + potential file-lock races in tests).
    Tests should call _get_client.cache_clear() in teardown to reset state.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.config import Config


@lru_cache(maxsize=8)
def _get_client(path: str) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(config: Config):
    path = str(Path(config.chroma_path).expanduser())
    return _get_client(path).get_or_create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert(
    chunks: list[str],
    embeddings: list[list[float]],
    doc_id: str,
    config: Config,
    metadata: dict | None = None,
) -> None:
    """Upsert chunks into the vector store.  Idempotent on doc_id + chunk index."""
    collection = _get_collection(config)
    ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
    metas = [
        {**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metas,
    )


def query(query_embedding: list[float], config: Config) -> list[dict]:
    """Return top-k chunks at or above score_threshold, sorted by relevance desc."""
    collection = _get_collection(config)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=config.top_k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - dist  # cosine distance → cosine similarity
        if score >= config.score_threshold:
            chunks.append({"text": doc, "metadata": meta, "score": score})
    return sorted(chunks, key=lambda x: x["score"], reverse=True)


def delete_document(doc_id: str, config: Config) -> int:
    """Delete all chunks for a document.  Returns the count deleted."""
    collection = _get_collection(config)
    results = collection.get(where={"doc_id": doc_id})
    if results["ids"]:
        collection.delete(ids=results["ids"])
    return len(results["ids"])


def list_documents(config: Config) -> list[str]:
    """Return sorted unique doc_ids present in the collection."""
    collection = _get_collection(config)
    results = collection.get(include=["metadatas"])
    return sorted({m.get("doc_id", "unknown") for m in results["metadatas"]})


def collection_stats(config: Config) -> dict:
    """Return total chunk count and document list in a single collection access."""
    collection = _get_collection(config)
    results = collection.get(include=["metadatas"])
    doc_ids = sorted({m.get("doc_id", "unknown") for m in results["metadatas"]})
    return {
        "total_chunks": collection.count(),
        "documents": doc_ids,
    }
