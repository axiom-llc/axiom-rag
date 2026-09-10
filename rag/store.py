"""ChromaDB persistence for the in-process RAG pipeline."""
import threading
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings

from rag.config import Config

_init_lock = threading.Lock()
_write_lock = threading.RLock()


@lru_cache(maxsize=8)
def _get_client(path: str) -> chromadb.PersistentClient:
    with _init_lock:
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
    """Replace all chunks for doc_id with the supplied chunks and embeddings."""
    with _write_lock:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        collection = _get_collection(config)
        existing = collection.get(where={"doc_id": doc_id}, include=[])
        if not chunks:
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            return

        ids = [f"{doc_id}::{index}" for index in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "doc_id": doc_id, "chunk_index": index}
            for index in range(len(chunks))
        ]
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )
        # Validate and write the replacement before deleting stale chunks. In
        # particular, a rejected embedding dimension must not destroy the document.
        stale = sorted(set(existing["ids"]) - set(ids))
        if stale:
            collection.delete(ids=stale)


def store_query(query_embedding: list[float], config: Config) -> list[dict]:
    """Return relevant chunks at or above score_threshold, sorted descending."""
    collection = _get_collection(config)
    count = collection.count()
    if count == 0:
        return []
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(config.top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    chunks = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        score = 1.0 - float(distance)
        if score >= config.score_threshold:
            chunks.append({"text": document, "metadata": metadata, "score": score})
    return sorted(chunks, key=lambda chunk: chunk["score"], reverse=True)


def delete_document(doc_id: str, config: Config) -> int:
    """Delete all chunks for a document and return the deleted count."""
    with _write_lock:
        collection = _get_collection(config)
        results = collection.get(where={"doc_id": doc_id}, include=[])
        ids = results["ids"]
        if ids:
            collection.delete(ids=ids)
        return len(ids)


def list_documents(config: Config) -> list[str]:
    """Return sorted unique document identifiers in the collection."""
    collection = _get_collection(config)
    results = collection.get(include=["metadatas"])
    return sorted(
        {
            metadata.get("doc_id", "unknown")
            for metadata in (results.get("metadatas") or [])
            if metadata is not None
        }
    )


def collection_stats(config: Config) -> dict:
    """Return total chunk count and sorted document identifiers."""
    collection = _get_collection(config)
    return {"total_chunks": collection.count(), "documents": list_documents(config)}


def query(query_embedding: list[float], config: Config) -> list[dict]:
    return store_query(query_embedding, config)
