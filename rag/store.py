"""ChromaDB persistence for the in-process RAG pipeline."""
from functools import wraps
from pathlib import Path

from rag.config import Config
from rag.space import EmbeddingSpace
from rag import persistence
from chromadb.errors import NotFoundError

_write_lock = persistence.lock


def _get_client(path):
    return persistence.get_client(path)


def _retain_owners():
    """Legacy test cache-clear hook: ownership must last until process exit."""


_get_client.cache_clear = _retain_owners


def _serialized(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        persistence.check_process()
        with _write_lock:
            return function(*args, **kwargs)
    return wrapped


@_serialized
def _existing_collection(config: Config):
    path = str(Path(config.chroma_path).expanduser())
    try:
        return _get_client(path).get_collection(config.collection_name, embedding_function=None)
    except NotFoundError:
        return None


@_serialized
def create_collection(config: Config):
    """Create a fresh tagged namespace. Never adopt or overwrite an existing one."""
    identity = EmbeddingSpace.configured(config)
    return _get_client(str(Path(config.chroma_path).expanduser())).create_collection(
        name=config.collection_name,
        metadata={"hnsw:space": "cosine", **identity.metadata()},
        embedding_function=None,
    )


@_serialized
def _get_collection(config: Config):
    identity = EmbeddingSpace.configured(config)
    collection = _existing_collection(config)
    if collection is None:
        # A concurrent creator causes an explicit failure, never metadata adoption.
        collection = create_collection(config)
    identity.validate(collection.metadata, config.collection_name)
    return collection


@_serialized
def upsert(
    chunks: list[str],
    embeddings: list[list[float]],
    doc_id: str,
    config: Config,
    metadata: dict | None = None,
) -> None:
    """Replace all chunks for doc_id with the supplied chunks and embeddings."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    EmbeddingSpace.configured(config).validate_vectors(embeddings)
    collection = _get_collection(config)
    persistence.replace(collection, config.chroma_path, chunks, embeddings, doc_id, metadata)


@_serialized
def store_query(query_embedding: list[float], config: Config) -> list[dict]:
    """Return relevant chunks at or above score_threshold, sorted descending."""
    EmbeddingSpace.configured(config).validate_vectors([query_embedding])
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


@_serialized
def delete_document(doc_id: str, config: Config) -> int:
    """Delete all chunks for a document and return the deleted count."""
    collection = _get_collection(config)
    return persistence.replace(collection, config.chroma_path, [], [], doc_id, None)


@_serialized
def list_documents(config: Config) -> list[str]:
    """Return sorted unique document identifiers in the collection."""
    collection = _existing_collection(config)
    if collection is None:
        return []
    results = collection.get(include=["metadatas"])
    return sorted(
        {
            metadata.get("doc_id", "unknown")
            for metadata in (results.get("metadatas") or [])
            if metadata is not None
        }
    )


@_serialized
def collection_stats(config: Config) -> dict:
    """Return total chunk count and sorted document identifiers."""
    collection = _existing_collection(config)
    return {"total_chunks": collection.count() if collection else 0, "documents": list_documents(config)}


def query(query_embedding: list[float], config: Config) -> list[dict]:
    return store_query(query_embedding, config)
