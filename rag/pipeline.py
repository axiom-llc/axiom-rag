"""Public interface for the RAG pipeline.

Two primary functions: ingest() and query().
All other modules are implementation details; callers should not import them.

ingest() — chunk → embed → store
query()  — embed query → retrieve → generate
"""
from __future__ import annotations
from pathlib import Path

from rag.config import Config
from rag import chunker, embedder, store, generator


def ingest(
    text: str,
    doc_id: str,
    config: Config,
    metadata: dict | None = None,
    strategy: str = "fixed",
) -> dict:
    """Chunk, embed, and store a document.

    Args:
        text:     Full document text.
        doc_id:   Stable unique identifier (filename, URL, UUID, …).
        config:   Resolved Config instance.
        metadata: Optional key-value pairs stored alongside every chunk.
        strategy: "fixed" (default) or "sentences".

    Returns:
        {"doc_id": str, "chunks_stored": int}
    """
    if strategy == "sentences":
        chunks = chunker.chunk_sentences(text, config.chunk_size)
    else:
        chunks = chunker.chunk_fixed(text, config.chunk_size, config.chunk_overlap)

    if not chunks:
        return {"doc_id": doc_id, "chunks_stored": 0}

    all_embeddings = embedder.embed_texts(chunks, config)
    store.upsert(chunks, all_embeddings, doc_id, config, metadata)
    return {"doc_id": doc_id, "chunks_stored": len(chunks)}


def query(question: str, config: Config) -> dict:
    """Retrieve relevant chunks and generate a grounded answer.

    Returns:
        {
            "answer":      str,
            "sources":     list[str],
            "chunks":      list[dict],  # retrieved context with scores
            "chunk_count": int,
        }
    """
    q_embedding = embedder.embed_query(question, config)
    chunks = store.query(q_embedding, config)
    result = generator.generate_answer(question, chunks, config)
    return {**result, "chunks": chunks}


def ingest_file(
    path: str,
    config: Config,
    metadata: dict | None = None,
    strategy: str = "fixed",
) -> dict:
    """Read a file and ingest it.  doc_id is set to the filename."""
    p = Path(path).expanduser()
    text = p.read_text(encoding="utf-8", errors="replace")
    return ingest(text, doc_id=p.name, config=config, metadata=metadata, strategy=strategy)


def ingest_directory(
    directory: str,
    config: Config,
    extensions: list[str] | None = None,
    strategy: str = "fixed",
) -> list[dict]:
    """Ingest all matching files in a directory (recursive).

    Args:
        extensions: File extensions to match (default: [".txt", ".md"]).

    Returns:
        List of ingest result dicts, one per file.
    """
    d = Path(directory).expanduser()
    exts = set(extensions or [".txt", ".md"])
    results = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix in exts:
            results.append(ingest_file(str(p), config, strategy=strategy))
    return results
