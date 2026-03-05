"""Text chunking strategies.

Both functions return list[str].  Strategy is selected at ingest time via
pipeline.ingest(strategy=...).

Fixed-size (default)
    Splits on word boundaries to approximately chunk_size words, with a
    configurable overlap between consecutive chunks.  Overlap preserves context
    that would otherwise be severed at a boundary.

Sentence
    Groups sentences (split on sentence-ending punctuation) until chunk_size
    words is reached.  Better recall for prose; less predictable for structured
    or technical documents.
"""
from __future__ import annotations
import re


def chunk_fixed(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping fixed-size chunks (word-boundary aligned)."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def chunk_sentences(text: str, chunk_size: int) -> list[str]:
    """Group sentences into chunks of up to chunk_size words."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if count + word_count > chunk_size and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += word_count
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]
