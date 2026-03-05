"""Gemini embedding adapter.

Stateless — takes text, returns float vectors.  Batches up to 100 texts per
API call (Gemini embedding API limit).

task_type asymmetry
    Gemini's text-embedding-004 model is trained with asymmetric tasks:
    - "retrieval_document" for content being indexed
    - "retrieval_query"    for the search query at retrieval time
    Using the wrong type measurably degrades retrieval precision.
    Both are set explicitly; do not remove them.
"""
from __future__ import annotations
from google import genai
from google.genai import types
from rag.config import Config

_BATCH_LIMIT = 100


def _client(config: Config) -> genai.Client:
    return genai.Client(api_key=config.gemini_api_key)


def embed_texts(texts: list[str], config: Config) -> list[list[float]]:
    """Embed a list of document texts.  Returns one vector per text."""
    config.requires_api_key()
    client = _client(config)
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), _BATCH_LIMIT):
        batch = texts[i : i + _BATCH_LIMIT]
        response = client.models.embed_content(
            model=config.embedding_model,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        all_embeddings.extend(e.values for e in response.embeddings)
    return all_embeddings


def embed_query(query: str, config: Config) -> list[float]:
    """Embed a single query string for retrieval."""
    config.requires_api_key()
    client = _client(config)
    response = client.models.embed_content(
        model=config.embedding_model,
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return list(response.embeddings[0].values)
