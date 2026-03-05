"""LLM answer generation over retrieved context.

Stateless.  Takes retrieved chunks and the original query; returns a grounded
answer dict.

Grounding discipline (enforced in the system prompt, not configurable by
default): the model must answer only from the provided context passages, cite
doc_id inline, and explicitly state when context is insufficient rather than
speculate.  The system_prompt parameter exists for domain adaptation
(different languages, specialised instruction sets) but changing it to allow
prior-knowledge use defeats the purpose of the pipeline.
"""
from __future__ import annotations
from google import genai
from google.genai import types
from rag.config import Config

_DEFAULT_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the provided context passages.
If the context does not contain enough information to answer, say so clearly.
Do not use prior knowledge. Do not speculate beyond the context.
Cite the source document (doc_id) inline when referencing specific facts."""


def generate_answer(
    query: str,
    context_chunks: list[dict],
    config: Config,
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> dict:
    """Generate a grounded answer from retrieved chunks.

    Args:
        query:          The user's question.
        context_chunks: Output of store.query() — list of {text, metadata, score}.
        config:         Resolved Config instance.
        system_prompt:  Override for domain adaptation; default enforces strict
                        grounding.

    Returns:
        {
            "answer":      str,
            "sources":     list[str],   # unique doc_ids referenced
            "chunk_count": int,
        }
    """
    if not context_chunks:
        return {
            "answer": "No relevant documents found for this query.",
            "sources": [],
            "chunk_count": 0,
        }

    config.requires_api_key()
    context_block = "\n\n".join(
        f"[{c['metadata'].get('doc_id', 'unknown')}] {c['text']}"
        for c in context_chunks
    )
    prompt = f"Context:\n{context_block}\n\nQuestion: {query}"

    config.requires_api_key()
    client = genai.Client(api_key=config.gemini_api_key)
    response = client.models.generate_content(
        model=config.generation_model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    sources = sorted({c["metadata"].get("doc_id", "unknown") for c in context_chunks})

    return {
        "answer": response.text,
        "sources": sources,
        "chunk_count": len(context_chunks),
    }
