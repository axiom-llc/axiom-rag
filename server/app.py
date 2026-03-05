"""Flask REST API.

Endpoints
    POST   /ingest
    POST   /query
    GET    /documents
    DELETE /documents/<doc_id>
    GET    /stats

Config is loaded lazily on first request so that importing this module in
tests does not raise even when GEMINI_API_KEY is absent.
"""
from __future__ import annotations
from flask import Flask, request, jsonify, abort
from rag.config import load_config, Config
from rag import pipeline, store

app = Flask(__name__)

_config: Config | None = None


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


# ── Error handling ────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_error(exc: Exception):
    try:
        from google.genai import errors as genai_errors
        if isinstance(exc, genai_errors.APIError):
            app.logger.error("Gemini API error: %s", exc)
            return jsonify({"error": "upstream API error", "detail": str(exc)}), 502
    except ImportError:
        pass
    if isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 400
    app.logger.exception("Unhandled error")
    return jsonify({"error": "internal server error"}), 500


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/ingest")
def ingest():
    """
    POST /ingest
    Body: {"text": "...", "doc_id": "...", "metadata": {...}, "strategy": "fixed"}
    Response 201: {"doc_id": "...", "chunks_stored": N}
    """
    body = request.get_json(force=True, silent=True) or {}
    text   = body.get("text",   "").strip()
    doc_id = body.get("doc_id", "").strip()
    if not text or not doc_id:
        abort(400, description="text and doc_id are required")
    result = pipeline.ingest(
        text=text,
        doc_id=doc_id,
        config=_get_config(),
        metadata=body.get("metadata"),
        strategy=body.get("strategy", "fixed"),
    )
    return jsonify(result), 201


@app.post("/query")
def query():
    """
    POST /query
    Body: {"question": "..."}
    Response 200: {"answer": "...", "sources": [...], "chunks": [...], "chunk_count": N}
    """
    body = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        abort(400, description="question is required")
    return jsonify(pipeline.query(question, _get_config()))


@app.get("/documents")
def documents():
    return jsonify({"documents": store.list_documents(_get_config())})


@app.delete("/documents/<doc_id>")
def delete_document(doc_id: str):
    n = store.delete_document(doc_id, _get_config())
    return jsonify({"doc_id": doc_id, "chunks_deleted": n})


@app.get("/stats")
def stats():
    return jsonify(store.collection_stats(_get_config()))


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)
