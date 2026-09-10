"""Flask REST API.

Endpoints
    POST   /ingest
    POST   /query
    GET    /documents
    DELETE /documents/<path:doc_id>
    GET    /stats

Authentication
    If RAG_API_TOKEN is set in the environment, every request must include:
        Authorization: Bearer <token>
    If RAG_API_TOKEN is unset, authentication is disabled (local use only).

Config is loaded lazily on first request so that importing this module in
tests does not raise even when GEMINI_API_KEY is absent.
"""
from __future__ import annotations
import os
import hmac
import ipaddress
from werkzeug.exceptions import HTTPException
from flask import Flask, request, jsonify, abort
from dotenv import load_dotenv
from rag.config import load_config, Config
load_dotenv()
from rag import pipeline, store

app = Flask(__name__)

_config: Config | None = None
_api_token: str = os.environ.get("RAG_API_TOKEN", "")


def _get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def _check_auth() -> None:
    """Abort 401 if RAG_API_TOKEN is set and the request bearer doesn't match."""
    if not _api_token:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        abort(401, description="Authorization header missing or malformed")
    token = auth[len("Bearer "):]
    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(token, _api_token):
        abort(401, description="Invalid token")


# ── Error handling ────────────────────────────────────────────────────────────

@app.errorhandler(Exception)
def handle_error(exc: Exception):
    if isinstance(exc, HTTPException):
        response = exc.get_response()
        response.data = app.json.dumps({"error": exc.description})
        response.content_type = "application/json"
        return response
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

def _body() -> dict:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, description="JSON object body is required")
    return body


def _text(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        abort(400, description=f"{key} must be a non-empty string")
    return value.strip()


@app.post("/ingest")
def ingest():
    """
    POST /ingest
    Body: {"text": "...", "doc_id": "...", "metadata": {...}, "strategy": "fixed"}
    Response 201: {"doc_id": "...", "chunks_stored": N}
    """
    _check_auth()
    body = _body()
    text = _text(body, "text")
    doc_id = _text(body, "doc_id")
    if body.get("metadata") is not None and not isinstance(body["metadata"], dict):
        abort(400, description="metadata must be an object")
    if body.get("strategy", "fixed") not in ("fixed", "sentences"):
        abort(400, description="strategy must be fixed or sentences")
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
    _check_auth()
    question = _text(_body(), "question")
    return jsonify(pipeline.query(question, _get_config()))


@app.get("/documents")
def documents():
    _check_auth()
    return jsonify({"documents": store.list_documents(_get_config())})


@app.delete("/documents/<path:doc_id>")
def delete_document(doc_id: str):
    _check_auth()
    n = store.delete_document(doc_id, _get_config())
    return jsonify({"doc_id": doc_id, "chunks_deleted": n})


@app.get("/stats")
def stats():
    _check_auth()
    return jsonify(store.collection_stats(_get_config()))


def main():
    host = os.environ.get("RAG_HOST", "127.0.0.1")
    try:
        loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback and not _api_token:
        raise SystemExit("Set RAG_API_TOKEN before binding to a non-loopback address")
    app.run(debug=False, host=host, port=8000)


if __name__ == "__main__":
    main()
