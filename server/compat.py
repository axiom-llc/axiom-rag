"""Versioned trusted-application transport; all storage goes through rag.store."""
from dataclasses import asdict, replace
from functools import wraps
import json
import math
import os

from flask import Blueprint, g, request, Response
from werkzeug.exceptions import HTTPException
from google.genai.errors import APIError
from chromadb.api.types import validate_metadata

from rag import pipeline, store
from rag.space import EmbeddingSpace


class Rejected(ValueError):
    def __init__(self, code='invalid_request', status=400):
        self.code, self.status = code, status


def object_fields(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() or value.keys() - set(required) - set(optional):
        raise Rejected()
    return value


def string(value, nonempty=False):
    if not isinstance(value, str) or (nonempty and not value):
        raise Rejected()
    return value


def number(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise Rejected()
    return value


def metadata(value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise Rejected()
    for key, item in value.items():
        if not isinstance(key, str) or type(item) not in (str, bool, int, float):
            raise Rejected()
        if type(item) in (int, float):
            number(item)
    if value:
        try:
            validate_metadata(value)
        except (ValueError, TypeError):
            raise Rejected() from None


def vector(value, dimension):
    if not isinstance(value, list) or len(value) != dimension:
        raise Rejected('invalid_vector')
    try:
        for item in value:
            number(item)
    except Rejected:
        raise Rejected('invalid_vector') from None


def parse_json(value):
    def invalid(_):
        raise Rejected()
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise Rejected()
            result[key] = item
        return result
    try:
        return json.loads(value, parse_constant=invalid, object_pairs_hook=pairs)
    except (ValueError, UnicodeError, RecursionError):
        raise Rejected() from None


def policy(base):
    """Server-local environment only; never accept these settings from requests."""
    try:
        raw = os.environ.get('RAG_HTTP_NAMESPACES')
        namespaces = parse_json(raw) if raw is not None else {
            base.collection_name: asdict(EmbeddingSpace.configured(base))}
        if not isinstance(namespaces, dict):
            raise Rejected()
        configs = {}
        for name, space in namespaces.items():
            string(name, True)
            object_fields(space, ('provider', 'model', 'dimension', 'schema'))
            cfg = replace(base, collection_name=name, embedding_model=string(space['model'], True),
                          embedding_dimension=space['dimension'])
            expected = asdict(EmbeddingSpace.configured(cfg))
            if any(type(space[k]) is not type(v) or space[k] != v for k, v in expected.items()):
                raise Rejected()
            configs[name] = cfg
        raw = os.environ.get('RAG_HTTP_GENERATION_MODELS')
        models = parse_json(raw) if raw is not None else [
            'gemini-2.5-flash', 'gemini-3.5-flash-lite', base.generation_model]
        if not isinstance(models, list) or any(not isinstance(m, str) or not m for m in models):
            raise Rejected()
        # Legacy endpoints remain restricted to the configured default namespace.
        if base.collection_name not in configs or EmbeddingSpace.configured(configs[base.collection_name]) != EmbeddingSpace.configured(base):
            raise Rejected()
        return configs, sorted(set(models))
    except (ValueError, TypeError, AttributeError):
        raise Rejected('server_unconfigured', 503) from None


def register(app, get_config, check_auth):
    api = Blueprint('compat', __name__, url_prefix='/v1')

    def reply(value, status=200):
        return Response(json.dumps(value, allow_nan=False, ensure_ascii=True),
                        status=status, mimetype='application/json')

    def operation(required, optional=()):
        def decorate(fn):
            @wraps(fn)
            def run():
                g.compat_dispatched = False
                check_auth()
                if not request.is_json:
                    raise Rejected()
                body = object_fields(parse_json(request.get_data()), ('namespace', *required), optional)
                name = string(body['namespace'], True)
                try:
                    configs, models = policy(get_config())
                except Rejected:
                    raise
                except Exception:
                    raise Rejected('server_unconfigured', 503) from None
                if name not in configs:
                    raise Rejected('namespace_not_allowed', 403)
                return reply(fn(body, configs[name], models))
            return run
        return decorate

    def agreement(body, cfg):
        expected = asdict(EmbeddingSpace.configured(cfg))
        space = object_fields(body['space'], expected)
        if any(type(space[k]) is not type(v) or space[k] != v for k, v in expected.items()):
            raise Rejected('space_mismatch', 409)

    def settings(body, cfg, models, fields):
        values = object_fields(body['settings'], fields)
        for key in ('chunk_size', 'chunk_overlap', 'top_k'):
            if key in values and type(values[key]) is not int:
                raise Rejected()
        if 'chunk_size' in values and not (values['chunk_size'] > 0 and 0 <= values['chunk_overlap'] < values['chunk_size']):
            raise Rejected()
        if 'top_k' in values and values['top_k'] <= 0:
            raise Rejected()
        if 'score_threshold' in values and not -1 <= number(values['score_threshold']) <= 1:
            raise Rejected()
        if 'generation_model' in values and string(values['generation_model'], True) not in models:
            raise Rejected('model_not_allowed', 403)
        return replace(cfg, **values)

    def preflight(body, cfg):
        agreement(body, cfg)
        # Validate existing provenance before provider work, without adoption.
        collection = store._existing_collection(cfg)
        if collection is not None:
            try:
                EmbeddingSpace.configured(cfg).validate(collection.metadata, cfg.collection_name)
            except ValueError:
                raise Rejected('space_mismatch', 409) from None

    @api.post('/inspect')
    @operation(())
    def inspect(body, cfg, models):
        # Lock spans existence + stats for a consistent supported read.
        store.persistence.check_process()
        with store._write_lock:
            existing = store._existing_collection(cfg)
            return {'namespace': cfg.collection_name, 'exists': existing is not None,
                    'space': asdict(EmbeddingSpace.configured(cfg)),
                    'stored_space': {k: (existing.metadata or {}).get('rag:embedding:' + k)
                                     for k in ('provider', 'model', 'dimension', 'schema')} if existing else None,
                    'settings': {k: getattr(cfg, k) for k in ('chunk_size', 'chunk_overlap', 'top_k', 'score_threshold', 'generation_model')},
                    'generation_models': models, **store.collection_stats(cfg)}

    @api.post('/create')
    @operation(('space',))
    def create(body, cfg, models):
        agreement(body, cfg)
        store.persistence.check_process()
        with store._write_lock:
            if store._existing_collection(cfg) is not None:
                raise Rejected('namespace_exists', 409)
            g.compat_dispatched = True
            store.create_collection(cfg)
        return {'namespace': cfg.collection_name, 'created': True}

    @api.post('/replace')
    @operation(('space', 'doc_id', 'chunks', 'embeddings'), ('metadata',))
    def replace_document(body, cfg, models):
        string(body['doc_id'], True)
        metadata(body.get('metadata'))
        chunks, vectors = body['chunks'], body['embeddings']
        if not isinstance(chunks, list) or not isinstance(vectors, list) or len(chunks) != len(vectors):
            raise Rejected()
        for chunk in chunks:
            string(chunk)
        for item in vectors:
            vector(item, cfg.embedding_dimension)
        preflight(body, cfg)
        g.compat_dispatched = True
        store.upsert(chunks, vectors, body['doc_id'], cfg, body.get('metadata'))
        return None

    @api.post('/ingest')
    @operation(('space', 'doc_id', 'text', 'settings', 'strategy'), ('metadata',))
    def ingest(body, cfg, models):
        string(body['doc_id'], True)
        string(body['text'])
        metadata(body.get('metadata'))
        if body['strategy'] not in ('fixed', 'sentences'):
            raise Rejected()
        cfg = settings(body, cfg, models, ('chunk_size', 'chunk_overlap'))
        preflight(body, cfg)
        if body['text'].strip() and not cfg.gemini_api_key:
            raise Rejected('server_unconfigured', 503)
        g.compat_dispatched = True
        return pipeline.ingest(body['text'], body['doc_id'], cfg, body.get('metadata'), body['strategy'])

    @api.post('/fetch')
    @operation(('space', 'vector', 'settings'))
    def fetch(body, cfg, models):
        vector(body['vector'], cfg.embedding_dimension)
        cfg = settings(body, cfg, models, ('top_k', 'score_threshold'))
        preflight(body, cfg)
        g.compat_dispatched = True
        return store.store_query(body['vector'], cfg)

    @api.post('/query')
    @operation(('space', 'question', 'settings'))
    def query(body, cfg, models):
        string(body['question'])
        cfg = settings(body, cfg, models, ('top_k', 'score_threshold', 'generation_model'))
        preflight(body, cfg)
        if not cfg.gemini_api_key:
            raise Rejected('server_unconfigured', 503)
        g.compat_dispatched = True
        return pipeline.pipeline_query(body['question'], cfg)

    @api.post('/delete')
    @operation(('space', 'doc_id'))
    def delete(body, cfg, models):
        string(body['doc_id'], True)
        preflight(body, cfg)
        g.compat_dispatched = True
        return {'doc_id': body['doc_id'], 'chunks_deleted': store.delete_document(body['doc_id'], cfg)}

    def error(exc):
        if isinstance(exc, Rejected):
            code, status = exc.code, exc.status
        elif isinstance(exc, HTTPException):
            code, status = {401: 'unauthorized', 404: 'not_found', 405: 'method_not_allowed', 413: 'request_too_large'}.get(exc.code, 'invalid_request'), exc.code
        elif isinstance(exc, APIError):
            code, status = 'provider_error', 502
        elif isinstance(exc, ValueError):
            code, status = 'operation_failed', 400
        else:
            code, status = 'storage_unavailable', 503
        response = reply({'error': {'code': code, 'outcome': 'unknown' if getattr(g, 'compat_dispatched', False) else 'not_started'}}, status)
        if isinstance(exc, HTTPException) and 'Allow' in exc.get_response().headers:
            response.headers['Allow'] = exc.get_response().headers['Allow']
        return response

    app.register_blueprint(api)
    return error
