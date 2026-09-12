"""Versioned wire acceptance against real isolated Chroma, with mocked providers."""
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from rag import persistence, store
from rag.config import load_config
from rag.http_client import Client, RemoteError
from rag.space import EmbeddingSpace
import server.app as api


@pytest.fixture
def wire(tmp_path, monkeypatch):
    cfg = load_config(chroma_path=str(tmp_path), collection_name='transport', embedding_dimension=3, embedding_model='gemini-embedding-2', gemini_api_key='server-secret')
    monkeypatch.setattr(api, '_get_config', lambda: cfg)
    monkeypatch.setattr(api, '_api_token', 'test-token')
    monkeypatch.delenv('RAG_HTTP_NAMESPACES', raising=False)
    monkeypatch.delenv('RAG_HTTP_GENERATION_MODELS', raising=False)
    return cfg, api.app.test_client(), asdict(EmbeddingSpace.configured(cfg))


def post(wire, operation, **fields):
    cfg, http, space = wire
    return http.post('/v1/' + operation, json={'namespace': cfg.collection_name, **({} if operation == 'inspect' else {'space': space}), **fields}, headers={'Authorization': 'Bearer test-token'})


def test_startup_inspection_creation_and_legacy(wire, monkeypatch):
    cfg, http, space = wire
    monkeypatch.setenv('RAG_HOST', '127.0.0.1')
    monkeypatch.setattr(api.app, 'run', Mock())
    api.main()
    assert store._existing_collection(cfg) is None
    result = post(wire, 'inspect').get_json()
    assert result['exists'] is False and result['total_chunks'] == 0 and result['documents'] == []
    assert store._existing_collection(cfg) is None
    assert post(wire, 'create').get_json() == {'namespace': 'transport', 'created': True}
    assert post(wire, 'create').get_json()['error'] == {'code': 'namespace_exists', 'outcome': 'not_started'}
    legacy = replace(cfg, collection_name='legacy')
    col = store._get_client(cfg.chroma_path).create_collection('legacy')
    monkeypatch.setenv('RAG_HTTP_NAMESPACES', json.dumps({'transport': space, 'legacy': space}))
    result = http.post('/v1/inspect', json={'namespace': 'legacy'}, headers={'Authorization': 'Bearer test-token'})
    assert result.get_json()['stored_space'] == dict.fromkeys(space)
    assert col.metadata is None
    response = http.post('/v1/fetch', json={'namespace': 'legacy', 'space': space, 'vector': [1, 0, 0], 'settings': {'top_k': 5, 'score_threshold': 0}}, headers={'Authorization': 'Bearer test-token'})
    assert response.status_code == 409
    assert store._existing_collection(legacy).metadata is None


@pytest.mark.parametrize('identity', [' padded ', '/nested/file', ' ', 'a::0/\x00b', 'é/文件'])
def test_raw_identity_metadata_and_replacement(wire, identity, monkeypatch):
    cfg, _, _ = wire
    provider = Mock(side_effect=AssertionError('raw operation called provider'))
    monkeypatch.setattr('rag.embedder.genai.Client', provider)
    meta = {'source': 'unit', 'flag': True, 'integer': 7, 'float': .125, 'doc_id': 'spoof', 'generation': 999}
    assert post(wire, 'replace', doc_id=identity, chunks=['first', 'second'], embeddings=[[1, .25, 0], [0, 1, .5]], metadata=meta).get_json() is None
    stored = store._get_collection(cfg).get(include=['embeddings', 'metadatas', 'documents'])
    pairs = sorted(zip(stored['documents'], stored['embeddings'].tolist()))
    assert pairs == [('first', [1, .25, 0]), ('second', [0, 1, .5])]
    chunks = post(wire, 'fetch', vector=[1, .25, 0], settings={'top_k': 5, 'score_threshold': -1}).get_json()
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk['metadata']['doc_id'] == identity
        assert chunk['metadata']['generation'] == 1
        for key in ('source', 'flag', 'integer', 'float'):
            assert chunk['metadata'][key] == meta[key]
    assert post(wire, 'replace', doc_id=identity, chunks=['short'], embeddings=[[1, 0, 0]]).status_code == 200
    assert post(wire, 'inspect').get_json()['total_chunks'] == 1
    assert post(wire, 'delete', doc_id=identity).get_json() == {'doc_id': identity, 'chunks_deleted': 1}
    assert post(wire, 'delete', doc_id=identity).get_json()['chunks_deleted'] == 0
    provider.assert_not_called()


@pytest.mark.parametrize('vector', [[float('nan'), 0, 0], [float('inf'), 0, 0], [-float('inf'), 0, 0], [1, 0], [True, 0, 0], ['1', 0, 0], None])
def test_bad_vectors_preserve_document(wire, vector):
    cfg, _, _ = wire
    store.upsert(['old'], [[1, 0, 0]], 'doc', cfg)
    for op, fields in [('replace', dict(chunks=['new'], embeddings=[vector], doc_id='doc')), ('fetch', dict(vector=vector, settings={'top_k': 1, 'score_threshold': 0}))]:
        response = post(wire, op, **fields)
        assert response.status_code == 400
        assert response.get_json()['error']['outcome'] == 'not_started'
    assert store._get_collection(cfg).get()['documents'] == ['old']


@pytest.mark.parametrize('meta', [{'x': None}, {'x': []}, {'x': {}}, {'x': float('nan')}, {'x': float('inf')}, {'chroma:document': 'reserved'}, [], 'bad'])
def test_bad_metadata_fails_before_creation_or_provider(wire, meta, monkeypatch):
    provider = Mock()
    monkeypatch.setattr('rag.embedder.genai.Client', provider)
    for op, fields in [('replace', dict(chunks=['new'], embeddings=[[1, 0, 0]])), ('ingest', dict(text='new', strategy='fixed', settings={'chunk_size': 512, 'chunk_overlap': 64}))]:
        assert post(wire, op, doc_id='doc', metadata=meta, **fields).status_code == 400
    assert store._existing_collection(wire[0]) is None
    provider.assert_not_called()


@pytest.mark.parametrize('model', ['gemini-2.5-flash', 'gemini-3.5-flash-lite'])
def test_pipeline_settings_credentials_exact_text_and_empty(wire, monkeypatch, model):
    cfg, _, _ = wire
    embeddings = Mock(side_effect=lambda texts, config: [[1, 0, 0]] * len(texts))
    monkeypatch.setattr('rag.embedder.embed_texts', embeddings)
    response = post(wire, 'ingest', text=' one two three ', doc_id=' doc ', strategy='fixed', settings={'chunk_size': 2, 'chunk_overlap': 0})
    assert response.get_json() == {'doc_id': ' doc ', 'chunks_stored': 2}
    assert embeddings.call_args.args[1].gemini_api_key == 'server-secret'
    query_embed = Mock(return_value=[1, 0, 0])
    generate = Mock(return_value={'answer': 'ok', 'sources': [' doc '], 'chunk_count': 1})
    monkeypatch.setattr('rag.embedder.embed_query', query_embed)
    monkeypatch.setattr('rag.generator.generate_answer', generate)
    result = post(wire, 'query', question=' ', settings={'top_k': 1, 'score_threshold': 0.0, 'generation_model': model})
    assert result.status_code == 200 and len(result.get_json()['chunks']) == 1
    query_embed.assert_called_once()
    assert query_embed.call_args.args[0] == ' '
    assert generate.call_args.args[2].generation_model == model
    assert generate.call_args.args[2].gemini_api_key == 'server-secret'
    assert post(wire, 'ingest', text='', doc_id=' doc ', strategy='invalid', settings={'chunk_size': 2, 'chunk_overlap': 0}).status_code == 400
    embeddings.reset_mock()
    assert post(wire, 'ingest', text='', doc_id=' doc ', strategy='fixed', settings={'chunk_size': 2, 'chunk_overlap': 0}).get_json()['chunks_stored'] == 0
    embeddings.assert_not_called()
    assert post(wire, 'inspect').get_json()['documents'] == []


@pytest.mark.parametrize('fields,code,status', [
    ({'namespace': 'not-allowed'}, 'namespace_not_allowed', 403),
    ({'space': {'provider': 'google-gemini', 'model': 'other', 'dimension': 3, 'schema': 1}}, 'space_mismatch', 409),
    ({'gemini_api_key': 'caller-secret'}, 'invalid_request', 400),
    ({'chroma_path': '/tmp/other'}, 'invalid_request', 400),
    ({'settings': {'top_k': 1, 'score_threshold': 0, 'generation_model': 'not-allowed'}}, 'model_not_allowed', 403),
    ({'settings': {'top_k': True, 'score_threshold': 0, 'generation_model': 'gemini-2.5-flash'}}, 'invalid_request', 400),
])
def test_rejections(wire, fields, code, status):
    body = dict(question='q', settings={'top_k': 1, 'score_threshold': 0, 'generation_model': 'gemini-2.5-flash'})
    body.update(fields)
    response = post(wire, 'query', **body)
    assert response.status_code == status
    assert response.get_json() == {'error': {'code': code, 'outcome': 'not_started'}}
    assert store._existing_collection(wire[0]) is None


def test_auth_malformed_config_and_errors(wire, monkeypatch):
    cfg, http, _ = wire
    assert http.post('/v1/inspect', json={}).get_json()['error']['code'] == 'unauthorized'
    for body in ('{', '[]', '{"namespace":"transport","namespace":"other"}'):
        response = http.post('/v1/inspect', data=body, content_type='application/json', headers={'Authorization': 'Bearer test-token'})
        assert response.status_code == 400
    assert http.get('/v1/query').status_code == 405
    monkeypatch.setenv('RAG_HTTP_NAMESPACES', '{}')
    assert post(wire, 'inspect').get_json()['error']['code'] == 'server_unconfigured'
    monkeypatch.delenv('RAG_HTTP_NAMESPACES')
    monkeypatch.setattr(api.store, '_existing_collection', Mock(side_effect=RuntimeError('private path')))
    response = post(wire, 'inspect')
    assert response.status_code == 503 and b'private path' not in response.data


@pytest.mark.parametrize('boundary,expected', [('staged_chunk', 'old'), ('committed_durable', 'new')])
def test_http_failure_recovery_outcome(wire, monkeypatch, boundary, expected):
    cfg, _, _ = wire
    store.upsert(['old'], [[1, 0, 0]], 'doc', cfg)
    def fail(name):
        if name == boundary:
            raise RuntimeError('private interruption')
    monkeypatch.setattr(persistence, 'checkpoint', fail)
    result = post(wire, 'replace', doc_id='doc', chunks=['new'], embeddings=[[0, 1, 0]])
    assert result.status_code == 503 and result.get_json()['error']['outcome'] == 'unknown'
    monkeypatch.setattr(persistence, 'checkpoint', lambda _: None)
    for _ in range(2):
        chunks = post(wire, 'fetch', vector=[1, 1, 0], settings={'top_k': 2, 'score_threshold': -1}).get_json()
        assert [c['text'] for c in chunks] == [expected]


class Bridge:
    def __init__(self, http):
        self.http, self.calls = http, []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        result = self.http.post(request.full_url.split('http://test')[1], data=request.data, headers=dict(request.header_items()))
        response = io.BytesIO(result.data)
        response.code = result.status_code
        return response


def test_client_full_surface(wire, monkeypatch):
    cfg, http, space = wire
    client = Client('http://test', cfg.collection_name, space, token='test-token')
    bridge = Bridge(http)
    client._opener = bridge
    assert client.inspect()['exists'] is False
    assert client.create()['created']
    with pytest.raises(RemoteError) as error:
        client.create()
    assert error.value.code == 'namespace_exists' and error.value.outcome == 'not_started'
    assert client.replace(['one'], [[1, 0, 0]], ' /doc ', {'x': 1}) is None
    assert client.fetch([1, 0, 0], top_k=1, score_threshold=0)[0]['metadata']['doc_id'] == ' /doc '
    monkeypatch.setattr('rag.embedder.embed_texts', lambda texts, cfg: [[1, 0, 0]] * len(texts))
    monkeypatch.setattr('rag.embedder.embed_query', lambda q, cfg: [1, 0, 0])
    monkeypatch.setattr('rag.generator.generate_answer', lambda q, chunks, cfg: {'answer': 'a', 'sources': [' /doc '], 'chunk_count': len(chunks)})
    assert client.ingest('text', ' /doc ', chunk_size=2, chunk_overlap=0)['chunks_stored'] == 1
    assert client.query('', top_k=1, score_threshold=0, generation_model='gemini-3.5-flash-lite')['answer'] == 'a'
    assert client.delete(' /doc ')['chunks_deleted'] == 1
    assert all(t == 60 for _, t in bridge.calls)
    assert all(b'server-secret' not in r.data and b'chroma_path' not in r.data for r, _ in bridge.calls)


@pytest.mark.parametrize('payload,status', [(b'{', 200), (b'{}', 200), (b'null', 500), (b'[]', 200)])
def test_client_protocol_failures_never_retry(wire, payload, status):
    client = Client('http://test', 'transport', wire[2], max_response_bytes=10)
    response = io.BytesIO(payload)
    response.code = status
    client._opener = Mock()
    client._opener.open.return_value = response
    with pytest.raises(RemoteError):
        client.delete('doc')
    client._opener.open.assert_called_once()


def test_client_transport_timeout_size_nan_no_fallback(wire):
    client = Client('http://test', 'transport', wire[2], max_response_bytes=3)
    client._opener = Mock()
    client._opener.open.side_effect = TimeoutError('secret')
    with pytest.raises(RemoteError, match='unknown'):
        client.replace([], [], 'doc')
    client._opener.open.assert_called_once()
    client._opener.open.reset_mock()
    with pytest.raises(ValueError):
        client.replace(['bad'], [[float('nan'), 0, 0]], 'doc')
    client._opener.open.assert_not_called()
    client._opener.open.side_effect = None
    response = io.BytesIO(b'12345')
    response.code = 200
    client._opener.open.return_value = response
    with pytest.raises(RemoteError) as error:
        client.inspect()
    assert error.value.code == 'response_too_large'


def test_client_import_does_not_load_store():
    code = "from rag.http_client import Client; import sys; assert 'chromadb' not in sys.modules; assert 'rag.store' not in sys.modules"
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


SERVER = r'''
import os, sys
from dataclasses import asdict
from werkzeug.serving import make_server
from rag import store, persistence
from rag.config import load_config
import server.app as api
root, boundary = sys.argv[1:]
cfg = load_config(chroma_path=root, collection_name='transport', embedding_dimension=3, embedding_model='gemini-embedding-2')
api._config = cfg
api._api_token = 'test-token'
store._get_client(root)
if store._existing_collection(cfg) is None:
    store.upsert(['old'], [[1., 0., 0.]], ' /doc ', cfg)
def crash(name):
    if name == boundary:
        os._exit(73)
persistence.checkpoint = crash
server = make_server('127.0.0.1', 0, api.app)
print(server.server_port, flush=True)
server.serve_forever()
'''


@pytest.mark.parametrize('boundary,expected', [('staged_chunk', 'old'), ('committed_durable', 'new')])
def test_separate_owner_client_crash_recovery(tmp_path, boundary, expected, monkeypatch):
    import select
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    env.pop('RAG_HTTP_NAMESPACES', None)
    env.pop('RAG_HTTP_GENERATION_MODELS', None)
    def start(point):
        process = subprocess.Popen([sys.executable, '-c', SERVER, str(tmp_path), point], env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert select.select([process.stdout], [], [], 30)[0], 'server startup timed out'
            line = process.stdout.readline().strip()
            assert line.isdigit(), process.stderr.read() if process.poll() is not None else line
            return process, Client('http://127.0.0.1:' + line, 'transport',
                                   dict(provider='google-gemini', model='gemini-embedding-2', dimension=3, schema=1),
                                   token='test-token', timeout=5)
        except BaseException:
            process.kill()
            process.communicate(timeout=10)
            raise
    process, client = start(boundary)
    # Any client attempt to use the parent process's store is a test failure.
    monkeypatch.setattr(store, '_get_client', Mock(side_effect=AssertionError('client opened Chroma')))
    try:
        assert client.inspect()['documents'] == [' /doc ']
        probe = subprocess.run([sys.executable, '-c',
                                'from rag import store; import sys; store._get_client(sys.argv[1])', str(tmp_path)],
                               env=env, capture_output=True, text=True, timeout=30)
        assert probe.returncode != 0 and 'already owned' in probe.stderr
        with pytest.raises(RemoteError) as error:
            client.replace(['new'], [[0, 1, 0]], ' /doc ')
        assert error.value.outcome == 'unknown'
        assert process.wait(timeout=10) == 73
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)
    for _ in range(2):
        process, client = start('')
        try:
            chunks = client.fetch([1, 1, 0], top_k=5, score_threshold=-1)
            assert [c['text'] for c in chunks] == [expected]
            assert chunks[0]['metadata']['doc_id'] == ' /doc '
        finally:
            process.kill()
            process.communicate(timeout=10)
    store._get_client.assert_not_called()


def test_provider_failure_contained_and_no_context(wire, monkeypatch):
    from google.genai.errors import ClientError
    monkeypatch.setattr('rag.embedder.embed_query', lambda q, cfg: [1, 0, 0])
    provider = Mock(side_effect=AssertionError('no-context generation'))
    monkeypatch.setattr('rag.generator.genai.Client', provider)
    fields = dict(question='', settings=dict(top_k=5, score_threshold=0, generation_model='gemini-2.5-flash'))
    assert post(wire, 'query', **fields).get_json()['chunk_count'] == 0
    provider.assert_not_called()
    store.upsert(['text'], [[1, 0, 0]], 'doc', wire[0])
    provider.side_effect = ClientError(400, {'error': {'message': 'server-secret document'}})
    response = post(wire, 'query', **fields)
    assert response.status_code == 502
    assert response.get_json()['error'] == {'code': 'provider_error', 'outcome': 'unknown'}
    assert b'server-secret' not in response.data


def test_redirect_is_not_followed(wire):
    from rag.http_client import _NoRedirect
    from urllib.request import Request
    handler = _NoRedirect()
    request = Request('http://test/v1/replace', data=b'{}', headers={'Authorization': 'Bearer secret'})
    assert handler.redirect_request(request, None, 307, '', {}, 'http://other') is None


@pytest.mark.parametrize('op', ['inspect', 'create', 'replace', 'ingest', 'fetch', 'query', 'delete'])
def test_all_operations_authenticate_before_access(wire, monkeypatch, op):
    monkeypatch.setattr(api, '_get_config', Mock(side_effect=AssertionError('unauthorized configuration access')))
    result = wire[1].post('/v1/' + op, json={})
    assert result.status_code == 401
    assert result.get_json()['error'] == {'code': 'unauthorized', 'outcome': 'not_started'}


def test_transport_does_not_round_or_normalize_vectors(wire, monkeypatch):
    values = [0.1234567890123456, -0.8765432109876543, 1.1234567890123457]
    upsert = Mock()
    fetch = Mock(return_value=[])
    monkeypatch.setattr(store, 'upsert', upsert)
    monkeypatch.setattr(store, 'store_query', fetch)
    assert post(wire, 'replace', chunks=['text'], embeddings=[values], doc_id='doc').status_code == 200
    assert upsert.call_args.args[1] == [values]
    assert post(wire, 'fetch', vector=values, settings={'top_k': 1, 'score_threshold': -1}).get_json() == []
    assert fetch.call_args.args[0] == values


def test_client_request_size_rejects_before_send(wire):
    client = Client('http://test', 'transport', wire[2], max_request_bytes=1)
    client._opener = Mock()
    with pytest.raises(ValueError, match='byte limit'):
        client.create()
    client._opener.open.assert_not_called()


def test_missing_server_credentials_preserves_raw_and_empty_operations(wire, monkeypatch):
    cfg, _, _ = wire
    monkeypatch.setattr(api, '_get_config', lambda: replace(cfg, gemini_api_key=''))
    query = dict(question='q', settings=dict(top_k=1, score_threshold=0, generation_model='gemini-2.5-flash'))
    ingest = dict(text='text', doc_id='doc', settings=dict(chunk_size=2, chunk_overlap=0), strategy='fixed')
    for op, fields in [('query', query), ('ingest', ingest)]:
        result = post(wire, op, **fields)
        assert result.status_code == 503
        assert result.get_json()['error'] == {'code': 'server_unconfigured', 'outcome': 'not_started'}
    assert store._existing_collection(cfg) is None
    assert post(wire, 'replace', chunks=['text'], embeddings=[[1, 0, 0]], doc_id='doc').status_code == 200
    ingest['text'] = ''
    assert post(wire, 'ingest', **ingest).get_json()['chunks_stored'] == 0


def test_explicit_additional_namespace_and_model_configuration(wire, monkeypatch):
    cfg, http, space = wire
    other_space = {**space, 'dimension': 2}
    monkeypatch.setenv('RAG_HTTP_NAMESPACES', json.dumps({'transport': space, 'additional': other_space}))
    other = Client('http://test', 'additional', other_space, token='test-token')
    other._opener = Bridge(http)
    other.create()
    other.replace(['text'], [[1, 0]], 'doc')
    assert other.fetch([1, 0], top_k=1, score_threshold=-1)[0]['text'] == 'text'
    assert store._existing_collection(cfg) is None
    monkeypatch.setenv('RAG_HTTP_GENERATION_MODELS', json.dumps(['gemini-2.5-flash']))
    result = post(wire, 'query', question='', settings={'top_k': 1, 'score_threshold': 0, 'generation_model': 'gemini-3.5-flash-lite'})
    assert result.get_json()['error']['code'] == 'model_not_allowed'
