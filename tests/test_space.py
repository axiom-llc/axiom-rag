"""Real synthetic Chroma stores; no provider or legacy database access."""
from dataclasses import replace
from unittest.mock import patch

import pytest
from rag import store, pipeline, embedder
from rag.config import load_config
from rag.space import EmbeddingSpace


@pytest.fixture
def cfg(tmp_path):
    return load_config(chroma_path=str(tmp_path / 'chroma'), collection_name='fresh',
                       embedding_model='gemini-embedding-2', embedding_dimension=3,
                       gemini_api_key='synthetic')


def test_creation_and_normalized_reopen(cfg):
    col = store.create_collection(cfg)
    assert col.metadata == {'hnsw:space': 'cosine', 'rag:embedding:provider': 'google-gemini',
                            'rag:embedding:model': 'gemini-embedding-2',
                            'rag:embedding:dimension': 3, 'rag:embedding:schema': 1}
    assert store._get_collection(replace(cfg, embedding_model='models/gemini-embedding-2')).id == col.id
    with pytest.raises(Exception):
        store.create_collection(cfg)
    assert col.count() == 0


@pytest.mark.parametrize('field,value', [('embedding_model', 'another-model'), ('embedding_dimension', 2)])
def test_config_mismatch_preserves_data(cfg, field, value):
    store.upsert(['old'], [[1., 0., 0.]], 'doc', cfg)
    col = store._get_collection(cfg)
    metadata = dict(col.metadata)
    before = col.get(include=['documents', 'embeddings', 'metadatas'])
    other = replace(cfg, **{field: value})
    for operation in [lambda: store.upsert(['new'], [[1.] * other.embedding_dimension], 'doc', other),
                      lambda: store.query([1.] * other.embedding_dimension, other),
                      lambda: store.delete_document('doc', other)]:
        with pytest.raises(ValueError, match='incompatible'):
            operation()
    after = col.get(include=['documents', 'embeddings', 'metadatas'])
    assert before['documents'] == after['documents']
    assert (before['embeddings'] == after['embeddings']).all()
    assert store._existing_collection(cfg).metadata == metadata


@pytest.mark.parametrize('populated', [False, True])
def test_unknown_never_adopted_even_when_empty(cfg, populated):
    col = store._get_client(cfg.chroma_path).create_collection(cfg.collection_name, metadata={'hnsw:space': 'cosine'})
    if populated:
        col.add(ids=['legacy'], documents=['old'], embeddings=[[1., 0., 0.]], metadatas=[{'doc_id': 'old'}])
    for operation in [lambda: store.upsert(['new'], [[1., 0., 0.]], 'new', cfg),
                      lambda: store.query([1., 0., 0.], cfg),
                      lambda: pipeline.ingest('', 'old', cfg),
                      lambda: pipeline.ingest('new', 'old', cfg),
                      lambda: pipeline.query('question', cfg)]:
        with patch('rag.embedder.genai.Client') as client:
            with pytest.raises(ValueError, match='unknown embedding provenance'):
                operation()
            client.assert_not_called()
    assert store.collection_stats(cfg)['total_chunks'] == int(populated)
    assert col.get()['documents'] == (['old'] if populated else [])
    assert store._existing_collection(cfg).metadata == {'hnsw:space': 'cosine'}
    fresh = replace(cfg, collection_name='separate')
    store.upsert(['new'], [[0., 1., 0.]], 'new', fresh)
    assert store.list_documents(fresh) == ['new']
    assert col.count() == int(populated)


@pytest.mark.parametrize('key,value', [('provider', 'other'), ('schema', 2), ('dimension', '3')])
def test_invalid_metadata_fails_closed(cfg, key, value):
    metadata = EmbeddingSpace.configured(cfg).metadata()
    metadata['rag:embedding:' + key] = value
    store._get_client(cfg.chroma_path).create_collection(cfg.collection_name, metadata=metadata)
    with pytest.raises(ValueError, match='incompatible'):
        store._get_collection(cfg)


def test_vector_dimension_rejected_before_creation(cfg):
    with pytest.raises(ValueError, match='dimension'):
        store.upsert(['bad'], [[1.]], 'bad', cfg)
    assert store._existing_collection(cfg) is None


@pytest.mark.parametrize('model', ['gemini-embedding-2', 'other-model'])
def test_adapter_requests_and_checks_dimension(cfg, model):
    from types import SimpleNamespace
    cfg = replace(cfg, embedding_model=model)
    with patch('rag.embedder.genai.Client') as client:
        call = client.return_value.models.embed_content
        call.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1., 0., 0.])])
        assert embedder.embed_texts(['text'], cfg) == [[1., 0., 0.]]
        assert call.call_args.kwargs['config'].output_dimensionality == 3
        assert embedder.embed_query('query', cfg) == [1., 0., 0.]
        assert call.call_args.kwargs['config'].output_dimensionality == 3
        call.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1.])])
        with pytest.raises(ValueError, match='dimension'):
            embedder.embed_query('query', cfg)


def test_inspection_does_not_create_collection(cfg):
    assert store.collection_stats(cfg) == {'total_chunks': 0, 'documents': []}
    assert store._existing_collection(cfg) is None


def test_server_rejects_legacy_embedding_operations(cfg, monkeypatch):
    import server.app as api
    col = store._get_client(cfg.chroma_path).create_collection(cfg.collection_name)
    col.add(ids=['legacy'], documents=['old'], embeddings=[[1., 0., 0.]])
    monkeypatch.setattr(api, '_get_config', lambda: cfg)
    monkeypatch.setattr(api, '_api_token', 'test-token')
    client = api.app.test_client()
    headers = {'Authorization': 'Bearer test-token'}
    with patch('rag.embedder.genai.Client') as provider:
        for path, body in [('/query', {'question': 'question'}), ('/ingest', {'text': 'new', 'doc_id': 'legacy'})]:
            response = client.post(path, json=body, headers=headers)
            assert response.status_code == 400
            assert 'unknown embedding provenance' in response.get_json()['error']
        provider.assert_not_called()
    assert col.get()['documents'] == ['old']


def test_default_namespace_avoids_legacy_collection(monkeypatch, tmp_path):
    from rag.config import load_config
    for name in ['RAG_COLLECTION', 'RAG_EMBEDDING_MODEL', 'RAG_EMBEDDING_DIMENSION']:
        monkeypatch.delenv(name, raising=False)
    config = load_config(chroma_path=str(tmp_path))
    legacy = store._get_client(str(tmp_path)).create_collection('documents')
    legacy.add(ids=['old'], embeddings=[[1., 0., 0.]], documents=['legacy'])
    assert config.embedding_model == 'gemini-embedding-2'
    assert config.collection_name == 'documents-gemini-embedding-2'
    store.create_collection(config)
    assert legacy.get()['documents'] == ['legacy']
    assert legacy.metadata is None
    monkeypatch.setenv('RAG_COLLECTION', 'documents')
    with pytest.raises(ValueError, match='unknown embedding provenance'):
        store._get_collection(load_config(chroma_path=str(tmp_path)))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_replacement_preserves_all_chunks(cfg, value):
    store.upsert(['old one', 'old two', 'old three'], [[1., 0., 0.]] * 3, 'doc', cfg)
    col = store._get_collection(cfg)
    before = col.get(include=['documents', 'embeddings', 'metadatas'])
    with pytest.raises(ValueError, match='finite'):
        store.upsert(['new one', 'new two'], [[0., 1., 0.], [value, 0., 0.]], 'doc', cfg)
    after = col.get(include=['documents', 'embeddings', 'metadatas'])
    for key in ['ids', 'documents', 'metadatas']:
        assert after[key] == before[key]
    assert (after['embeddings'] == before['embeddings']).all()


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_vectors_rejected_before_store_access(cfg, value):
    with patch('rag.store._get_collection') as collection:
        for operation in [lambda: store.upsert(['bad'], [[value, 0., 0.]], 'doc', cfg),
                          lambda: store.query([0., value, 0.], cfg)]:
            with pytest.raises(ValueError, match='finite'):
                operation()
        collection.assert_not_called()
    assert store._existing_collection(cfg) is None


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_provider_vectors_rejected(cfg, value):
    from types import SimpleNamespace
    with patch('rag.embedder.genai.Client') as client:
        client.return_value.models.embed_content.return_value = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0., 0., value])])
        with pytest.raises(ValueError, match='finite'):
            embedder.embed_texts(['text'], cfg)
        with pytest.raises(ValueError, match='finite'):
            embedder.embed_query('question', cfg)
