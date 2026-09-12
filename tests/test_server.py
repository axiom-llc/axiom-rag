"""HTTP validation and authentication failures retain their proper status codes."""
from unittest.mock import Mock

import pytest

import server.app as api


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "_api_token", "test-token")
    monkeypatch.setattr(api, "_get_config", Mock(return_value=None))
    return api.app.test_client()


@pytest.mark.parametrize("path", ["/documents", "/stats"])
def test_unauthorized_request_is_401_without_store_access(client, monkeypatch, path):
    store = Mock(side_effect=AssertionError("unauthorized access"))
    monkeypatch.setattr(api.store, "list_documents", store)
    monkeypatch.setattr(api.store, "collection_stats", store)
    assert client.get(path).status_code == 401
    store.assert_not_called()


@pytest.mark.parametrize("body", [None, [], 42, "text", {}, {"question": None}, {"question": []}, {"question": " "}])
def test_invalid_query_is_400(client, monkeypatch, body):
    query = Mock(side_effect=AssertionError("invalid input reached query"))
    monkeypatch.setattr(api.pipeline, "query", query)
    assert client.post("/query", json=body, headers={"Authorization": "Bearer test-token"}).status_code == 400
    query.assert_not_called()


@pytest.mark.parametrize("body", [[], {"text": 42, "doc_id": "x"}, {"text": "x", "doc_id": []}, {"text": "x", "doc_id": "x", "metadata": []}, {"text": "x", "doc_id": "x", "strategy": []}])
def test_invalid_ingest_is_400(client, monkeypatch, body):
    ingest = Mock(side_effect=AssertionError("invalid input reached ingest"))
    monkeypatch.setattr(api.pipeline, "ingest", ingest)
    assert client.post("/ingest", json=body, headers={"Authorization": "Bearer test-token"}).status_code == 400
    ingest.assert_not_called()


def test_nested_document_id_can_be_deleted(client, monkeypatch):
    delete = Mock(return_value=2)
    monkeypatch.setattr(api.store, "delete_document", delete)
    response = client.delete("/documents/a/faq.txt", headers={"Authorization": "Bearer test-token"})
    assert response.status_code == 200
    delete.assert_called_once_with("a/faq.txt", None)


def test_http_error_status_and_headers_preserved(client):
    assert client.get("/missing").status_code == 404
    response = client.get("/query")
    assert response.status_code == 405
    assert "POST" in response.headers["Allow"]


@pytest.mark.parametrize("host,token,allowed", [("127.0.0.1", "", True), ("::1", "", True), ("0.0.0.0", "", False), ("0.0.0.0", "configured", True)])
def test_nonlocal_bind_requires_auth(monkeypatch, host, token, allowed):
    monkeypatch.setenv("RAG_HOST", host)
    monkeypatch.setattr(api, "_api_token", token)
    monkeypatch.setattr(api.store, "_get_client", Mock())
    run = Mock()
    monkeypatch.setattr(api.app, "run", run)
    if allowed:
        api.main()
        run.assert_called_once_with(debug=False, host=host, port=8000)
    else:
        with pytest.raises(SystemExit):
            api.main()
        run.assert_not_called()


def test_provider_error_does_not_leak_body_or_credentials(monkeypatch, caplog):
    from google.genai.errors import ClientError
    import server.app as api
    monkeypatch.setattr(api, '_api_token', '')
    def fail(*args, **kwargs):
        raise ClientError(400, {'error': {'message': 'secret-key private-document', 'status': 'INVALID_ARGUMENT'}})
    monkeypatch.setattr(api.pipeline, 'query', fail)
    response = api.app.test_client().post('/query', json={'question': 'test'})
    assert response.status_code == 502
    assert response.get_json() == {'error': 'upstream API error'}
    assert 'secret-key' not in caplog.text
    assert 'private-document' not in caplog.text


def test_startup_recovery_failure_prevents_serving(monkeypatch):
    monkeypatch.setenv('RAG_HOST', '127.0.0.1')
    def blocked(config):
        raise RuntimeError('RAG recovery blocked')
    monkeypatch.setattr(api.store, '_get_client', blocked)
    run = Mock()
    monkeypatch.setattr(api.app, 'run', run)
    with pytest.raises(RuntimeError, match='recovery blocked'):
        api.main()
    run.assert_not_called()
