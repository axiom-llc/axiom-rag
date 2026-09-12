"""Mapped adapters: no local owner, explicit settings, isolated live HTTP owner."""
import io
import json
import os
from pathlib import Path
import select
import subprocess
import sys
from dataclasses import replace
from unittest.mock import Mock

import pytest
from rag import remote
from rag.config import load_config
from rag.http_client import Client, RemoteError


@pytest.fixture
def cfg(monkeypatch):
    for key in list(os.environ):
        if key.startswith('RAG_'):
            monkeypatch.delenv(key)
    monkeypatch.setenv('RAG_BASE_URL', 'http://127.0.0.1:8000')
    return load_config(gemini_api_key='caller-secret-never-forward')


@pytest.mark.parametrize('override', [dict(chroma_path='/unmapped'), dict(collection_name='documents'),
    dict(embedding_dimension=3), dict(embedding_model='other')])
def test_unmapped_config_never_dispatches(cfg, override, monkeypatch):
    send = Mock(side_effect=AssertionError('dispatched'))
    monkeypatch.setattr(Client, '_call', send)
    with pytest.raises(ValueError, match='not mapped'):
        remote.list_documents(replace(cfg, **override))
    send.assert_not_called()


def test_target_required(cfg, monkeypatch):
    monkeypatch.delenv('RAG_BASE_URL')
    with pytest.raises(ValueError, match='RAG_BASE_URL'):
        remote.list_documents(cfg)


def test_settings_and_failure_are_single_attempt(cfg, monkeypatch):
    from urllib.request import OpenerDirector
    calls = []
    def send(self, request, timeout):
        calls.append(request)
        raise TimeoutError('private transport information')
    monkeypatch.setattr(OpenerDirector, 'open', send)
    cfg = replace(cfg, chunk_size=17, chunk_overlap=2, top_k=7, score_threshold=-1)
    operations = [lambda: remote.ingest(' text ', ' /doc ', cfg, {'x': True}, 'sentences'),
                  lambda: remote.query(' q ', cfg), lambda: remote.store_query([0.] * 3072, cfg)]
    for operation in operations:
        with pytest.raises(RemoteError, match='unknown') as error:
            operation()
        assert 'private' not in str(error.value)
    assert len(calls) == 3
    bodies = [json.loads(r.data) for r in calls]
    assert bodies[0]['doc_id'] == ' /doc ' and bodies[0]['text'] == ' text '
    assert bodies[0]['settings'] == {'chunk_size': 17, 'chunk_overlap': 2}
    assert bodies[0]['strategy'] == 'sentences'
    assert bodies[1]['settings'] == {'top_k': 7, 'score_threshold': -1, 'generation_model': 'gemini-2.5-flash'}
    assert bodies[2]['settings'] == {'top_k': 7, 'score_threshold': -1}
    assert all(b'caller-secret' not in r.data and b'chroma_path' not in r.data for r in calls)


SERVER = r'''
import sys
from werkzeug.serving import make_server
from rag.config import load_config
from rag import store, embedder, generator
import server.app as api
api._config = load_config(gemini_api_key='server-test-only')
api._api_token = ''
store._get_client(api._config.chroma_path)
embedder.embed_texts = lambda texts, cfg: [[1.] + [0.] * 3071 for _ in texts]
embedder.embed_query = lambda text, cfg: [1.] + [0.] * 3071
def generate(q, chunks, cfg):
    return {'answer': cfg.generation_model, 'sources': sorted({c['metadata']['doc_id'] for c in chunks}), 'chunk_count': len(chunks)}
generator.generate_answer = generate
server = make_server('127.0.0.1', 0, api.app)
print(server.server_port, flush=True)
server.serve_forever()
'''

# Executed in a fresh caller process: even importing Chroma or local owners fails.
CALLER = r'''
import sys, io, contextlib, json, importlib.abc
from pathlib import Path
class NoOwner(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname in ('rag.store', 'rag.persistence', 'rag.pipeline') or fullname.startswith('chromadb'):
            raise AssertionError('caller imported storage owner: ' + fullname)
sys.meta_path.insert(0, NoOwner())
import cli
from rag import remote
from rag.config import load_config
from rag.http_client import RemoteError
cfg=load_config()
assert not cfg.gemini_api_key
assert remote.inspect(cfg)['exists'] is False
assert remote.collection_stats(cfg)=={'total_chunks':0,'documents':[]}
def run(*args):
    sys.argv=['rag',*args]
    out=io.StringIO()
    with contextlib.redirect_stdout(out): cli.main()
    return out.getvalue()
assert 'Created' in run('create')
try: run('create')
except RemoteError as exc: assert exc.code=='namespace_exists'
else: raise AssertionError('create adopted existing namespace')
root=Path(__import__('os').environ['TEST_FILES'])
root.mkdir()
(root/'single .txt').write_bytes(b'hello\xff world')
assert 'single .txt' in run('ingest',str(root/'single .txt'))
(root/'left').mkdir(); (root/'right').mkdir()
(root/'left'/'same.md').write_text('left text')
(root/'right'/'same.md').write_text('right text')
out=run('ingest',str(root))
assert out.index('left/same.md') < out.index('right/same.md') < out.index('single .txt')
assert 'Ingested 3 document(s).' in out
assert 'left/same.md' in run('list')
assert json.loads(run('stats'))['total_chunks']==3
assert 'gemini-2.5-flash' in run('query','question')
assert "Deleted 1 chunk(s)" in run('delete','left/same.md')
id=' /exact/nested id '
v=[1.]+[0.]*3071
assert remote.upsert(['raw'],[v],id,cfg,{'scalar':True}) is None
chunks=remote.store_query(v,cfg)
assert any(c['text']=='raw' and c['metadata']['doc_id']==id and c['score']>0.999 for c in chunks)
assert remote.ingest('',id,cfg)['chunks_stored']==0
assert id not in remote.list_documents(cfg)
assert 'rag.store' not in sys.modules and 'chromadb' not in sys.modules
print('HTTP-only CLI and raw adapter assertions passed')
'''


def test_separate_owner_cli(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('RAG_') and k != 'GEMINI_API_KEY'}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(Path(__file__).resolve().parents[1]),
               TEST_FILES=str(tmp_path / 'files'), PYTHON_DOTENV_DISABLED='1')
    process = subprocess.Popen([sys.executable, '-c', SERVER], env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 30)[0], 'owner startup timed out'
        port = process.stdout.readline().strip()
        assert port.isdigit(), process.stderr.read() if process.poll() is not None else 'owner failed startup'
        env['RAG_BASE_URL'] = 'http://127.0.0.1:' + port
        result = subprocess.run([sys.executable, '-c', CALLER], env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert 'assertions passed' in result.stdout
    finally:
        process.kill()
        process.communicate(timeout=10)
