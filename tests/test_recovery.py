"""Real Chroma 1.5.2 subprocess termination tests; no provider or live store."""
import json
import os
from pathlib import Path
import subprocess
import select
import sys
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from rag import persistence, store
from rag.config import load_config

DRIVER = r'''
import json, os, sys
from rag import store, persistence
from rag.config import load_config
root, mode, boundary = sys.argv[1:]
cfg = load_config(chroma_path=root, collection_name='crash_test', embedding_dimension=3)
def crash(name):
    if name == boundary:
        os._exit(73)
if mode in ('replace', 'delete', 'empty'):
    col = store._get_collection(cfg)
    col.add(ids=['legacy0', 'legacy1', 'legacy2'], documents=['old0', 'old1', 'old2'],
            embeddings=[[1., 0., 0.]] * 3,
            metadatas=[{'doc_id': 'doc', 'chunk_index': i} for i in range(3)])
    persistence.checkpoint = crash
    if mode == 'delete':
        store.delete_document('doc', cfg)
    elif mode == 'empty':
        store.upsert([], [], 'doc', cfg)
    else:
        store.upsert(['new0', 'new1'], [[0., 1., 0.]] * 2, 'doc', cfg)
elif mode == 'owner':
    store._get_collection(cfg)
    print('ready', flush=True)
    sys.stdin.read()
elif mode == 'read':
    persistence.checkpoint = crash
    # Inspection, not only mutations, must recover first.
    stats = store.collection_stats(cfg)
    col = store._get_collection(cfg)
    result = col.get(include=['documents', 'metadatas', 'embeddings'])
    pairs = sorted(zip(result['documents'], result['embeddings'].tolist(), result['metadatas']), key=lambda x: x[0])
    print(json.dumps({'stats': stats, 'records': pairs,
                      'retrieved': sorted(r['text'] for r in store.store_query([1., 1., 0.], cfg))}))
'''


def run(root, mode='read', boundary='', code=0):
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, '-c', DRIVER, str(root), mode, boundary],
                            capture_output=True, text=True, timeout=45, env=env)
    assert result.returncode == code, result.stdout + result.stderr
    return result


@pytest.mark.parametrize('boundary,committed', [
    ('before_journal', False),
    ('journal_temporary_STAGING', False),
    ('journal_renamed_STAGING', False),
    ('staging_durable', False),
    ('staged_chunk', False),
    ('before_commit', False),
    ('journal_temporary_COMMITTED', False),
    ('journal_renamed_COMMITTED', True),
    ('committed_durable', True),
    ('cleanup_COMMITTED', True),
    ('before_journal_removal', True),
])
def test_replacement_process_crash_recovers_exact_generation(tmp_path, boundary, committed):
    run(tmp_path, 'replace', boundary, 73)
    for _ in range(2):
        result = json.loads(run(tmp_path).stdout)
        expected = ['new0', 'new1'] if committed else ['old0', 'old1', 'old2']
        assert [r[0] for r in result['records']] == expected
        assert result['retrieved'] == expected
        assert [r[1] for r in result['records']] == ([[0., 1., 0.]] * 2 if committed else [[1., 0., 0.]] * 3)
        assert result['stats'] == {'total_chunks': len(expected), 'documents': ['doc']}
        if committed:
            assert all(r[2]['generation'] == 1 for r in result['records'])
    assert not list((tmp_path / '.rag-journal').glob('*.json'))


@pytest.mark.parametrize('state,boundary', [('STAGING', 'staged_chunk'), ('COMMITTED', 'committed_durable')])
def test_recovery_itself_can_be_interrupted(tmp_path, state, boundary):
    run(tmp_path, 'replace', boundary, 73)
    run(tmp_path, 'read', 'cleanup_' + state, 73)
    result = json.loads(run(tmp_path).stdout)
    assert [r[0] for r in result['records']] == (['new0', 'new1'] if state == 'COMMITTED' else ['old0', 'old1', 'old2'])
    assert json.loads(run(tmp_path).stdout) == result


@pytest.mark.parametrize('mode', ['delete', 'empty'])
@pytest.mark.parametrize('boundary,deleted', [('staging_durable', False), ('committed_durable', True), ('cleanup_COMMITTED', True)])
def test_delete_and_empty_replacement_crashes(tmp_path, mode, boundary, deleted):
    run(tmp_path, mode, boundary, 73)
    result = json.loads(run(tmp_path).stdout)
    assert [r[0] for r in result['records']] == ([] if deleted else ['old0', 'old1', 'old2'])
    assert json.loads(run(tmp_path).stdout) == result


@pytest.mark.parametrize('corruption', ['json', 'overlap', 'collection', 'missing_target', 'omitted_old'])
def test_ambiguous_journal_blocks_without_cleanup(tmp_path, corruption):
    run(tmp_path, 'replace', 'staged_chunk', 73)
    path = next((tmp_path / '.rag-journal').glob('*.json'))
    record = json.loads(path.read_text())
    if corruption == 'json':
        path.write_text('{')
    else:
        if corruption == 'overlap':
            record['old_ids'].append(record['new_ids'][0])
        elif corruption == 'collection':
            record['collection_id'] = 'wrong-collection'
        elif corruption == 'omitted_old':
            record['old_ids'].pop()
        else:
            record['state'] = 'COMMITTED'  # Only one of two target records exists.
        path.write_text(json.dumps(record))
    before = path.read_bytes()
    for _ in range(2):
        assert 'RAG recovery blocked' in run(tmp_path, code=1).stderr
        assert path.read_bytes() == before


def test_second_owner_excluded_and_sigkill_releases_lock(tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    owner = subprocess.Popen([sys.executable, '-c', DRIVER, str(tmp_path), 'owner', ''],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env)
    try:
        assert select.select([owner.stdout], [], [], 30)[0], 'owner startup timed out'
        assert owner.stdout.readline().strip() == 'ready'
        alias = tmp_path / 'alias'
        alias.symlink_to(tmp_path, target_is_directory=True)
        assert 'already owned' in run(alias, code=1).stderr
    finally:
        owner.kill()
        owner.communicate(timeout=15)
    assert json.loads(run(tmp_path).stdout)['records'] == []


def test_supported_reads_wait_for_complete_replacement(tmp_path, monkeypatch):
    cfg = load_config(chroma_path=str(tmp_path), collection_name='threads', embedding_dimension=3)
    store.upsert(['old'], [[1., 0., 0.]], 'doc', cfg)
    staged, release, reading, finished = (threading.Event() for _ in range(4))
    def pause(name):
        if name == 'staged_chunk':
            staged.set()
            assert release.wait(10)
    monkeypatch.setattr(persistence, 'checkpoint', pause)
    def read():
        reading.set()
        result = store.collection_stats(cfg)
        finished.set()
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(store.upsert, ['new0', 'new1'], [[0., 1., 0.]] * 2, 'doc', cfg)
        assert staged.wait(10)
        reader = pool.submit(read)
        try:
            assert reading.wait(10)
            assert not finished.wait(0.1)
        finally:
            release.set()
        writer.result()
        assert reader.result() == {'total_chunks': 2, 'documents': ['doc']}


def test_long_lived_client_and_opaque_identity(tmp_path):
    cfg = load_config(chroma_path=str(tmp_path), collection_name='identity', embedding_dimension=3)
    first = store._get_client(str(tmp_path))
    store._get_client.cache_clear()
    assert store._get_client(str(tmp_path / '.')) is first
    for _ in range(2):
        store.upsert(['text'], [[1., 0., 0.]], 'a::0/\u0000b', cfg, {'generation': 999, 'rag:operation': 'wrong'})
    result = store._get_collection(cfg).get()
    assert len(result['ids'][0]) == 64
    assert result['metadatas'][0]['generation'] == 2
    assert result['metadatas'][0]['rag:operation'] != 'wrong'


def test_exception_recovers_before_next_supported_read(tmp_path, monkeypatch):
    cfg = load_config(chroma_path=str(tmp_path), collection_name='exceptions', embedding_dimension=3)
    store.upsert(['old'], [[1., 0., 0.]], 'doc', cfg)
    def fail(name):
        if name == 'staged_chunk':
            raise RuntimeError('interrupted request')
    monkeypatch.setattr(persistence, 'checkpoint', fail)
    with pytest.raises(RuntimeError, match='interrupted request'):
        store.upsert(['new'], [[0., 1., 0.]], 'doc', cfg)
    monkeypatch.setattr(persistence, 'checkpoint', lambda name: None)
    assert store.collection_stats(cfg) == {'total_chunks': 1, 'documents': ['doc']}
    assert store.store_query([1., 0., 0.], cfg)[0]['text'] == 'old'


def test_forked_owner_fails_before_inherited_lock(tmp_path):
    code = r'''
import os, sys
from rag import store, persistence
from rag.config import load_config
cfg = load_config(chroma_path=sys.argv[1], collection_name='fork_test', embedding_dimension=3)
store._get_collection(cfg)
with persistence.lock:
    pid = os.fork()
    if pid == 0:
        try:
            store.collection_stats(cfg)
        except RuntimeError as error:
            os._exit(0 if 'Forked RAG owners' in str(error) else 2)
        os._exit(3)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
'''
    env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1])}
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], env=env,
                            capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
