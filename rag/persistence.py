"""Cooperative local ownership and process-crash recovery for Chroma mutations."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

import chromadb
from chromadb.config import Settings

lock = threading.RLock()
_pid = os.getpid()
_owners = {}  # Retain both client and flock descriptor until process exit.


def checkpoint(name):
    """No-op boundary patched by subprocess crash tests; never environment-driven."""


def chunk_id(operation, doc_id, generation, index):
    return hashlib.sha256(json.dumps(
        [operation, doc_id, generation, index], ensure_ascii=True,
        separators=(',', ':'),
    ).encode()).hexdigest()


def journal_path(root, collection):
    key = hashlib.sha256(collection.encode()).hexdigest()
    return Path(root) / '.rag-journal' / (key + '.json')


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save(path, record):
    path.parent.mkdir(exist_ok=True)
    _sync_dir(path.parent.parent)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(record, handle, allow_nan=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    checkpoint('journal_temporary_' + record['state'])
    os.replace(temporary, path)
    checkpoint('journal_renamed_' + record['state'])
    _sync_dir(path.parent)


def _remove(path):
    path.unlink()
    _sync_dir(path.parent)


def _validate(record, path):
    fields = {'schema', 'state', 'operation', 'collection', 'collection_id',
              'doc_id', 'generation', 'old_ids', 'new_ids'}
    if not isinstance(record, dict) or set(record) != fields:
        raise ValueError('invalid journal fields')
    if type(record['schema']) is not int or record['schema'] != 1:
        raise ValueError('unsupported journal schema')
    if record['state'] not in ('STAGING', 'COMMITTED'):
        raise ValueError('invalid journal state')
    for key in ('operation', 'collection', 'collection_id', 'doc_id'):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError('invalid journal identity')
    if uuid.UUID(record['operation']).hex != record['operation']:
        raise ValueError('invalid operation identity')
    generation = record['generation']
    if type(generation) is not int or generation < 1:
        raise ValueError('invalid generation')
    for key in ('old_ids', 'new_ids'):
        ids = record[key]
        if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('invalid journal IDs')
    if set(record['old_ids']) & set(record['new_ids']):
        raise ValueError('overlapping journal IDs')
    expected = [chunk_id(record['operation'], record['doc_id'], generation, i)
                for i in range(len(record['new_ids']))]
    if record['new_ids'] != expected or path != journal_path(path.parent.parent, record['collection']):
        raise ValueError('journal identity mismatch')


def _records(collection, ids):
    return collection.get(ids=ids, include=['metadatas']) if ids else {'ids': [], 'metadatas': []}


def _verify(collection, record):
    if str(collection.id) != record['collection_id']:
        raise ValueError('journal collection identity mismatch')
    old = _records(collection, record['old_ids'])
    for metadata in old['metadatas']:
        metadata = metadata or {}
        generation = metadata.get('generation', 0)
        if (metadata.get('doc_id') != record['doc_id'] or
                type(generation) is not int or not 0 <= generation < record['generation']):
            raise ValueError('superseded record identity mismatch')
    present = collection.get(where={'doc_id': record['doc_id']}, include=[])['ids']
    if not set(present) <= set(record['old_ids'] + record['new_ids']):
        raise ValueError('unjournaled document records')
    if record['state'] == 'STAGING' and set(old['ids']) != set(record['old_ids']):
        raise ValueError('prior records missing during staging')
    new = _records(collection, record['new_ids'])
    expected = {i: n for n, i in enumerate(record['new_ids'])}
    for identity, metadata in zip(new['ids'], new['metadatas']):
        metadata = metadata or {}
        if (metadata.get('doc_id') != record['doc_id'] or
                type(metadata.get('generation')) is not int or
                metadata.get('generation') != record['generation'] or
                metadata.get('rag:operation') != record['operation'] or
                type(metadata.get('chunk_index')) is not int or
                metadata.get('chunk_index') != expected[identity]):
            raise ValueError('staged record identity mismatch')
    if record['state'] == 'COMMITTED' and set(new['ids']) != set(record['new_ids']):
        raise ValueError('committed records missing')


def _finish(collection, path, record):
    _verify(collection, record)  # Validate before any recovery deletion.
    ids = record['new_ids'] if record['state'] == 'STAGING' else record['old_ids']
    for identity in ids:
        collection.delete(ids=[identity])
        checkpoint('cleanup_' + record['state'])
    checkpoint('before_journal_removal')
    _remove(path)


def _recover(client, root):
    for path in sorted((Path(root) / '.rag-journal').glob('*.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            _validate(record, path)
            collection = client.get_collection(record['collection'], embedding_function=None)
            _finish(collection, path, record)
        except Exception as error:
            raise RuntimeError(f'RAG recovery blocked; preserve and inspect journal {path}') from error


def check_process():
    if os.getpid() != _pid:
        raise RuntimeError('Forked RAG owners are unsupported; start a fresh process')


def get_client(path):
    check_process()
    root = str(Path(path).expanduser().resolve())
    with lock:
        if root not in _owners:
            Path(root).mkdir(parents=True, exist_ok=True)
            handle = open(Path(root) / '.rag-owner.lock', 'a+b')
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                client = chromadb.PersistentClient(path=root, settings=Settings(anonymized_telemetry=False))
            except BlockingIOError as error:
                handle.close()
                raise RuntimeError('RAG persistence root already owned by another process; use its server or stop it') from error
            except BaseException:
                handle.close()
                raise
            _owners[root] = (client, handle)
        client = _owners[root][0]
        # Also retry a pending operation after an ordinary exception. No supported
        # reader or later writer can observe its intermediate representation.
        _recover(client, root)
        return client


def replace(collection, root, chunks, embeddings, doc_id, metadata):
    existing = collection.get(where={'doc_id': doc_id}, include=['metadatas'])
    generations = [(m or {}).get('generation', 0) for m in existing['metadatas']]
    if any(type(g) is not int or g < 0 for g in generations):
        raise ValueError('invalid existing document generation')
    generation = max(generations, default=0) + 1
    operation = uuid.uuid4().hex
    ids = [chunk_id(operation, doc_id, generation, i) for i in range(len(chunks))]
    if ids and collection.get(ids=ids, include=[])['ids']:
        raise RuntimeError('staged record identity collision')
    record = dict(schema=1, state='STAGING', operation=operation,
                  collection=collection.name, collection_id=str(collection.id),
                  doc_id=doc_id, generation=generation, old_ids=existing['ids'], new_ids=ids)
    path = journal_path(Path(root).expanduser().resolve(), collection.name)
    _validate(record, path)
    checkpoint('before_journal')
    _save(path, record)
    checkpoint('staging_durable')
    for index, identity in enumerate(ids):
        collection.add(ids=[identity], embeddings=[embeddings[index]], documents=[chunks[index]],
                       metadatas=[{**(metadata or {}), 'doc_id': doc_id, 'generation': generation,
                                   'chunk_index': index, 'rag:operation': operation}])
        checkpoint('staged_chunk')
    record['state'] = 'COMMITTED'
    _verify(collection, record)
    checkpoint('before_commit')
    _save(path, record)
    checkpoint('committed_durable')
    _finish(collection, path, record)
    return len(existing['ids'])
