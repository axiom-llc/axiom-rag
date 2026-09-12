"""HTTP-only adapters for the explicitly mapped host-local CLI/APEX deployment.

The canonical rag.store/rag.pipeline remain owner-side and evaluator interfaces.
This module never imports either, Chroma, or provider implementations.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
from pathlib import Path

from rag.config import Config
from rag.http_client import Client
from rag.space import EmbeddingSpace


def _client(config: Config) -> Client:
    target = os.environ.get('RAG_BASE_URL', '')
    if not target:
        raise ValueError('RAG_BASE_URL must explicitly select the RAG service')
    if Path(config.chroma_path).expanduser().resolve() != Path('~/.rag/chroma').expanduser().resolve():
        raise ValueError('RAG_CHROMA_PATH is not mapped to the RAG service')
    space = asdict(EmbeddingSpace.configured(config))
    if config.collection_name != 'documents-gemini-embedding-2' or space != {
        'provider': 'google-gemini', 'model': 'gemini-embedding-2',
        'dimension': 3072, 'schema': 1,
    }:
        raise ValueError('Namespace or embedding space is not mapped to the RAG service')
    return Client(target, config.collection_name, space,
                  token=os.environ.get('RAG_API_TOKEN', ''))


def inspect(config: Config) -> dict:
    return _client(config).inspect()


def create_collection(config: Config) -> dict:
    return _client(config).create()


def upsert(chunks, embeddings, doc_id, config: Config, metadata=None) -> None:
    return _client(config).replace(chunks, embeddings, doc_id, metadata)


def store_query(query_embedding, config: Config) -> list[dict]:
    return _client(config).fetch(query_embedding, top_k=config.top_k,
                                 score_threshold=config.score_threshold)


def delete_document(doc_id: str, config: Config) -> int:
    return _client(config).delete(doc_id)['chunks_deleted']


def list_documents(config: Config) -> list[str]:
    return inspect(config)['documents']


def collection_stats(config: Config) -> dict:
    result = inspect(config)
    return {key: result[key] for key in ('total_chunks', 'documents')}


def ingest(text, doc_id, config: Config, metadata=None, strategy='fixed') -> dict:
    if strategy not in ('fixed', 'sentences'):
        raise ValueError('strategy must be one of: fixed, sentences')
    return _client(config).ingest(text, doc_id, chunk_size=config.chunk_size,
                                 chunk_overlap=config.chunk_overlap,
                                 strategy=strategy, metadata=metadata)


def pipeline_query(question: str, config: Config) -> dict:
    return _client(config).query(question, top_k=config.top_k,
                                 score_threshold=config.score_threshold,
                                 generation_model=config.generation_model)


def query(target, config: Config):
    if isinstance(target, list):
        return store_query(target, config)
    return pipeline_query(target, config)


def ingest_file(path, config: Config, metadata=None, strategy='fixed', doc_id=None):
    _client(config)  # Reject unmapped configuration before reading input files.
    file_path = Path(path).expanduser()
    text = file_path.read_text(encoding='utf-8', errors='replace')
    return ingest(text, doc_id or file_path.name, config, metadata, strategy)


def ingest_directory(directory, config: Config, extensions=None, strategy='fixed', max_workers=4):
    if max_workers <= 0:
        raise ValueError('max_workers must be greater than 0')
    if strategy not in ('fixed', 'sentences'):
        raise ValueError('strategy must be one of: fixed, sentences')
    _client(config)
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise ValueError(f'directory not found: {root}')
    suffixes = set(extensions or ['.txt', '.md'])
    paths = sorted((p for p in root.rglob('*') if p.is_file() and p.suffix in suffixes),
                   key=lambda p: p.relative_to(root).as_posix())
    # Preserve bounded parallel ingestion and input/result ordering.
    # As before, a directory can partially succeed and is not a transaction.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(
            lambda p: ingest_file(str(p), config, strategy=strategy,
                                  doc_id=p.relative_to(root).as_posix()), paths))
