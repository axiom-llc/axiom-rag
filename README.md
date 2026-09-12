[![PyPI](https://img.shields.io/pypi/v/axiom-rag.svg)](https://pypi.org/project/axiom-rag/)
![CI](https://github.com/axiom-llc/axiom-rag/actions/workflows/ci.yml/badge.svg)

# axiom-rag

Canonical AXIOM Retrieval-Augmented Generation library and HTTP service.

Version 1.5.0 (unreleased) supports document ingestion, Gemini embeddings, ChromaDB-backed
semantic retrieval, grounded source-cited generation, retrieval evaluation,
single-owner persistent storage, journaled process-crash recovery, and a bounded
HTTP compatibility client.

Python 3.11+ · Gemini API · ChromaDB 1.5.2 · Flask · MIT

## Current boundary

`axiom-rag` supports two access modes:

- embedded owner-side Python/evaluator access to the local persistent store;
- a server-owned HTTP boundary.

A persistence root has one cooperating process owner. Local evaluators and
owner-side Python callers cannot open the root concurrently with its RAG server.

The versioned HTTP storage API and `rag.http_client.Client` are implemented.
CLI and APEX public storage adapters use that client through `rag.remote`.
`rag_multi_query` continues to use the existing `/query` route.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
````

For a regular installation:

```bash
python -m pip install axiom-rag
```

Server-side Gemini work requires the server's own `GEMINI_API_KEY`. Migrated
CLI/APEX callers do not need or forward a provider key. Standalone local
provider operations and evaluators retain their own credentials.

## CLI

Start the host-local server separately with `python -m server.app`, providing
`GEMINI_API_KEY` only in its environment for text ingestion/query. Then explicitly
set the caller target:

```bash
export RAG_BASE_URL=http://127.0.0.1:8000
```

This deployment maps only canonical `~/.rag/chroma`, collection
`documents-gemini-embedding-2`, and space `google-gemini / gemini-embedding-2 /
3072 / schema 1`. Missing URL or other roots/namespaces/spaces fail closed.
These are adapter deployment constraints, not changed `rag.config` defaults.
The root selector is checked locally; clients do not open it or send it to the server.
Loopback uses no token; an explicitly configured `RAG_API_TOKEN` is sent as bearer.
No new service discovery, retry, redirect or local fallback is provided.


```bash
rag create
rag ingest ./docs
rag ingest ./docs/policy.txt
rag ingest ./docs --strategy sentences
rag query "what is the cancellation window?"
rag list
rag delete policy.txt
rag stats
```

`rag create` creates only an unused configured namespace and fails when that
namespace already exists.

Single-file ingestion uses the filename as the document ID. Recursive directory
ingestion uses paths relative to the input directory so equal basenames in
different directories remain distinct.

## Configuration

Environment variables are resolved explicitly through `rag.config`.

| Variable                     | Default                                    | Purpose                                         |
| ---------------------------- | ------------------------------------------ | ----------------------------------------------- |
| `RAG_BASE_URL`              | no storage-client default                  | Required explicit CLI/APEX service target.      |
| `GEMINI_API_KEY`             | unset                                      | Authenticate Gemini embedding/generation calls. |
| `RAG_CHROMA_PATH`            | `~/.rag/chroma`                            | Persistent ChromaDB root.                       |
| `RAG_COLLECTION`             | `documents-gemini-embedding-2`             | Default collection/namespace.                   |
| `RAG_CHUNK_SIZE`             | `512`                                      | Words per fixed-size chunk.                     |
| `RAG_CHUNK_OVERLAP`          | `64`                                       | Overlap between adjacent fixed chunks.          |
| `RAG_TOP_K`                  | `5`                                        | Maximum retrieved chunks.                       |
| `RAG_SCORE_THRESHOLD`        | `0.4`                                      | Cosine-similarity floor.                        |
| `RAG_EMBEDDING_MODEL`        | `gemini-embedding-2`                       | Embedding model.                                |
| `RAG_EMBEDDING_DIMENSION`    | `3072`                                     | Requested and enforced vector length.           |
| `RAG_GENERATION_MODEL`       | `gemini-2.5-flash`                         | Local pipeline generation model.                |
| `RAG_API_TOKEN`              | unset                                      | Bearer token for HTTP authentication.           |
| `RAG_HOST`                   | `127.0.0.1`                                | HTTP bind host.                                 |
| `RAG_HTTP_NAMESPACES`        | default namespace only                     | Optional JSON namespace/space allowlist.        |
| `RAG_HTTP_GENERATION_MODELS` | established defaults plus configured model | Optional JSON generation-model allowlist.       |

A non-loopback HTTP bind requires a non-empty `RAG_API_TOKEN`.

Changing an embedding model requires a fresh namespace. Never mix vectors from
different embedding spaces or retroactively assign provenance to an unknown
collection.

## Embedding-space identity

New compatible collections record:

* provider: `google-gemini`;
* normalized model ID;
* vector dimension;
* schema version, currently `1`.

Every semantic query and mutation validates this identity. Missing, partial, or
conflicting provenance fails closed. Unknown existing collections may be
inspected but are not adopted.

Raw vectors must have the configured dimension and contain only finite numeric
values. NaN, positive infinity, negative infinity, and booleans are rejected
before storage mutation.

## Replacement and crash recovery

Re-ingesting an existing document replaces its chunks using opaque versioned
record IDs and increasing generation metadata.

The storage boundary:

1. acquires an exclusive advisory `flock` for the canonical persistence root;
2. retains one Chroma client for that root in the process;
3. serializes supported reads and mutations;
4. writes a file-synchronized `STAGING` intent before staging replacement data;
5. verifies the new records and durably records `COMMITTED`;
6. removes the superseded generation;
7. removes the journal only after successful completion.

Startup and subsequent store access recover interrupted operations:

| Journal state       | Recovery                                                            |
| ------------------- | ------------------------------------------------------------------- |
| `STAGING`           | Remove staged records and preserve the prior generation.            |
| `COMMITTED`         | Verify the replacement, preserve it, and finish old-record cleanup. |
| malformed/ambiguous | Fail closed and retain the journal for diagnosis.                   |

Recovery is repeatable after another interruption. Empty replacement retains
document-deletion semantics.

These guarantees cover tested process termination on local Linux filesystems
with ChromaDB 1.5.2. They do not establish host-power-loss/kernel-panic
atomicity across Chroma SQLite/HNSW, distributed writer coordination, or safety
against direct non-cooperating Chroma access.

## HTTP service

Start the loopback server:

```bash
python -m server.app
```

The server listens on port `8000`. It acquires ownership and performs recovery
without creating a missing collection at startup.

### Existing API

The existing routes remain supported:

| Route                 | Method | Purpose                                                   |
| --------------------- | ------ | --------------------------------------------------------- |
| `/ingest`             | POST   | Chunk, embed, and ingest text into the default namespace. |
| `/query`              | POST   | Retrieve and generate a grounded answer.                  |
| `/documents`          | GET    | List document IDs.                                        |
| `/documents/<doc_id>` | DELETE | Delete one document.                                      |
| `/stats`              | GET    | Return collection statistics.                             |

`rag_multi_query` in APEX currently uses `/query`.

### Versioned compatibility API

All compatibility operations are strict JSON `POST` requests:

| Route         | Purpose                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------- |
| `/v1/inspect` | Inspect configured namespace existence, provenance, settings, models, and contents without creating/adopting it. |
| `/v1/create`  | Create an allowed namespace only when unused.                                                                    |
| `/v1/ingest`  | Ingest exact text/document identity through the server pipeline.                                                 |
| `/v1/replace` | Replace exact chunks using caller-supplied validated vectors.                                                    |
| `/v1/fetch`   | Perform vector retrieval without embedding or generation.                                                        |
| `/v1/query`   | Retrieve and generate with explicit validated settings/model.                                                    |
| `/v1/delete`  | Delete an exact document identity.                                                                               |

All operations require an allowed namespace. Operations other than inspection
also require the exact configured embedding-space assertion.

The canonical default space is:

```json
{
  "provider": "google-gemini",
  "model": "gemini-embedding-2",
  "dimension": 3072,
  "schema": 1
}
```

Caller-visible document identities are preserved exactly by the versioned
surface. Scalar write metadata accepts string keys with string, boolean,
integer, or finite-float values. Nested values, arrays, null members, and
non-finite floats fail closed.

Versioned errors use:

```json
{
  "error": {
    "code": "stable_code",
    "outcome": "not_started"
  }
}
```

`outcome` becomes `unknown` once dispatch may have begun. Error responses do
not expose provider exception bodies, credentials, or internal traceback data.
A failed HTTP response does not prove that a mutation rolled back.

## HTTP client

`rag.http_client.Client` is a standard-library client for the `/v1` surface.

It requires:

* explicit base URL;
* explicit namespace;
* exact embedding-space assertion;
* optional bearer token.

It exposes `inspect`, `create`, `ingest`, `replace`, `fetch`, `query`, and
`delete`.

The client:

* performs one request per operation;
* does not retry;
* refuses redirects;
* validates response shapes;
* has configurable socket-I/O timeout;
* bounds request and response sizes;
* never falls back to direct Chroma access.

Provider credentials and filesystem roots are server-owned and are never
transported by this client.

## Architecture

```text
cli.py
server/
├── app.py              legacy HTTP service + server startup
└── compat.py           /v1 compatibility policy/routes
rag/
├── chunker.py          fixed/sentence chunking
├── config.py           validated configuration
├── embedder.py         Gemini embedding adapter
├── generator.py        grounded Gemini generation
├── http_client.py      bounded /v1 client
├── remote.py           mapped CLI/APEX HTTP adapters and local file reads
├── persistence.py      ownership/journal primitives
├── pipeline.py         ingest/query orchestration
├── space.py            embedding-space/vector validation
└── store.py            Chroma storage + replacement/recovery
eval/                   retrieval evaluation
tests/                  unit, HTTP, ownership, and crash-recovery tests
```

Direct Chroma access outside `rag.store` is outside the supported ownership and
recovery contract.

## Evaluation

Measure retrieval quality with Precision@k and MRR:

```bash
python eval/eval_retrieval.py --dataset eval/dataset.json
python eval/eval_retrieval.py --dataset eval/dataset.json --json
```

The evaluation corpus must already be ingested and its `relevant_doc_ids` must
match ingest document IDs.

## Test and build

```bash
python -m compileall -q rag server cli.py tests
python -m pytest tests -q
python -m build
python -m pip check
git diff --check
```

Tests mock provider access unless a manual live-provider check is explicitly
selected.

## Migration status and limitations

CLI and APEX storage adapters are migrated for the explicit host-local mapping.
Install matching source/wheels; APEX requires RAG 1.5.0 for these adapters.
`rag.store` and `rag.pipeline` remain embedded owner/evaluator interfaces.
Evaluators retain their local `documents` dataset; no HTTP grant or migration is
implied. Public remote create returns an acknowledgement rather than a Chroma
handle. HTTP failures raise redacted `RemoteError` and CLI execution fails visibly;
there is no success inference or replay after an unknown mutation outcome.

File discovery/reads remain local, preserving basename and relative POSIX IDs,
UTF-8 replacement decoding, bounded parallel directory work and ordered results.
Directory ingestion remains nontransactional and may partially succeed.
Caller-resolved chunk/retrieval settings are sent explicitly. RAG/CLI keeps
`gemini-2.5-flash`; APEX keeps `gemini-3.5-flash-lite`; the server permits both.
Do not force a shared generation override merely to configure this deployment.

Do not assume:

* exactly-once HTTP mutations;
* request deduplication;
* automatic retry safety after an uncertain response;
* host-power-loss atomicity;
* distributed/multi-worker ownership;
* protection from non-cooperating direct Chroma writers.

Server journal recovery remains authoritative after uncertain mutation outcomes.

## License

MIT — [AXIOM LLC](https://axiom-llc.github.io/)
