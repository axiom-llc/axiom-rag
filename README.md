[![PyPI](https://img.shields.io/pypi/v/axiom-rag.svg)](https://pypi.org/project/axiom-rag/)
# axiom-rag
![CI](https://github.com/axiom-llc/axiom-rag/actions/workflows/ci.yml/badge.svg)

Production-grade Retrieval-Augmented Generation pipeline.  Ingest documents,
embed them, store vectors locally, retrieve semantically, and generate grounded
answers — from a clean CLI or REST API.

No hallucination from prior knowledge.  Every answer is bounded by what you
put in.  Sources are cited inline.

```bash
rag ingest ./docs
rag query "what is our refund policy?"
```

Python 3.11+ · Gemini API · ChromaDB · Flask · MIT

---

## What It Does

1. **Ingest** — chunks documents, embeds them via the configured Gemini embedding model,
   and stores vectors in a local ChromaDB collection
2. **Retrieve** — embeds the query and finds the top-k most semantically
   similar chunks above a configurable similarity threshold
3. **Generate** — passes retrieved context to Gemini 2.5 Flash with a strict
   grounding prompt; the model answers only from what was retrieved

The pipeline is fully local by default.  No database server required.
ChromaDB runs embedded.  The only network calls are to the Gemini API.

---

## Installation

```bash
pip install axiom-rag
cp .env.example .env  # set GEMINI_API_KEY
```

Or from source:

```bash
git clone https://github.com/axiom-llc/axiom-rag.git
cd axiom-rag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# For development (includes pytest)
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY at minimum
```

`.env` is loaded automatically at startup via `python-dotenv`.

---

## CLI

```bash
rag ingest ./docs                         # ingest directory (.txt and .md)
rag ingest ./docs/policy.txt              # ingest single file
rag ingest ./docs --strategy sentences    # use sentence chunking

rag query "what is the cancellation window?"

rag list                                  # list ingested documents
rag delete policy.txt                     # delete a document by ID
rag stats                                 # collection statistics (JSON)
```

Store-only commands (`list`, `delete`, `stats`) do not require `GEMINI_API_KEY`.

---

## REST API

```bash
python -m server.app   # default: 127.0.0.1:8000
```

For a non-loopback bind, set `RAG_HOST` and a non-empty `RAG_API_TOKEN`.
Clients send `Authorization: Bearer <token>`. The server refuses an unauthenticated
non-loopback bind. Authentication failures return 401, malformed inputs return
400, and HTTP routing errors retain their status codes.

### `POST /ingest`

```json
{
  "text": "Full document text...",
  "doc_id": "policy-v2",
  "metadata": {"category": "legal"},
  "strategy": "fixed"
}
```

Response `201`:
```json
{"doc_id": "policy-v2", "chunks_stored": 14}
```

### `POST /query`

```json
{"question": "what is the cancellation window?"}
```

Response `200`:
```json
{
  "answer": "The cancellation window is 30 days from purchase...",
  "sources": ["policy-v2"],
  "chunk_count": 3,
  "chunks": [
    {
      "text": "...",
      "metadata": {"doc_id": "policy-v2", "chunk_index": 4},
      "score": 0.91
    }
  ]
}
```

### `GET /documents`
### `DELETE /documents/<doc_id>`
### `GET /stats`

---

## Architecture

```
query / ingest
      │
      ▼
  cli.py / server/app.py        ← entry points; no business logic
      │
      ▼
  rag/pipeline.py               ← ingest() and query(); public interface
      │
      ├─ rag/chunker.py         ← fixed-size or sentence-boundary chunking
      ├─ rag/embedder.py        ← Gemini embedding adapter (stateless)
      ├─ rag/store.py           ← ChromaDB upsert / cosine retrieval
      └─ rag/generator.py       ← Gemini 2.5 Flash; context-grounded answers

  rag/config.py                 ← frozen Config dataclass; env + override resolution
```

All modules are stateless.  `pipeline.py` is the only file that calls more
than one module.  Config is resolved once at startup and passed explicitly —
no globals, no module-level singletons.

---

## Design Notes

**Embedding model.** Use `gemini-embedding-2` with the default fresh namespace
`documents-gemini-embedding-2`. The retired `text-embedding-004` still fails
closed. Re-embed into an unused collection when changing models; never mix
embedding spaces. APEX uses the same canonical collection and embedding defaults.

**Embedding asymmetry.** Gemini Embedding 2 uses explicit document and search
query prefixes in `embedder.py`. Other models use `RETRIEVAL_DOCUMENT` and
`RETRIEVAL_QUERY` task types.

**Score threshold.**  Retrieved chunks below the configured cosine similarity
floor (`RAG_SCORE_THRESHOLD`, default `0.4`) are dropped before generation.
This prevents low-relevance noise from polluting the context window.  Tune
down for broader recall, up for stricter precision.

**Chunk overlap.**  Fixed-size chunking uses a configurable overlap window
(`RAG_CHUNK_OVERLAP`, default `64` tokens) between consecutive chunks.
Overlap preserves context at chunk boundaries at the cost of slight index
size increase.

**Grounding discipline.**  The generation prompt instructs the model to answer
only from provided context, cite `doc_id` inline, and state explicitly when
context is insufficient rather than speculate.  The `system_prompt` parameter
in `generator.generate_answer()` exists for domain adaptation but changing it
to permit prior-knowledge use defeats the pipeline's purpose.

**pgvector swap path.**  `store.py` is the only file that references
ChromaDB.  To swap in pgvector: implement `upsert`, `query`,
`delete_document`, `list_documents`, and `collection_stats` in a new
`store_pg.py` and update the single import in `pipeline.py`.  Nothing else
changes.

**Lazy API key validation.**  `GEMINI_API_KEY` is not required at config load
time.  Validation fires at the entry to `embedder.embed_texts()` and
`embedder.embed_query()`.  This allows store-only CLI commands to work
without a key present.

**Environment loading.**  Both `cli.py` and `server/app.py` call
`load_dotenv()` at startup.  A `.env` file in the project root is loaded
automatically — no manual `export` required.

---

## Configuration

| Variable                | Default                        | Description                        |
|-------------------------|--------------------------------|------------------------------------|
| `GEMINI_API_KEY`        | *(required for embed/generate)*| Gemini API key                     |
| `RAG_CHROMA_PATH`       | `~/.rag/chroma`                | ChromaDB persistence directory     |
| `RAG_COLLECTION`        | `documents-gemini-embedding-2` | ChromaDB collection name           |
| `RAG_CHUNK_SIZE`        | `512`                          | Approximate words per chunk        |
| `RAG_CHUNK_OVERLAP`     | `64`                           | Overlap between consecutive chunks |
| `RAG_TOP_K`             | `5`                            | Max chunks retrieved per query     |
| `RAG_SCORE_THRESHOLD`   | `0.4`                          | Min cosine similarity (0–1)        |
| `RAG_EMBEDDING_DIMENSION` | `3072` | Requested and enforced vector length |
| `RAG_EMBEDDING_MODEL`   | `gemini-embedding-2`         | Gemini embedding model             |
| `RAG_GENERATION_MODEL`  | `gemini-2.5-flash`             | Gemini generation model            |

---

## Tests

```bash
pytest tests/ -v --tb=short
```

All tests mock Gemini API calls.  No live API or network access required.

```
tests/test_chunker.py    — chunking strategies, overlap, edge cases
tests/test_store.py      — score filtering, sort order, delete, list, stats
tests/test_pipeline.py   — ingest/query integration, file and directory helpers
```

To run with coverage (requires `pytest-cov`):

```bash
pip install pytest-cov
pytest tests/ -v --tb=short --cov=rag --cov-report=term-missing
```

---

## Evaluation

Retrieval quality is measured using Precision@k and MRR against a ground-truth
dataset.

```bash
python eval/eval_retrieval.py --dataset eval/dataset.json
python eval/eval_retrieval.py --dataset eval/dataset.json --json
```

The eval harness requires documents to be ingested before running.  The
`relevant_doc_ids` in `eval/dataset.json` must match the `doc_id` values
used at ingest time (i.e. the filename including extension when ingesting
via the CLI).

See [`eval/README.md`](eval/README.md) for full usage and tuning guidance.

---

## License

MIT — [AXIOM LLC](https://axiom-llc.github.io)

## Shared retrieval and document replacement

Version 1.1 is the canonical retrieval implementation used by APEX 3.1.
It incorporates APEX's validated chunk limits, embedding-count checks, empty-store
handling, and deterministic recursive ingestion. Single-file ingestion keeps
filename IDs; directory ingestion uses paths relative to the input directory,
so `a/faq.txt` and `b/faq.txt` remain distinct documents.

Re-ingestion replaces a document using opaque IDs and increasing generation
metadata (legacy records without a generation are generation 0). Empty
replacement retains the existing deletion semantics. Invalid dimensions and
non-finite embeddings are rejected before storage mutation.

### Local ownership and process-crash recovery

On Linux, the shared store boundary acquires an exclusive advisory `flock`
before opening Chroma and retains one client and lock per canonical persistence
root until process exit. A second cooperating process fails explicitly. Stop the
owner before using embedded CLI/APEX access to the same root, or use the running
server's HTTP API. CLI and Python calls do not yet become HTTP clients. Do not
use multiple server workers, fork an initialized owner, replace/unlink the lock
file, or modify the root through a direct Chroma client.

`python -m server.app` acquires ownership and recovers before serving. Embedded
and other WSGI entry points acquire ownership/recover at first store access.
Supported store reads and mutations serialize under one process lock; embedding
and generation requests remain outside that lock. Private collection/client
handles are outside the read-consistency guarantee.

Before staging any new chunk, `.rag-journal` records the complete old and new
IDs in a file-synchronized `STAGING` intent. Staging uses separate records and
preserves the old generation. After verifying target IDs and identity metadata,
an atomic file replacement and directory synchronization persist `COMMITTED`;
only then are old records retired. Deletes use the same decision protocol with
an empty target set. A successful return follows journal removal.

On startup or the next store access after an exception:

- `STAGING`: remove staged records and preserve the old generation.
- `COMMITTED`: verify the target records, preserve them and finish old cleanup.
- Malformed journals, identity mismatches or missing committed records: fail
  closed and retain the journal for operator diagnosis. Do not delete it to
  bypass recovery.

Recovery can itself be interrupted and repeated. Temporary journal files have
no authority until renamed; no Chroma mutation precedes durable intent. A caller
whose process/connection failed may not know whether commit occurred; this is
not request deduplication or automatic retry authorization.

The supported evidence is process termination on a local Linux filesystem with
Chroma **1.5.2**, pinned for reproducible recovery behavior. Tests interrupt
staging, commit, cleanup and recovery and reopen real persistent stores. This
journal does **not** establish host-power-loss/kernel-panic atomicity across
Chroma SQLite/HNSW, protection from non-cooperating access, or distributed writer
coordination. APEX's existing import paths and model defaults remain supported.

When developing all packages locally, install `axiom-rag` before `axiom-apex`,
then `axiom-ason`. Release them in that order for the new minimum versions.

### Collection compatibility and future reindexing

New collections store `rag:embedding:provider` (`google-gemini`), `model`
(normalized without `models/`), `dimension`, and integer `schema` (currently 1).
Schema 1 includes this adapter's retrieval document/query preprocessing.
`RAG_EMBEDDING_DIMENSION` defaults to 3072 and is explicitly requested from the
provider; returned and supplied vectors must match it. NaN and infinity are
rejected before vector-store access, preserving existing chunks on invalid
replacement input. Raw-vector callers must
supply vectors from the declared model/adapter, not merely matching lengths.

Every semantic query and mutation validates this identity. Missing, partial,
or conflicting metadata fails closed, including empty untagged collections.
No existing metadata is adopted or rewritten. Empty collections are not adopted
because a count check followed by metadata assignment cannot exclude concurrent
writers. `rag list` and `rag stats` permit non-embedding inspection and do not
create a collection. Chroma may perform database maintenance when opened; use
read-only SQLite or a copy when byte-for-byte preservation is required.

The default namespace is `documents-gemini-embedding-2`; explicit environment
overrides remain authoritative. APEX delegates to this same boundary. The legacy
`documents` collection remains unverified and is never adopted automatically.

The local development corpus was re-ingested on 2026-09-11: seven `testdata`
files, eight chunks, 3072 dimensions. Live refund, shipping, and payment retrieval
checks passed before default cutover. The old `documents` collection remains
preserved as legacy rollback evidence; no provenance was assigned retroactively.
This records a local migration, not migration of external deployments.

For a future authorized reindex, choose an **unused** namespace explicitly:

```bash
export RAG_EMBEDDING_MODEL=gemini-embedding-2
export RAG_EMBEDDING_DIMENSION=3072
export RAG_COLLECTION=documents-gemini-embedding-2-v1
rag create
rag ingest /path/to/verified-source-corpus
```

`rag create` needs no provider credentials and fails if the name already exists.
Ingestion requires credentials. Retries use the same tagged collection and source
IDs; directory ingestion uses relative paths. Keep the corpus and chunk settings
fixed across retries. Validate retrieval before explicitly configuring each
consumer with the new collection, model, and dimension. No automatic cutover,
legacy deletion, or in-place conversion occurs. Rollback preserves the old state;
it does not make unknown or retired embedding spaces safe to query.

The supported replacement candidate is documented by
[Google](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2).
Its availability does not establish compatibility with existing vectors.
Direct Chroma access or external metadata changes are outside this library's
invariant; restrict other writers. Cooperative process exclusion and journal recovery apply at the shared store
boundary; external direct access remains unsupported.

### Live provider checks

On 2026-09-11, explicit manual checks created one synthetic document in temporary
Chroma storage using `gemini-embedding-2`. Ingestion, 3072-dimensional vectors,
provenance metadata, semantic retrieval, and grounded `gemini-2.5-flash` generation
passed. A nonexistent embedding model returned HTTP 404; the retired model was
rejected locally. Both persistent collections remained byte-for-byte unchanged.

Gemini requests use a 60-second HTTP timeout and one SDK attempt. Retry explicitly
after assessing the failure; generation may already have consumed provider quota.
HTTP endpoints return a generic 502 for provider API failures and omit upstream
error bodies from responses and logs. Keep live checks manual and use synthetic
input with temporary storage; do not place live credentials in CI.
