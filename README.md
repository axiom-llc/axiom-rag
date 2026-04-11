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

1. **Ingest** — chunks documents, embeds them via Gemini `gemini-embedding-001`,
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
python -m server.app   # default: 0.0.0.0:8000
```

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
      ├─ rag/embedder.py        ← Gemini gemini-embedding-001 (stateless)
      ├─ rag/store.py           ← ChromaDB upsert / cosine retrieval
      └─ rag/generator.py       ← Gemini 2.5 Flash; context-grounded answers

  rag/config.py                 ← frozen Config dataclass; env + override resolution
```

All modules are stateless.  `pipeline.py` is the only file that calls more
than one module.  Config is resolved once at startup and passed explicitly —
no globals, no module-level singletons.

---

## Design Notes

**Embedding model.**  This pipeline uses `models/gemini-embedding-001`, the
currently available Gemini embedding model via the `google-genai` SDK.  The
default in `config.py` and `.env.example` reflects this.  Use
`RAG_EMBEDDING_MODEL` to override if your API key has access to additional
models.

**Embedding asymmetry.**  The Gemini embedding API distinguishes `task_type`:
`RETRIEVAL_DOCUMENT` for ingestion and `RETRIEVAL_QUERY` for queries.  Using
the wrong type for either degrades retrieval precision measurably.  Both are
set explicitly in `embedder.py`.

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
| `RAG_COLLECTION`        | `documents`                    | ChromaDB collection name           |
| `RAG_CHUNK_SIZE`        | `512`                          | Approximate words per chunk        |
| `RAG_CHUNK_OVERLAP`     | `64`                           | Overlap between consecutive chunks |
| `RAG_TOP_K`             | `5`                            | Max chunks retrieved per query     |
| `RAG_SCORE_THRESHOLD`   | `0.4`                          | Min cosine similarity (0–1)        |
| `RAG_EMBEDDING_MODEL`   | `models/gemini-embedding-001`  | Gemini embedding model             |
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
