# Changelog

Format: Keep a Changelog (keepachangelog.com/en/1.1.0/)
Versioning: Semantic Versioning (semver.org)

---

## [1.0.1] — 2026-03-07

### Fixed
- cli.py and server/app.py: load_dotenv() at startup; .env loads automatically
- Default embedding model: models/gemini-embedding-001
- eval/eval_retrieval.py: retrieve() uses functional API (embedder, store)

### Added
- testdata/: eval documents aligned to eval/dataset.json
- README and repo description updated to current model names

---

## [1.0.0] — 2026-03-06

Initial stable release.

### Added
- rag ingest / rag query CLI; directory and single-file ingestion
- Gemini gemini-embedding-001 with RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY asymmetry
- ChromaDB cosine retrieval; configurable RAG_SCORE_THRESHOLD (default 0.4)
- Fixed-size and sentence-boundary chunking with configurable overlap
- Grounding-disciplined generation; inline doc_id citation
- Flask REST API: POST /ingest, POST /query, GET /documents, DELETE, GET /stats
- Store-only CLI commands operable without GEMINI_API_KEY
- pgvector swap path documented; store.py is sole ChromaDB reference
- Precision@k and MRR eval harness (eval/)
- 33 tests (all mocked); CI: pytest on Python 3.11 / 3.12
- Frozen Config dataclass; all modules stateless; no globals

---

[1.0.1]: https://github.com/axiom-llc/axiom-rag/releases/tag/v1.0.1
[1.0.0]: https://github.com/axiom-llc/axiom-rag/releases/tag/v1.0.0
