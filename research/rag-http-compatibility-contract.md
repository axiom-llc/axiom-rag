# RAG HTTP compatibility contract

Date: 2026-09-11. **Historical audit gate: decisions in §5.** The 2026-09-12
[decision-resolution supplement](rag-http-decision-resolution.md) now recommends
one policy per group and unlocks scoped implementation specification. Runtime
implementation remains unperformed; the supplement records explicit scope limits.
This is a source-grounded compatibility contract and implementation gate, not an
approved new wire API. No runtime change or CLI/APEX migration is made here.

Inspected clean source snapshots: RAG
`71cd66b386174e09697e6af5eb80cbde7e626095`, APEX
`13ae4233e5fea3e1e6ea91e9f18cdbd4d6961d7f`, research baseline
`c1aea6c867049415f77b3914d28ac859ae41d576`. Local and live remote heads matched.
The accepted [ownership/journal direction](rag-crash-consistency-architecture.md)
is implemented at the shared store boundary; its HTTP-only routing remains a
proposal. Source and tests below take precedence over older architecture prose.

## 1. Caller inventory and required observable behavior

The scan covered non-ignored production Python/shell/JS/TS and workflow sources
across the AXIOM workspace, including hidden workflow directories. The concrete
storage callers are below. No other production storage caller was found in that
scope; this is not a claim about external users of the published packages.

### CLI and canonical Python pipeline

[CLI](../axiom-rag/cli.py) loads one local `Config` from environment/overrides and
calls [pipeline](../axiom-rag/rag/pipeline.py) and
[store](../axiom-rag/rag/store.py) directly. It has no HTTP client or server URL
setting. Provider credentials are checked only by operations that need them.

| Entry point | Required inputs and result | Semantics a migration must preserve |
| --- | --- | --- |
| `rag ingest FILE [--strategy fixed\|sentences]` → `ingest_file(path, config, metadata=None, strategy='fixed', doc_id=None)` | Local path; result `{doc_id: str, chunks_stored: int}` | Client-local `expanduser`, UTF-8 decoding with `errors='replace'`; default ID is basename (`doc_id or basename` for Python override). Preserve ID exactly, including spaces. No server-side filesystem path access is required. |
| `rag ingest DIR` → `ingest_directory(directory, config, extensions=None, strategy='fixed', max_workers=4)` | Ordered list of the same per-document results | Recursive `.txt`/`.md` default selection; relative POSIX-path IDs; deterministic path/result order; positive workers and strategy validated before work. Existing parallel ingestion can partially succeed before an exception; no directory transaction/rollback exists. File reads/scanning stay client-local. |
| `ingest(text, doc_id, config, metadata=None, strategy='fixed')` | Text string, exact nonempty document identity, metadata dict or None, chosen strategy; `{doc_id, chunks_stored}` | Whole-document replacement, not append. Empty/whitespace-only text yields zero chunks, deletes the document, returns count 0, and does not call the provider. Invalid strategy fails even for empty text. Provenance/ownership checks remain required. |
| `rag query ...` → `pipeline.query(str, config)` / `pipeline_query(question, config)` | Question string; `{answer: str, sources: list[str], chunk_count: int, chunks: list[Chunk]}` | Embed question, retrieve, then generate only when chunks exist. CLI joins question arguments with spaces and prints answer plus source count/list. Embedded query does not strip or reject blank questions itself. |
| `rag list` → `list_documents(config)` | Sorted unique `list[str]` | Missing collection returns `[]`; inspect legacy namespaces without adopting provenance; does not create a collection. CLI prints IDs or “No documents ingested.” |
| `rag stats` → `collection_stats(config)` | `{total_chunks: int, documents: list[str]}` | Missing collection returns zero/empty; same inspection/provenance behavior as list. CLI prints JSON. |
| `rag delete ID` → `delete_document(doc_id, config)` | Exact ID; Python returns integer deleted chunk count | Delete only that document, missing document gives 0. CLI prints count. Provenance validation applies; missing collection may be created by `_get_collection`. |
| `rag create` → `create_collection(config)` | Unused configured namespace and embedding identity; Python returns Chroma collection handle | Create only; existing namespace raises, even if empty/compatible. CLI uses no handle methods: it only prints confirmation. Do not substitute “get or create”. |

`Chunk` is `{text: str, metadata: dict, score: float}`. Score is `1 - distance`,
filtered inclusively by `score >= config.score_threshold`, then sorted descending;
retrieval requests at most `min(top_k, collection_count)` candidates. Repeated
document IDs across chunks are meaningful. No stable tie-break order is promised.
Returned metadata includes user fields and persisted `doc_id`, `chunk_index`,
`generation`, `rag:operation` for new records; legacy fields are not fabricated.

Python `upsert(chunks: list[str], embeddings: list[list[float]], doc_id: str,
config, metadata: dict | None=None) -> None` is also exposed. Equal lengths,
configured dimensions and finite values are checked before store access. It
accepts precomputed vectors without embedding/generation calls, replaces the
whole document, and treats empty lists as deletion. `store.query(vector, config)`
and `store_query` return `list[Chunk]`; `pipeline.query(list, config)` dispatches
to that same raw-vector path rather than generating an answer.

Metadata is passed through Chroma 1.5.2 validation after the store overwrites its
identity fields. Inspected callers/tests use JSON scalar metadata such as
`{'source': 'unit_test'}`. Python's dict annotation is not a JSON-only promise:
installed Chroma validation also accepts lists and `SparseVector` objects.
Do not promise arbitrary Python object transport from this audit; any narrowing
of the public metadata/collection-handle API must be explicit.

### APEX has two distinct integration paths

1. The registered tool
   [`rag_multi_query`](../axiom-apex/apex/core/tools.py), installed by
   [toolloader](../axiom-apex/apex/core/toolloader.py), **already uses HTTP**.
   Input: `{question: str, hops?: int}`; default hops 2, clamped to 1–5.
   It sends sequential `POST /query` requests containing only `{question}` to
   `RAG_BASE_URL` (default `http://localhost:8000`), with optional bearer
   `RAG_API_TOKEN`, JSON content type and 60-second urllib timeout per hop.
   It reads `answer` (default empty string) and `sources` (default empty list),
   chooses a subsequent question from an answer line containing `?` or reuses
   the original, and returns `{answer: str, hops: int, hop_summaries: list[str],
   sources: list[str]}` with first-seen source deduplication. Transport/HTTP/JSON
   failures become `RuntimeError` naming the hop; response value types are not
   comprehensively validated. It has `retry_safe=False`; the
   [executor](../axiom-apex/apex/core/loop.py) does not automatically retry it.
   The executor applies a 300-second deadline to the whole tool effect and
   normalizes failures to `ToolTimeout`/`ToolExecutionError` results. Preserve that
   behavior; multi-hop continuation is not transport retry.
2. [`apex.core.rag.pipeline`](../axiom-apex/apex/core/rag/pipeline.py) and
   [`store`](../axiom-apex/apex/core/rag/store.py) directly re-export canonical
   Python functions, including private `_get_client`/`_get_collection` handles.
   Their public operations require all pipeline/store behaviors above. The
   [APEX configuration adapter](../axiom-apex/apex/core/rag/config.py) intentionally
   preserves a different generation-model default. Embedder/generator exports
   also remain direct canonical functions; they do not themselves own Chroma.
   The registered execution tool does not call these embedded exports.

[APEX evaluator](../axiom-apex/benchmarks/eval_rag.py) and
[RAG evaluator](../axiom-rag/eval/eval_retrieval.py) both perform
`embedder.embed_query` locally followed by `store.query(vector, config)`, returning
ranked `metadata['doc_id']` values, one per chunk. They deliberately request
`score_threshold=0.0` and configurable `top_k`, root and collection. Both parser
defaults still select legacy `documents`, rather than the canonical configuration
default. Do not silently retarget their datasets. They require a local API key
and do not generate answers. Their docstrings advertise exit code 2 for backend
errors, but runtime retrieval exceptions can escape without that normalization;
source behavior, not that advertised exit code, is the current evidence.

[ASON policy](../axiom-ason/ason/validator.py) classifies `rag_multi_query` as a
network-reaching tool; it is not another storage client. APEX's other SQLite
memory facilities are separate from this RAG collection and are outside migration.

### Tests and non-production interfaces

- [RAG pipeline tests](../axiom-rag/tests/test_pipeline.py) and
  [APEX pipeline tests](../axiom-apex/tests/test_pipeline.py) exercise replacement,
  empty deletion, relative filenames, metadata and generated result shape.
  APEX explicitly tests that its exported functions are the canonical functions.
- [RAG store tests](../axiom-rag/tests/test_store.py),
  [space tests](../axiom-rag/tests/test_space.py), and
  [APEX store tests](../axiom-apex/tests/test_store.py) use raw vectors, arbitrary
  temporary roots/collections, dimension overrides and configurable thresholds
  (including -1), plus private handles to seed/inspect legacy data.
- [Recovery tests](../axiom-rag/tests/test_recovery.py) require genuine isolated
  embedded owners and crash injection. Keep these owner-side tests; do not
  translate private raw Chroma handles into HTTP bypasses to satisfy fixtures.
- [HTTP tests](../axiom-rag/tests/test_server.py) explicitly assert rejection of
  blank questions, authentication behavior, provider redaction and startup failure.
  Some invalid requests are rejected before config/storage access. Future transport
  tests must be distinct from private owner-side fixtures. No inspected production
  caller requires an HTTP representation of a Chroma collection/client object.

## 2. Existing server wire API

[Server implementation](../axiom-rag/server/app.py) has exactly these five data
routes (excluding Flask static/automatic HEAD/OPTIONS). Config is one cached
server-local `Config`; unrecognized JSON fields are ignored, not overrides.

| Method/path | Request | Success response | Current mismatch |
| --- | --- | --- | --- |
| `POST /ingest` | JSON object with nonblank string `text`, `doc_id`; optional dict/None `metadata`; `strategy` fixed or sentences, default fixed | 201 `{doc_id, chunks_stored}` | `_text` strips text and ID; rejects empty/whitespace text instead of deletion. No caller chunk-size/overlap/namespace/embedding settings. |
| `POST /query` | JSON object with nonblank string `question` | 200 `{answer, sources, chunk_count, chunks}` | Strips question before embedding/generation; no raw-vector or retrieval-only mode, nor per-call retrieval/model controls. |
| `GET /documents` | No operation parameters | 200 `{documents: list[str]}` | Adapter must unwrap list; server-selected namespace only. |
| `DELETE /documents/<path:doc_id>` | URL-path ID | 200 `{doc_id, chunks_deleted: int}` | Adapter must unwrap count; nested relative IDs work, but leading-slash IDs do not dispatch. Exact identity transport needs coverage, not naive URL interpolation. |
| `GET /stats` | No operation parameters | 200 `{total_chunks, documents}` | Compatible result shape for the server-selected namespace. |

There is no creation, raw replacement, raw-vector retrieval, embedding-only,
generation-only, configuration-discovery or namespace-selection route. Missing
operations are established from the complete route map, not guessed URL names.

Every route checks optional bearer authentication first. With a configured token,
missing/malformed/wrong bearer gives 401. Without a token auth is disabled;
`main()` refuses unauthenticated non-loopback binding. A Flask/Werkzeug HTTP error
retains its status and headers (e.g. 405/Allow), with JSON `{error: description}`.
Provider `APIError` becomes redacted 502 `{error: 'upstream API error'}`;
`ValueError` becomes 400 with its message; other exceptions become generic 500
`{error: 'internal server error'}`. Ownership/recovery failures therefore are not
machine-distinguishable from other internal failures. Public Python calls instead
propagate validation, provider, ownership/recovery or Chroma exceptions. The CLI
has usage exits but no general remote-error mapping. HTTP status alone must not
be interpreted as proof that a replacement did or did not commit.

Normal `main()` calls `_get_collection` before serving, acquiring ownership,
recovering and opening/**creating** the configured collection. Thus even an
added “create current collection” route could not meet CLI create-only-if-unused
semantics under this startup sequence. Inspection can access untagged namespaces
through Python, while normal server startup rejects an untagged configured one.

## 3. Configuration and embedding compatibility constraints

[Canonical configuration](../axiom-rag/rag/config.py) accepts explicit overrides,
then environment settings, then defaults. The HTTP server receives none of those
caller overrides. Preserve or explicitly reject mismatches before provider calls
or mutations; silently using another namespace/model/settings is incompatible.

| Property | Current caller contract | Required decision/constraint at HTTP boundary |
| --- | --- | --- |
| `chroma_path`, `collection_name` | Caller-selected root/name; defaults `~/.rag/chroma`, `documents-gemini-embedding-2`; evaluators override both | Root ownership stays server-side. Specify mapping/allowed namespace selection and create/inspect lifecycle; do not transmit arbitrary local paths as authority. |
| `embedding_model`, `embedding_dimension` | Defaults `gemini-embedding-2`, 3072; explicit overrides supported | Require compatible declared space, including provider/model/dimension/schema, before mutation/retrieval. Same vector length alone does not establish provenance. |
| `chunk_size`, `chunk_overlap`, `strategy` | Defaults 512/64 words, fixed; positive size and `0 <= overlap < size`; sentences ignores overlap when grouping | Preserve requested chunking or explicitly validate a fixed server profile. Strategy alone is insufficient. |
| `top_k`, `score_threshold` | Positive k, threshold in [-1,1]; defaults 5/0.4; evaluators request 0.0 | Preserve per-call ranking/filter settings or reject unsupported profiles; do not silently ignore them. |
| `generation_model` | Canonical RAG default `gemini-2.5-flash`; APEX embedded default `gemini-3.5-flash-lite`; explicit/env override supported | One server default cannot preserve both automatically. Select permitted caller controls/profiles or approve an intentional compatibility change. |
| `gemini_api_key` | Caller-supplied/environment credential; store-only operations do not require it | Select server/provider credential ownership and embedding location. No approval exists to forward caller secrets. |

[Embedding adapter](../axiom-rag/rag/embedder.py) batches documents in groups of
100 and checks response count, dimensions and finite values. For embedding-2,
document preprocessing is `title: none | text: <chunk>` and query preprocessing
is `task: search result | query: <question>`, with requested output dimension.
Other models use retrieval-document/query task types. The adapter explicitly
rejects the retired model configured as `text-embedding-004`. These are current
source behaviors, not a fresh assertion about external model availability.
Provider calls use a 60,000ms timeout and one attempt.

[Generator](../axiom-rag/rag/generator.py) returns an explicit no-context answer,
empty sources and count 0 without a generation call. Otherwise it uses the
selected model and grounded prompt, returns sorted unique sources and retrieved
chunk count, and likewise makes one bounded provider attempt. Its standalone
Python export also accepts a custom system prompt and preselected context.
Those provider-only exports do not need new storage endpoints; whether “HTTP-only”
also relocates their provider work is unresolved. A text retrieval-only endpoint
could avoid client embedding for evaluators, but cannot by itself preserve the
exported raw-vector store interface.

## 4. Smallest missing capability set (conditional, not approved routes)

Do not implement a filesystem-ingestion service, generic Chroma RPC or a separate
APEX multi-query endpoint. Local file scanning plus existing HTTP operations can
cover much of the CLI. For exact compatibility of the inspected public storage
surface, the minimum capabilities are:

1. An explicit target/configuration agreement resolving §5. All subsequent
   operations must bind to that selected namespace and compatible embedding space.
2. Raw retrieval: accept a finite, dimension-compatible vector plus the selected
   retrieval controls and return exactly `list[Chunk]`, without embedding or
   generation. Existing `/query` cannot substitute for this operation.
3. Raw document replacement: accept complete chunk/vector arrays, exact document
   ID and supported metadata, run the existing protected `store.upsert`, and let
   the Python adapter retain its `None` return. Never implement it with raw Chroma
   writes. No new rollback or retry semantics are needed.
4. Create-only-if-unused and compatible non-adopting inspection for the selected
   namespace, with a serializable creation acknowledgement for the CLI. Namespace
   lifecycle must first be reconciled with eager server creation. Arbitrary
   Python collection handles are not a transport requirement.
5. Lossless identity/question transport. Preserve padded filenames and question
   input. Existing HTTP strictness cannot be silently broadened while also claiming
   unchanged HTTP behavior; decide whether compatibility uses separate/versioned
   operations or an explicitly approved behavior change.
6. A documented error/transport contract distinguishing failures from successful
   empty results; preserve provider redaction and no direct-owner fallback.
   No blind mutation retry after timeout/disconnection or generic 500.

Empty text does **not inherently require another endpoint**: a client can validate
strategy/chunking, translate zero chunks to the existing deletion operation, and
return `{doc_id, chunks_stored: 0}`, provided identity/target handling is exact.
List/stats/delete response adapters are otherwise mechanical. File/directory
requests need not send server filesystem paths. `rag_multi_query` already fits
current `/query` when deliberately using the server's configuration.

## 5. Exact blockers and minimum decisions to resume

The missing implementation is not one determined endpoint patch. Runtime work
stops here rather than silently choosing any of these contracts:

| Decision | Concrete conflict to resolve | Minimum owner/application answer |
| --- | --- | --- |
| Namespace and lifecycle authority | Caller-selectable root/collection; CLI create-only-unused; legacy inspection; server eagerly creates one configured namespace | Which existing caller namespace choices must remain supported, how they map to a server-owned root, which creation/inspection operations are permitted, and whether startup must defer collection creation. |
| Per-call configuration and provider ownership | Distinct RAG/APEX model defaults, evaluator threshold overrides, caller chunking/embedding settings and credentials versus fixed server config | Which options clients may select, which are fixed with explicit mismatch rejection, and whether embedding/generation credentials/work move to the server. Do not silently standardize the two model defaults. |
| Supported public transport surface | Precomputed-vector APIs, exact string identity/blank-query behavior and arbitrary Python handles/types versus strict JSON HTTP routes | Confirm raw-vector read/write trust scope, JSON metadata scope, owner-side-only private handles, and how to preserve Python inputs without silently changing existing HTTP validation. |

Wire route names, status/error discriminators and URL/UDS configuration can then
be specified narrowly around those answers. No distributed ownership, direct
Chroma fallback, credential forwarding or automatic mutation retries are implied.

## 6. Falsifiable acceptance gates for the next implementation

- Client use while a separate owner holds the root never constructs a local
  Chroma client. Unavailable server fails explicitly; no fallback or silent
  configuration substitution occurs.
- Same accepted inputs/configuration yield the same IDs, chunk counts, metadata,
  ranking/filter results, provider preprocessing and selected generation model
  through HTTP and embedded owner paths. Raw-vector operations make zero provider
  calls; standalone provider behavior is preserved or explicitly scoped out.
- Empty replacement deletes exactly its target, shorter replacement removes stale
  chunks, padded/nested/leading-slash IDs do not alias another document, and file
  batches retain deterministic results and existing partial-failure semantics.
- Unknown/mismatched spaces and non-finite/dimension-invalid vectors fail before
  mutation/provider work as applicable. Creation never adopts an existing
  namespace. List/stats retain non-adopting inspection semantics.
- Existing `/query` multi-hop tool success/error/auth behavior remains compatible;
  HTTP validation/redaction is retained under the chosen versioning decision.
- Timeout/connection loss during a mutation is reported as an uncertain outcome;
  client code neither retries automatically nor opens storage to infer success.
  Server-side journal recovery remains the authority; no power-loss claim is added.

## 7. Validation of this contract

Isolated Flask/provider-mocked probes with a real temporary Chroma store confirmed:
empty HTTP ingestion rejects while embedded ingestion deletes; padded IDs are
trimmed only by HTTP; query overrides are ignored and questions trimmed;
raw-vector query has no accepted form; leading-slash deletion does not dispatch;
recovery errors become generic 500; eager startup prevents create-only-unused;
and the actual APEX HTTP tool consumes the current server response for two hops.
No live provider, production database, or socket service was used.

Validation commands from the AXIOM workspace (Python 3.12.9, Chroma 1.5.2):

```sh
PYTHONPATH=axiom-rag /tmp/axiom-recovery-venv/bin/python -m pytest axiom-rag/tests/test_server.py axiom-rag/tests/test_space.py -q
# 47 passed
PYTHONPATH=axiom-rag /tmp/axiom-recovery-venv/bin/python -m pytest axiom-rag/tests --ignore=axiom-rag/tests/test_recovery.py -q
# 92 passed
PYTHONPATH=axiom-apex:axiom-rag /tmp/axiom-recovery-venv/bin/python -m pytest axiom-apex/tests/test_store.py axiom-apex/tests/test_pipeline.py axiom-apex/tests/test_chunker.py axiom-apex/tests/test_tool_retries.py -q
# 47 passed
```

The accepted crash suite is not repeated: no runtime or journal code changed.
Fresh package builds are not applicable to this Markdown-only artifact. Source
links (24 resolved), executable probe syntax and Git diff cleanliness passed.
The embedded probe matches the successfully executed temporary probe exactly.
The following probe is self-contained and can be saved to a temporary `.py` file
and run with `PYTHONPATH=axiom-rag:axiom-apex` and the same Python environment.

```python
import io
import json
import os
import tempfile
from unittest.mock import Mock, patch
from rag import pipeline, store
from rag.config import load_config
from apex.core.rag.config import load_config as apex_config
from apex.core.tools import _rag_multi_query_effect, RAG_MULTI_QUERY
import server.app as api

with patch.dict(os.environ, {}, clear=True):
    assert load_config().generation_model == 'gemini-2.5-flash'
    assert apex_config().generation_model == 'gemini-3.5-flash-lite'
print('distinct RAG/APEX generation defaults confirmed')
routes = sorted((sorted(rule.methods - {'OPTIONS', 'HEAD'}), rule.rule)
                for rule in api.app.url_map.iter_rules() if rule.endpoint != 'static')
assert len(routes) == 5
print('routes:', routes)
with tempfile.TemporaryDirectory() as root:
    cfg = load_config(chroma_path=root, collection_name='contract_probe',
                      embedding_dimension=3, gemini_api_key='synthetic', top_k=5)
    with patch.object(api, '_get_config', return_value=cfg), patch.object(api, '_api_token', ''):
        client = api.app.test_client()
        store.upsert(['old'], [[1., 0., 0.]], 'doc', cfg)
        assert client.post('/ingest', json={'text': '', 'doc_id': 'doc'}).status_code == 400
        assert store.list_documents(cfg) == ['doc']
        assert pipeline.ingest('', 'doc', cfg) == {'doc_id': 'doc', 'chunks_stored': 0}
        assert store.list_documents(cfg) == []
        print('HTTP empty ingest rejects; direct empty ingest deletes')
        with patch('rag.embedder.embed_texts', side_effect=lambda chunks, config: [[1., 0., 0.]] * len(chunks)):
            assert client.post('/ingest', json={'text': 'text', 'doc_id': ' file '}).get_json()['doc_id'] == 'file'
            pipeline.ingest('text', ' file ', cfg)
        assert store.list_documents(cfg) == [' file ', 'file']
        print('HTTP trims document identity; embedded preserves it')
        answer = {'answer': 'answer', 'sources': ['file'], 'chunks': [], 'chunk_count': 0}
        with patch.object(api.pipeline, 'query', return_value=answer) as query:
            response = client.post('/query', json={'question': ' q ', 'top_k': 1,
                                   'score_threshold': -1, 'generation_model': 'different',
                                   'collection_name': 'different'})
            assert response.status_code == 200
            query.assert_called_once_with('q', cfg)
        assert client.post('/query', json={'query_embedding': [1., 0., 0.]}).status_code == 400
        print('HTTP trims questions, ignores configuration overrides, has no vector-query form')
        with patch.object(api.store, 'delete_document', return_value=1) as delete:
            assert client.delete('/documents//doc').status_code != 200
            delete.assert_not_called()
        print('leading-slash document ID is not supported by current delete path')
        with patch.object(api.store, 'collection_stats', side_effect=RuntimeError('RAG recovery blocked')):
            response = client.get('/stats')
            assert response.status_code == 500
            assert response.get_json() == {'error': 'internal server error'}
        print('recovery errors collapse to generic HTTP 500')
        # Same startup operation as main(), without binding a socket.
        store._get_collection(cfg)
        try:
            store.create_collection(cfg)
        except Exception:
            pass
        else:
            raise AssertionError('startup namespace unexpectedly unused')
        print('server startup namespace cannot satisfy create-only-if-unused')
        def bridge(request, timeout):
            assert timeout == 60 and request.full_url == 'http://contract/query'
            assert request.get_header('Authorization') == 'Bearer synthetic-token'
            response = client.post('/query', json=json.loads(request.data))
            assert response.status_code == 200
            return io.BytesIO(response.data)
        with patch.dict(os.environ, {'RAG_BASE_URL': 'http://contract', 'RAG_API_TOKEN': 'synthetic-token'}), \
             patch.object(api.pipeline, 'query', return_value=answer), \
             patch('urllib.request.urlopen', side_effect=bridge) as send:
            result = _rag_multi_query_effect({'question': 'question', 'hops': 2})
            assert result['answer'] == 'answer' and result['sources'] == ['file']
            assert result['hops'] == 2 and send.call_count == 2
        assert not RAG_MULTI_QUERY.retry_safe
        print('existing APEX HTTP tool accepts current server shape; two hops; no retry-safe flag')
```
