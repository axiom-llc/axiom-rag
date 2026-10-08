# RAG HTTP decision resolution

Date: 2026-09-12. Classification: **ACCEPT_WITH_FOLLOWUP** (recommended policy
resolved; runtime implementation and transport validation remain unperformed).
Exactly one decision cycle; no runtime changes. This supplements
[compatibility contract §5](rag-http-compatibility-contract.md#5-exact-blockers-and-minimum-decisions-to-resume)
with alternatives, tradeoffs and one recommendation per group. It does not claim
that the proposed API already exists or supersede the accepted durability limits.

## Evidence revalidated

Live RAG `71cd66b386174e09697e6af5eb80cbde7e626095` and APEX
`13ae4233e5fea3e1e6ea91e9f18cdbd4d6961d7f` are unchanged, clean and match live
remote main. Research baseline is `0348da16cbdbc4c25a15b32bdd278de965671c76`.

- Namespace: [store](../axiom-rag/rag/store.py) distinguishes create-only,
  non-creating/non-adopting inspection and provenance-checked get-or-create.
  [server main](../axiom-rag/server/app.py) eagerly calls `_get_collection`.
  [CLI](../axiom-rag/cli.py) uses only a creation confirmation, not a collection
  handle. Both [RAG evaluator](../axiom-rag/eval/eval_retrieval.py) and
  [APEX evaluator](../axiom-apex/benchmarks/eval_rag.py) select root/collection,
  default to `documents`, and request threshold 0.0. A legacy untagged collection
  is inspectable but is already rejected for retrieval; migration must not adopt it.
- Configuration: [canonical config](../axiom-rag/rag/config.py) validates caller
  chunking/ranking settings; [APEX config](../axiom-apex/apex/core/rag/config.py)
  preserves `gemini-3.5-flash-lite` versus canonical `gemini-2.5-flash`.
  [pipeline](../axiom-rag/rag/pipeline.py) checks collection identity before
  provider work. [embedder](../axiom-rag/rag/embedder.py) owns preprocessing and
  vector validation; [generator](../axiom-rag/rag/generator.py) uses the chosen
  model and avoids generation without context. Each uses one provider attempt.
  Evaluators embed locally and use raw retrieval; provider-only exports have no
  storage ownership. Server JSON extras currently cannot override cached config.
- Transport: store exports raw retrieval/replacement; APEX
  [store](../axiom-apex/apex/core/rag/store.py) and
  [pipeline](../axiom-apex/apex/core/rag/pipeline.py) re-export canonical behavior.
  Server trims text/IDs/questions, rejects blank input and has no raw operations.
  [space](../axiom-rag/rag/space.py) checks declared identity, dimensions and
  finiteness, not actual vector origin. [persistence](../axiom-rag/rag/persistence.py)
  requires nonempty string document identities, overwrites reserved metadata,
  serializes recovery and journaled replacement, and retains a cooperative root
  lock/client until process exit. Whitespace-only IDs are nonempty identities.
  [Registered APEX tool](../axiom-apex/apex/core/tools.py) already uses `/query`.

## Materially viable alternatives and evaluation

All alternatives below retain one server owner per root and route every storage
operation through the protected store. Direct Chroma RPC, client-selected server
filesystem paths, automatic fallback and credential forwarding are excluded:
they contradict the accepted boundary or have no demonstrated caller need.

| Group / alternative | Ownership and durability | Determinism / reproducibility | Trust clarity | Caller compatibility | Migration complexity | Security / misuse | Maintenance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Namespace: one configured namespace per service | Preserves owner/journal; startup must still recover without creating | Explicit endpoint per target | Smallest authority | Multiple named datasets need separately configured services; same-root owners cannot coexist | Low code, higher deployment work | Small namespace scope | More service instances/root rearrangement |
| Namespace: explicit server allowlist under one root **(recommended)** | One owner/client/journal boundary; no new writer | Stable names, explicit rejection; no dataset retargeting | Server grants namespace and create authority | Covers current collection selectors and legacy inspection; root mapping must be explicit | Small resolver plus startup separation | Bounds creation/enumeration to configured names | One static map; no dynamic registry |
| Namespace: any valid collection name under fixed root | Same owner/journal | Exact names, but caller-created namespace inventory grows | Caller controls namespace allocation | Broadest name compatibility | Slightly less configuration | Unbounded namespace creation/inspection scope | Cleanup and inventory burden |
| Config/providers: all provider work local; storage HTTP | Same storage boundary | Existing config/provider requests easiest to retain | Provider secrets stay with callers, storage separate | Best credential compatibility; current server provider path remains separate | Low adapter work, pipeline coordination stays client-side | More credential holders; raw write trust necessary | Two pipeline locations persist |
| Config/providers: server pipelines, explicit controls; standalone/evaluator provider calls stay local **(recommended)** | Owner validates before provider work and journal mutation | Explicit effective settings/space; external generation remains nondeterministic | Credentials stay where used; no key transport | Preserves ranking/chunking/models and evaluator raw path; remote pipeline intentionally uses server account | Narrow controls and mismatch checks | Server model allowlist bounds account use; trusted callers can incur provider cost | Reuses canonical pipeline; no provider-only RPC |
| Config/providers: server-only provider work, fixed named profiles | Same protected owner | Profiles reproducible when versioned; no arbitrary settings | Central account and policy | Requires evaluator rewrite and additional text-retrieval capability; profiles must cover distinct defaults/settings | Larger migration | Central credentials, smallest per-call model surface | Profile inventory plus extra capability |
| Transport: separate versioned JSON compatibility operations **(recommended)** | Raw writes invoke journaled upsert; reads serialize | Exact JSON strings and declared space, no normalization | Trusted raw caller asserts provenance; server validates structure | Preserves observed Python data and current HTTP validation | Small explicit schemas/adapters | Raw poisoning risk confined to trusted service principals; no object deserialization | Two clearly bounded contracts, shared implementation |
| Transport: change existing routes to lossless typed forms and add raw operations | Same protected owner | Exact inputs possible | Same raw trust assumption | Breaks tested blank rejection/trimming contract; requires intentional existing-HTTP migration | Fewer route families, more caller/test transition | Similar raw risk | One eventual contract, larger immediate regression surface |

There is no demonstrated need for multi-tenant roles, arbitrary Python metadata
serialization, Chroma handles over HTTP, a multi-hop server endpoint, or server
filesystem ingestion. A text-only API is insufficient for the exposed raw store
surface. These are not additional viable compatibility alternatives.

## Recommended policies

### 1. Server-owned root and explicit namespace allowlist

Configure one canonical persistence root per server, with allowed collection names
and expected embedding identity. Default to the existing configured collection;
additional names (including `documents` for inspection) require explicit server
configuration. Clients select exact names from this set, never filesystem paths.
Map existing caller roots to a server target explicitly in client configuration;
reject unmapped roots or namespace mismatches before any request with effects.
Do not infer a mapping from matching basenames or silently change evaluator defaults.

Own and recover the root before serving, without creating a collection. Inspection
returns existing list/stats or missing-empty, never adopts provenance. Permit
create-only-unused for configured names using server-defined embedding identity;
return a serializable acknowledgement, not a handle. Preserve existing protected
get-or-create behavior for data operations on missing configured names; existing
untagged/incompatible names remain inaccessible for data operations. Recovery
failure still prevents serving. Creating a collection is not newly claimed to be
a journaled document transaction.

### 2. Server pipelines with explicit bounded configuration

Server owns storage settings, namespace embedding identity and credentials for
HTTP text ingestion/query. Clients send expected provider/model/dimension/schema
as assertions, not commands to rewrite provenance. Validate target/identity before
provider work; non-adopting list/stats may inspect unknown provenance.

Preserve validated per-call chunk size/overlap/strategy and top-k/threshold
(including 0.0 and -1). Send the client's resolved generation model explicitly;
server permits configured model IDs, initially including both existing defaults.
Other model overrides require explicit server allowance or fail before provider
work. No silent default unification or ignored settings. Return effective public
target/settings through a narrow configuration agreement; revalidate on each
operation so discovery does not become stale authority. Do not add sessions.

Never serialize `gemini_api_key`. HTTP pipeline work uses the server account,
an explicit migration difference from caller-account execution. Standalone
embedder/generator exports and evaluator local embedding retain local credentials
and preprocessing. Their storage calls become HTTP raw retrieval. Thus HTTP-only
means storage ownership, not a ban on existing provider-only calls. Server provider
ownership removes duplicate pipeline authority where useful without inventing
embedding-only/generation-only services. Raw operations need no provider key.

Unsupported explicit client settings fail with an actionable mismatch, rather
than silently using defaults. Client documentation must state the credential
ownership change; a nonempty local key must never be treated as remote credential
selection. No inspected caller requires distinct remote billing identities.

### 3. Versioned, lossless, trusted JSON data operations

Leave existing five HTTP routes' request validation and APEX `/query` behavior
intact. Add a separate versioned compatibility surface with specific operations,
not a method-name dispatcher. Carry document IDs in JSON, including delete;
preserve exact nonempty strings, spaces, nested and leading slashes. Preserve
question strings including blank strings and input text without transport trimming.
Validate strategy/settings before empty ingestion maps to journaled deletion.
Preserve existing counts, raw list-of-Chunk results and Python upsert `None` adapter.

Raw read/write is for the same trusted application principals authorized to use
the configured service, not adversarial tenants. Apply existing bearer/local-only
binding restrictions to every new operation; no new role system is justified.
Require declared space equality, finite numeric (non-boolean) vectors of configured
dimension, matched chunk/vector counts and complete validation before mutation.
These checks cannot prove vectors came from the declared model: a trusted writer
can poison retrieval, just as the existing local raw writer can. Do not claim
provenance attestation or add arbitrary supplied embedding providers.

For new client writes, metadata is None or a string-keyed object of Chroma-supported
JSON scalars (string, boolean, integer, finite float); reject null values, arrays,
nested objects and Python objects explicitly. Empty metadata is allowed. Preserve
server overwrite semantics for reserved identity/generation/operation fields;
clients cannot set their effective values. This deliberately defines a narrower
remote input surface, not full arbitrary-dict compatibility. Existing JSON-safe
stored metadata may be returned losslessly; unsupported stored values produce an
explicit error, never silent conversion/drop. No inspected production caller
requires complex metadata. Private handles remain owner-side test facilities;
public create's remote result is explicitly an acknowledgement.

No automatic mutation retry, direct fallback, or success inference from HTTP
failure. Timeouts/disconnections may leave outcome uncertain; journal recovery
remains server authority. Preserve redaction. Transport/error schemas must separate
validated rejection from success and uncertain failures without asserting that
every 4xx/5xx proves non-commit. No new durability, tie-order, UUID determinism,
exactly-once, provider-output reproducibility or power-loss guarantee follows.

## Owner input and implementation prerequisite

No external owner choice is necessary to recommend these three policies for the
inspected trusted local application scope. This is a recommendation, not a claim
of separate owner approval. Namespace inventory/root mappings, allowed extra
models and server credential provisioning are explicit deployment inputs; missing
values fail closed. Do not infer access grants to unrelated namespaces.

An untrusted or separately billed multi-tenant deployment would require an owner
to choose raw-write authority and credential/billing isolation; it is outside the
observed callers and does not block this scoped contract. Requiring unsupported
Python object compatibility would likewise be a new requirement.

Unlocked: a bounded implementation specification can now assign concrete versioned
routes, target/config schemas, error discriminators and client adapters using the
three recommendations above. Runtime work remains unperformed by instruction,
and must satisfy compatibility contract §6 plus allowlist rejection, explicit
model selection, credential non-transport, lossless IDs/questions, schema rejection
before mutation and recovery-before-serving-without-creation tests. No further
architecture/owner decision is identified for that scope. Do not claim migration
complete until separate-owner HTTP tests establish no client Chroma construction.

## Validation

Python 3.12.9 / Chroma 1.5.2, from the workspace root:

```sh
PYTHONPATH=axiom-rag python -m pytest axiom-rag/tests/test_server.py axiom-rag/tests/test_space.py axiom-rag/tests/test_pipeline.py axiom-rag/tests/test_store.py -q
# 76 passed
PYTHONPATH=axiom-apex:axiom-rag python -m pytest axiom-apex/tests/test_store.py axiom-apex/tests/test_pipeline.py axiom-apex/tests/test_tool_retries.py -q
# 32 passed
PYTHONPATH=axiom-rag:axiom-apex python /tmp/rag-http-decision-probe.py
# All assertions passed; exact Python probe extracted from compatibility contract §7.
```

The probe reproduced all prior mismatches and actual APEX two-hop HTTP behavior
using temporary real Chroma and mocked providers. Its logged generic server error
is the expected recovery-error assertion. The prior `/tmp/axiom-recovery-venv`
was absent; installed Python had the required dependencies, so no installation
was needed. No live provider or production database was used. Source-link checks
and Git diff checks passed. Crash tests/builds were not repeated for Markdown-only
changes; passing existing tests validates the factual baseline, not the proposed API.
Read-only remote checks required host access after sandbox DNS/SSH failures and
then confirmed all three baseline remote heads.
