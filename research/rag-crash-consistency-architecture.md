# RAG Crash-Consistency Architecture

**Project:** AXIOM RAG
**Status:** Research architecture / implementation input
**Scope:** Local persistent Chroma-backed RAG lifecycle, concurrent access, replacement semantics, crash recovery, and compatibility
**Date:** 2026-09-11

## 1. Purpose

Define the smallest architecture that makes AXIOM RAG operationally safe under concurrent server use and ordinary process crashes without overstating guarantees that the underlying Chroma persistence model cannot provide.

The selected design is a **single-owner local RAG server** that exclusively owns the persistent Chroma client and collection. All normal CLI and consumer access crosses a server boundary rather than opening the persistence directory directly.

This document is an implementation input, not proof that Chroma itself is transactionally atomic across all of its internal persistence layers.

## 2. Problem Statement

A local persistent RAG system has several independent correctness hazards:

1. Multiple processes can open the same persistence directory and mutate overlapping logical documents.
2. Replacement is a multi-step operation: stage new chunks, make them authoritative, and retire the old generation.
3. A process can exit after some storage effects but before the caller records completion.
4. Chroma persistence spans SQLite metadata/state and HNSW index state; therefore application-level journaling cannot honestly guarantee atomic survival of arbitrary host power loss or kernel panic across every underlying write.
5. Legacy records may not carry the generation metadata required by a versioned replacement design.

The architecture must therefore distinguish **server-mediated logical atomicity and recoverability** from **physical storage atomicity**.

## 3. Selected Architecture

### 3.1 Single persistence owner

Exactly one long-lived server process owns:

- the Chroma `PersistentClient`;
- the active collection handle;
- mutation sequencing;
- startup recovery;
- server-side read/write synchronization.

Normal CLI operations (`ingest`, `query`, `delete`, `stats`) and downstream consumers use HTTP or a Unix-domain socket. They do not instantiate their own persistent Chroma clients against the live store.

For isolated unit tests, an ephemeral fixture may act as the sole embedded owner of a temporary persistence directory.

### 3.2 Startup ownership lock

The server acquires an advisory exclusive `flock` on a lock file associated with the persistence root **before** creating the long-lived `PersistentClient`.

If the lock cannot be acquired, startup fails closed. The implementation must not silently fall back to direct multi-process ownership.

The lock prevents cooperating AXIOM RAG processes from simultaneously owning the store. It cannot prevent an unrelated or deliberately non-cooperating program from bypassing the protocol and opening Chroma directly.

### 3.3 Long-lived client

Create one `chromadb.PersistentClient` during server initialization and retain it for the server lifetime. Do not create a fresh persistent client per request.

The lifecycle is:

```text
acquire process ownership lock
→ construct persistent client
→ open/create canonical collection
→ run recovery
→ begin serving requests
→ stop accepting work
→ release process resources
→ release ownership lock
```

### 3.4 In-process synchronization

Use an in-process read/write synchronization boundary:

- queries and other read-only operations may run concurrently when the underlying access pattern is safe;
- logical mutations serialize through the mutation/write lock;
- expensive embedding work occurs outside the mutation lock whenever possible;
- the final storage transition is performed while holding the mutation lock.

Read atomicity is guaranteed only for reads routed through the server protocol. Direct Chroma readers are outside the guarantee.

## 4. Logical Document Versioning

### 4.1 Opaque chunk IDs

Do not encode parseable delimiters or application semantics into storage IDs. Generate deterministic fixed-length IDs from structured fields, for example:

```python
import hashlib


def chunk_id(doc_id: str, generation: int, chunk_index: int) -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(generation).encode("ascii"))
    h.update(b"\x00")
    h.update(str(chunk_index).encode("ascii"))
    return h.hexdigest()
```

The ID is opaque. Authoritative interpretation comes from metadata, not string parsing.

### 4.2 Structured metadata

Each versioned chunk stores at least:

```python
{
    "doc_id": doc_id,
    "generation": generation,
    "chunk_index": chunk_index,
}
```

Preserve compatible user metadata separately without using it as a substitute for the required identity fields.

### 4.3 Legacy generation 0

Existing records that have `doc_id` but no generation metadata are treated as **generation 0**.

When computing the active generation for a document:

1. retrieve chunks matching `doc_id`;
2. interpret missing `generation` as `0`;
3. use the highest present generation as the current version;
4. replace legacy generation 0 with generation 1 on the first versioned replacement.

Do not rewrite legacy records merely to normalize metadata unless a migration is independently required.

## 5. Two-Phase Intent Journal

### 5.1 Goal

The journal records enough information to determine whether an interrupted mutation should be rolled back or completed when the server restarts.

The journal is an application-level recovery mechanism. It does not claim to make Chroma's separate persistence layers physically atomic under power failure.

### 5.2 Journal states

Use two durable states represented by atomic file replacement/rename semantics on the same filesystem:

```text
STAGING
COMMITTED
```

`STAGING` means the new generation is not yet logically authoritative and recovery may remove staged effects.

`COMMITTED` is the decision point: recovery must preserve the committed generation and finish cleanup of superseded generations if cleanup was interrupted.

### 5.3 Required intent contents

Before the first Chroma mutation, the `STAGING` record must contain all information required for deterministic recovery, including at minimum:

- operation identifier;
- document identifier;
- prior active generation, if any;
- target generation;
- **complete list of all staged chunk IDs**;
- operation kind (`replace`, `insert`, `delete`, or a smaller final set actually implemented);
- enough information to identify superseded chunk IDs or generations without relying on ambiguous string parsing.

The staged-ID list must be complete **before any staged chunk is written**. Appending IDs to the journal after writes begins recreates the crash gap the journal is intended to close.

### 5.4 Commit decision point

The authority transition is:

```text
write + fsync STAGING intent
→ write staged target-generation chunks
→ verify staged representation is complete enough for the operation
→ atomically transition journal to COMMITTED
→ retire superseded generation/chunks
→ remove completed journal
```

The transition to `COMMITTED` is the logical decision point. Cleanup after that point is recoverable and must not make the old generation authoritative again.

## 6. Replacement Protocol

For a non-empty replacement:

1. Read the active generation through the server.
2. Compute and embed the replacement outside the mutation lock where possible.
3. Determine target generation and all deterministic target chunk IDs.
4. Acquire the mutation lock.
5. Revalidate any state that could have changed while embeddings were computed.
6. Write and durably persist the complete `STAGING` intent.
7. Add all target-generation chunks.
8. Validate that the intended staged set is present.
9. Transition the journal to `COMMITTED` atomically.
10. Delete superseded chunks/generation.
11. Remove the completed journal.
12. Release the mutation lock.

A conflicting concurrent replacement must not silently overwrite state derived from a stale generation. Revalidation under the mutation lock must either advance from the latest generation or reject/retry at the application level.

## 7. Empty Replacement

Empty replacement semantics must be explicit before implementation because "replace document with zero chunks" can mean either deletion or an empty-but-present logical document.

Do not infer this behavior silently.

Until the product contract chooses one meaning, preserve the case as an unresolved API semantic and test it separately from ordinary non-empty replacement.

## 8. Delete Protocol

Deletion should use the same ownership and journaling model rather than becoming a special unjournaled mutation path.

At minimum, a recoverable delete intent records:

- document identifier;
- active generation(s)/chunk IDs targeted;
- operation identifier;
- decision state.

If deletion is not required in the first implementation increment, leave the existing safe behavior intact and do not claim the replacement journal already covers it.

## 9. Recovery

Run recovery after acquiring ownership and opening the canonical collection, before serving normal traffic.

### 9.1 Recover `STAGING`

For each valid `STAGING` intent:

1. identify the target generation and exact staged IDs from the intent;
2. remove staged target-generation records created by the interrupted operation;
3. preserve the previously active generation;
4. remove the recovered staging journal only after recovery actions complete.

This restores the pre-operation logical state for server-mediated clients.

### 9.2 Recover `COMMITTED`

For each valid `COMMITTED` intent:

1. preserve the target generation;
2. verify the committed target records that the application can validate;
3. finish deletion of superseded generation/chunks;
4. remove the completed journal only after cleanup completes.

Do not roll back a committed decision merely because post-commit cleanup was interrupted.

### 9.3 Malformed or ambiguous journals

Fail closed on recovery records that cannot be interpreted safely. Preserve them for operator diagnosis rather than guessing at destructive cleanup.

## 10. Filesystem Requirements

Journal transitions depend on same-filesystem atomic rename semantics. Keep staging and committed journal paths on the same filesystem and directory hierarchy unless the implementation explicitly proves equivalent semantics.

Where durability against ordinary process crash is required, use the appropriate file and directory synchronization sequence for the target filesystem.

Do not describe `rename()` alone as a guarantee against arbitrary power loss.

## 11. Server and CLI Boundary

The CLI becomes a protocol client:

```text
rag ingest ...  ─┐
rag query ...   ─┼→ local HTTP/UDS server → single Chroma owner
rag delete ...  ─┤
rag stats       ─┘
```

If the server is unavailable, commands should fail explicitly with an actionable message rather than silently opening the persistence directory directly.

This boundary is architectural, not cosmetic: bypassing it invalidates the single-owner and server-mediated read guarantees.

## 12. Correctness Invariants

The implementation should preserve the following invariants:

1. **Single cooperative owner:** at most one AXIOM RAG server owns a persistence root.
2. **One long-lived persistent client:** request handlers do not create competing persistent clients.
3. **No unjournaled replacement mutation:** all IDs needed for rollback are durable before the first target write.
4. **One logical decision point:** `COMMITTED` determines which generation recovery must preserve.
5. **Opaque storage IDs:** recovery and queries do not parse identity from ID delimiters.
6. **Legacy compatibility:** missing generation metadata means generation 0.
7. **Server-mediated read consistency:** supported readers observe states defined by the server protocol, not arbitrary intermediate application operations.
8. **No false physical-atomicity claim:** application recovery is not described as a proof of Chroma power-loss atomicity.

## 13. Validation Plan

### 13.1 Unit tests

Test at minimum:

- deterministic opaque ID generation;
- generation-0 interpretation;
- target-generation increment;
- complete intent creation before storage mutation;
- recovery of a `STAGING` replacement;
- recovery of a `COMMITTED` replacement;
- malformed journal fail-closed behavior;
- ownership lock rejection;
- concurrent server-mediated readers during safe read phases;
- conflicting replacement revalidation;
- explicit empty-replacement semantics once chosen.

### 13.2 Crash-injection tests

Against the pinned Chroma version used by AXIOM RAG, run subprocess/SIGKILL tests at every meaningful boundary:

```text
before journal
while journal is being durably created
immediately after STAGING is durable
between staged chunk writes
immediately before COMMITTED
immediately after COMMITTED
while deleting superseded chunks
before final journal removal
```

Restart the server after each forced termination and assert the logical invariants.

These tests are required before describing the design as crash-consistent for ordinary process termination.

### 13.3 Concurrency tests

Run multiple protocol clients that:

- query while a replacement is embedding;
- query during serialized mutation phases;
- race two replacements of the same `doc_id`;
- replace different documents concurrently where allowed;
- attempt to start a second server against the same persistence root.

## 14. Explicit Limitations

The architecture does **not** establish the following without further evidence:

- atomic survival of host power failure or kernel panic across Chroma SQLite and HNSW persistence;
- safety for processes that ignore the ownership protocol and open the persistence directory directly;
- linearizable reads from external direct Chroma clients;
- correctness of every Chroma release rather than the pinned version tested by AXIOM;
- a chosen semantic for empty replacement;
- complete filesystem durability without platform-specific synchronization testing.

Use precise claims such as **server-mediated crash recovery** or **process-crash-consistent replacement under the tested storage/runtime conditions**, not "provably atomic" or equivalent language unless stronger evidence is later established.

## 15. Implementation Order

Implement in this order to minimize risk:

1. Add the server ownership/lifecycle boundary and startup `flock`.
2. Route normal CLI consumers through HTTP/UDS; remove unrestricted direct-owner fallback.
3. Add structured generation metadata and opaque IDs while preserving legacy generation 0.
4. Add the mutation lock and stale-generation revalidation.
5. Add complete pre-mutation `STAGING` journal creation.
6. Add the `COMMITTED` decision transition and cleanup protocol.
7. Add startup recovery.
8. Add crash-injection and concurrency tests against the pinned Chroma version.
9. Only then broaden operational claims in documentation.

## 16. Decision

Adopt the **single-owner server + versioned records + two-phase intent journal** design as the canonical direction for local AXIOM RAG persistence.

Treat the journal as an application recovery mechanism, not as a substitute for underlying storage transactions. Preserve the unresolved storage-level durability limits and verify them empirically before making stronger guarantees.
