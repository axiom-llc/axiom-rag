# Retrieval evaluation

`eval_retrieval.py` measures a locally owned Chroma collection against a
ground-truth JSON dataset. It reports Precision@k and mean reciprocal rank
(MRR). This evaluator is an owner-side, live-provider tool; it is not part of
the server-owned HTTP storage API and must not open a persistence root while the
RAG server owns that root.

## Metrics

- **Precision@k**: relevant retrieved chunks divided by `k`.
- **MRR**: reciprocal rank of the first relevant retrieved chunk, or zero when
  none is retrieved.

Metrics assess the supplied dataset and collection only. They do not establish
general retrieval quality, grounding quality, or provider availability.

## Prepare an isolated evaluation collection

The evaluator imports `rag.embedder` and `rag.store` directly, calls Gemini to
embed each query, and has no HTTP-client fallback. Use a persistence root and
collection that are not concurrently owned by `python -m server.app` or another
cooperating RAG process. It defaults to `~/.rag/chroma` and collection
`documents` (not the migrated HTTP namespace).

Set `GEMINI_API_KEY`, then ingest the evaluation corpus through a compatible
local owner workflow before evaluation. Do not point it at the canonical RAG
service root while that service is running.

## Run

From the repository root:

```bash
export GEMINI_API_KEY='...'
python eval/eval_retrieval.py --dataset eval/dataset.json
python eval/eval_retrieval.py --dataset eval/dataset.json --top-k 3
python eval/eval_retrieval.py --dataset eval/dataset.json --json
```

Select a separate collection or root explicitly when required:

```bash
python eval/eval_retrieval.py \
  --dataset eval/dataset.json \
  --chroma-path /path/to/eval-chroma \
  --collection eval-documents \
  --top-k 5
```

`--dataset` must name a JSON list whose entries contain `query` and
`relevant_doc_ids`. The IDs must match the document IDs stored in the evaluated
collection. `--json` emits a machine-readable report; the default is a table.

The evaluator sets its query threshold to `0.0` to preserve ranking for the
metrics. This differs intentionally from the pipeline's configurable retrieval
threshold.

## Boundaries and validation

Evaluation performs live Gemini embedding calls and therefore is excluded from
offline CI. It does not create an HTTP namespace grant or validate a production
deployment. Verify corpus provenance, relevance labels, embedding-space
compatibility, and single-owner operation before using results to make decisions.
