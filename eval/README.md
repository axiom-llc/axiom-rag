# Retrieval Evaluation

Measures retrieval quality against a ground-truth dataset using two metrics:

- **Precision@k** — fraction of top-k retrieved chunks whose `doc_id` is in the ground-truth relevant set
- **MRR (Mean Reciprocal Rank)** — 1 / rank of the first relevant chunk; measures how early a relevant result surfaces

---

## Setup

Ingest your documents before running eval:

```bash
rag ingest ./docs
```

The eval script queries the same ChromaDB collection used by the pipeline.
`GEMINI_API_KEY` is required (embedding calls are live).

---

## Usage

```bash
# Default: precision@5, tabular output
python eval/eval_retrieval.py --dataset eval/dataset.json

# Adjust k
python eval/eval_retrieval.py --dataset eval/dataset.json --top-k 3

# JSON output (for scripting / CI integration)
python eval/eval_retrieval.py --dataset eval/dataset.json --json

# Non-default collection
python eval/eval_retrieval.py \
    --dataset eval/dataset.json \
    --chroma-path ~/.rag/chroma \
    --collection my-collection
```

### Example output

```
Query                                       P@5       RR
------------------------------------------------------
what is the refund policy?                  0.600     1.000
how do I cancel my subscription?            0.400     0.500
what data is collected about users?         0.800     1.000
------------------------------------------------------
Mean Precision@5: 0.600
MRR:              0.833
```

---

## Dataset format

`eval/dataset.json` is a list of query/relevant-doc-id pairs:

```json
[
  {
    "query": "what is the refund policy?",
    "relevant_doc_ids": ["refund-policy"]
  }
]
```

`relevant_doc_ids` should match the `doc_id` values used during `rag ingest`.
Replace the sample entries with queries and doc IDs from your own collection.

---

## Tuning guidance

| Metric is low | Likely cause | Adjustment |
|---|---|---|
| Precision@k | Score threshold too aggressive | Lower `RAG_SCORE_THRESHOLD` |
| MRR | Relevant chunks buried | Increase `RAG_TOP_K`; review chunk size |
| Both | Embedding mismatch | Verify `task_type` separation in `embedder.py` |

---

## Notes

- Score threshold is set to `0.0` during eval to capture full ranking. The production threshold (`RAG_SCORE_THRESHOLD`) is applied by the pipeline at query time, not here.
- Eval is intentionally omitted from CI — it requires a live Gemini API key and an ingested collection. Run it manually when tuning retrieval parameters.
