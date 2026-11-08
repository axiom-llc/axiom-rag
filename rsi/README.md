# Repository-native RSI: chunk-fixed single-slice

Run `python -m rsi.native --output /tmp/rag-rsi-candidates.json` and `python -m rsi.benchmark --output /tmp/rag-rsi-benchmark.json` from a clean revision-bound checkout. CI executes both on Python 3.11/3.12 and preserves separate artifacts. No production source mutation, network provider call, branch push, or autopromotion occurs.

The sole native deterministic rule removes the redundant second slice in `rag.chunker.chunk_fixed`. The range excludes empty windows for valid input; the candidate is a Git-applicable unified diff with source/candidate/patch hashes and a revision-bound stable identity. Holdouts include empty, short, maximal overlap, invalid arguments and a deterministic 12,000-word paired microbenchmark. This is synthetic machine-performance evidence, not production traffic or business ROI.

Promotion requires a passing full regression suite, independently measured positive net gain on representative workloads, additional holdout coverage as needed, authenticated exact-head CI, and a demonstrated normal-Git revert. The generator only proposes; its artifact does **not** authorize merge or deployment. A negative/noisy benchmark is an admissible research result, not an optimization success. Director federated RSI must treat source registration and native candidate evidence separately.
