"""Independent fixed-fixture paired baseline/candidate microbenchmark.

This is a machine-side fixture, not production workload or business ROI.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
import time

from rsi.native import generate


def load(source: str):
    namespace = {"__name__": "rsi_chunker_benchmark"}
    exec(compile(source, "rag/chunker.py", "exec"), namespace)
    return namespace["chunk_fixed"]


def benchmark(root: Path) -> dict:
    result = generate(root)
    if len(result["candidates"]) != 1:
        return {"status": "NO_APPLICABLE_RULE", "candidates": len(result["candidates"])}
    rule = json.loads((root / "rsi/generators.json").read_text())["rules"][0]
    source = (root / rule["path"]).read_text(encoding="utf-8")
    updated = source.replace(rule["before"], rule["after"], 1)
    a, b = load(source), load(updated)
    test_cases = [
        ("", 8, 0), ("hello", 8, 2), ("one two three", 2, 1),
        (" ".join(f"w{i}" for i in range(103)), 16, 15),
        (" ".join(f"word{i%83}" for i in range(12000)), 128, 32),
    ]
    for text, size, overlap in test_cases:
        if a(text, size, overlap) != b(text, size, overlap):
            raise ValueError("candidate changes public chunker semantics")
    for size, overlap in ((0, 0), (1, 1), (3, -1)):
        try:
            a("a b", size, overlap)
        except ValueError as e:
            baseline_error = str(e)
        else:
            raise ValueError("baseline did not reject invalid input")
        try:
            b("a b", size, overlap)
        except ValueError as e:
            if baseline_error != str(e):
                raise ValueError("candidate changes error behavior")
        else:
            raise ValueError("candidate bypasses invalid input")
    sample = test_cases[-1]
    trials = []
    # Interleave A/B and B/A to reduce fixed-order measurement bias.
    for i in range(11):
        measurements = {}
        for label, func in ((("base", a), ("candidate", b)) if i % 2 == 0
                            else (("candidate", b), ("base", a))):
            start = time.perf_counter_ns()
            for _ in range(100):
                func(*sample)
            measurements[label] = (time.perf_counter_ns() - start) / 100
        trials.append(measurements)
    baseline = median(t["base"] for t in trials)
    candidate = median(t["candidate"] for t in trials)
    gain = baseline - candidate
    return {
        "schema": "axiom-native-rsi/benchmark-v1",
        "source_revision": result["source_revision"],
        "candidate_id": result["candidates"][0]["candidate_id"],
        "source_sha256": sha256(source.encode()).hexdigest(),
        "candidate_sha256": sha256(updated.encode()).hexdigest(),
        "representative_input": "12000 whitespace-separated tokens, window 128, overlap 32",
        "workload_provenance": "deterministic synthetic fixture",
        "holdout_cases": len(test_cases),
        "paired_samples": len(trials),
        "operations_per_sample": 100,
        "baseline_median_ns_per_op": baseline,
        "candidate_median_ns_per_op": candidate,
        "paired_median_gain_ns_per_op": median(t["base"] - t["candidate"] for t in trials),
        "gain_percentage": 100 * gain / baseline if baseline > 0 else None,
        "quality_equivalent": True,
        "positive_machine_gain": gain > 0,
        "promotion_authorized": False,
        "status": "POSITIVE_MACHINE_MEASUREMENT" if gain > 0 else "NONPOSITIVE_MEASUREMENT",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = benchmark(Path(__file__).resolve().parents[1])
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
