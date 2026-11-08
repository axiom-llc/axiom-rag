"""Native RSI candidate generation, isolation, provenance and rollback regression."""
import json
from pathlib import Path
import subprocess
import tempfile

import pytest
from rsi.native import GenerationError, generate, propose, validate_rule

ROOT = Path(__file__).resolve().parents[1]
RULE = json.loads((ROOT / "rsi/generators.json").read_text())["rules"][0]
SOURCE = (ROOT / RULE["path"]).read_text(encoding="utf-8")
REV = "a" * 40


def test_revision_bound_executable_generation():
    result = generate(ROOT)
    assert result["repository"] == "axiom-llc/axiom-rag"
    assert not result["mutation_authority"]
    assert len(result["candidates"]) == 1
    candidate = result["candidates"][0]
    assert candidate["source_revision"] == result["source_revision"]
    assert candidate["source_path"] == RULE["path"]
    assert candidate["patch"].startswith("--- a/rag/chunker.py\n")
    assert not (ROOT / RULE["path"]).is_symlink()


def test_stable_identity_and_idempotent_saturation():
    a = propose(SOURCE, RULE, REV)
    b = propose(SOURCE, RULE, REV)
    assert a == b
    after = SOURCE.replace(RULE["before"], RULE["after"], 1)
    assert propose(after, RULE, REV) is None


def test_fail_closed_ambiguity_and_unsafe_targets():
    with pytest.raises(GenerationError, match="ambiguous"):
        propose(SOURCE + RULE["before"], RULE, REV)
    for path in ("../escape.py", "/absolute.py", "tests/test_x.py", "rsi/native.py"):
        with pytest.raises(GenerationError, match="unsafe"):
            validate_rule({**RULE, "path": path})
    with pytest.raises(GenerationError, match="unpinned"):
        propose(SOURCE, RULE, "main")
    with pytest.raises(GenerationError, match="revision mismatch"):
        generate(ROOT, REV)


def _run(*command, cwd=None, input=None):
    return subprocess.run(command, cwd=cwd, input=input, capture_output=True,
                          text=True, check=True)


def test_git_apply_reverse_restores_identical_original_bytes():
    patch = propose(SOURCE, RULE, REV)["patch"]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        file = root / RULE["path"]
        file.parent.mkdir(parents=True)
        file.write_text(SOURCE)
        _run("git", "init", "-q", cwd=root)
        _run("git", "add", ".", cwd=root)
        _run("git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
             "commit", "-qm", "baseline", cwd=root)
        initial = file.read_bytes()
        _run("git", "apply", "--check", "-", cwd=root, input=patch)
        _run("git", "apply", "-", cwd=root, input=patch)
        assert file.read_text() == SOURCE.replace(RULE["before"], RULE["after"], 1)
        _run("git", "apply", "-R", "--check", "-", cwd=root, input=patch)
        _run("git", "apply", "-R", "-", cwd=root, input=patch)
        assert file.read_bytes() == initial


def test_independent_synthetic_holdout_has_equivalent_output():
    from rsi.benchmark import benchmark
    report = benchmark(ROOT)
    assert report["quality_equivalent"]
    assert report["holdout_cases"] == 5
    assert report["paired_samples"] == 11
    assert not report["promotion_authorized"]
    assert report["source_revision"] == generate(ROOT)["source_revision"]
