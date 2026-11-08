"""Deterministic repository-native RSI candidate generator; read-only."""
from __future__ import annotations

import argparse
from difflib import unified_diff
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

SCHEMA = "axiom-native-rsi/v1"
REPOSITORY = "axiom-llc/axiom-rag"
RULES = Path(__file__).with_name("generators.json")
SAFE = re.compile(r"^[a-zA-Z0-9_./-]+$")
REV = re.compile(r"^[0-9a-f]{40}$")


class GenerationError(ValueError):
    pass


def _git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if p.returncode:
        raise GenerationError("revision-bound Git source is unavailable")
    return p.stdout.strip()


def validate_rule(rule: dict) -> None:
    if (not isinstance(rule, dict) or set(rule) !=
        {"id", "path", "before", "after", "benchmark", "tests"}):
        raise GenerationError("invalid generator rule")
    path = rule["path"]
    if (not isinstance(path, str) or not SAFE.fullmatch(path)
        or path.startswith("/") or any(part in ("", ".", "..") for part in path.split("/"))
        or not path.endswith(".py") or path.startswith(("tests/", "rsi/", ".github/"))):
        raise GenerationError("unsafe source path")
    for key in ("id", "benchmark"):
        if not isinstance(rule[key], str) or not rule[key]:
            raise GenerationError("invalid rule metadata")
    if (not re.fullmatch(r"[a-z0-9-]{3,60}", rule["id"])
        or not rule["benchmark"].startswith("rsi/")):
        raise GenerationError("invalid rule identity or benchmark")
    if (not isinstance(rule["tests"], list) or not rule["tests"]
        or any(not isinstance(t, str) or not t.startswith("tests/test_")
               or not t.endswith(".py") for t in rule["tests"])):
        raise GenerationError("invalid regression contract")
    old, new = rule["before"], rule["after"]
    if (not isinstance(old, str) or not 4 <= len(old) <= 2000
        or not isinstance(new, str) or len(new) > 4000 or old == new
        or "\x00" in old + new):
        raise GenerationError("invalid replacement")


def propose(source: str, rule: dict, revision: str) -> dict | None:
    """Build a Git-applicable patch from a pinned source, never mutate it."""
    validate_rule(rule)
    if not REV.fullmatch(revision):
        raise GenerationError("unpinned revision")
    before = rule["before"]
    count = source.count(before)
    if count == 0:
        return None
    if count != 1:
        raise GenerationError("ambiguous source anchor")
    updated = source.replace(before, rule["after"], 1)
    if updated == source:
        return None
    patch = "".join(unified_diff(
        source.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="a/" + rule["path"], tofile="b/" + rule["path"], n=3,
    ))
    if not patch or len(patch) > 65536:
        raise GenerationError("invalid or excessive patch")
    compile(updated, rule["path"], "exec")
    stable = sha256((REPOSITORY + "\0" + rule["id"] + "\0" + rule["path"]).encode()).hexdigest()[:20]
    return {
        "candidate_id": "rsi-" + stable,
        "repository": REPOSITORY,
        "source_revision": revision,
        "source_path": rule["path"],
        "source_sha256": sha256(source.encode()).hexdigest(),
        "candidate_sha256": sha256(updated.encode()).hexdigest(),
        "patch_sha256": sha256(patch.encode()).hexdigest(),
        "patch": patch,
        "benchmark": rule["benchmark"],
        "tests": rule["tests"],
        "evaluation": "INDEPENDENT_BENCHMARK_REQUIRED",
        "mutation_authority": False,
    }


def generate(root: Path, revision: str | None = None) -> dict:
    root = root.resolve()
    actual = _git(root, "rev-parse", "HEAD")
    if not REV.fullmatch(actual) or (revision is not None and revision != actual):
        raise GenerationError("source revision mismatch")
    if _git(root, "rev-parse", "--show-toplevel") != str(root):
        raise GenerationError("repository root mismatch")
    manifest = json.loads((root / "rsi" / "generators.json").read_text())
    if (set(manifest) != {"schema", "repository", "rules"}
        or manifest["schema"] != SCHEMA or manifest["repository"] != REPOSITORY
        or not isinstance(manifest["rules"], list) or len(manifest["rules"]) > 16):
        raise GenerationError("invalid native registry")
    seen = set()
    candidates = []
    for rule in manifest["rules"]:
        validate_rule(rule)
        if rule["id"] in seen:
            raise GenerationError("duplicate generator ID")
        seen.add(rule["id"])
        file = root / rule["path"]
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            raise GenerationError("invalid source file")
        if _git(root, "diff", "--name-only", "HEAD", "--", rule["path"]):
            raise GenerationError("dirty source file")
        source = file.read_text(encoding="utf-8")
        if len(source) > 1_000_000:
            raise GenerationError("source too large")
        candidate = propose(source, rule, actual)
        if candidate:
            candidates.append(candidate)
    return {"schema": SCHEMA, "repository": REPOSITORY, "source_revision": actual,
            "status": "GENERATED" if candidates else "NO_APPLICABLE_RULE",
            "candidates": sorted(candidates, key=lambda c: c["candidate_id"]),
            "mutation_authority": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path)
    p.add_argument("--revision")
    a = p.parse_args()
    result = generate(Path(__file__).resolve().parents[1], a.revision)
    if a.output:
        a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: len(v) if k == "candidates" else v for k, v in result.items()
                      if k != "mutation_authority"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
