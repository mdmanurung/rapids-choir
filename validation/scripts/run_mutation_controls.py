#!/usr/bin/env python3
"""Run all predeclared faults as temporary source-level mutations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import immutable_json

EXPECTED_MUTATIONS = {
    "tail_test_inequality",
    "decision_precedence",
    "batch_eligibility",
    "multiplicity_correction",
    "seed_derivation",
    "sampling_order",
    "resolution_stopping",
    "bridge_selection",
    "singleton_relabelling",
    "pruning_order",
    "membership_cache_hashing",
    "stale_artifact_detection",
}
EXPECTED_CATEGORIES = {
    "tail_test",
    "batch_and_multiplicity",
    "seed_and_sampling",
    "resolution_and_bridge",
    "singleton_and_pruning",
    "identity_and_staleness",
}
REQUIRED_FIELDS = {
    "mutation_id",
    "category",
    "target",
    "source",
    "find",
    "replace",
    "probe",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    package = root / "src/rapids_choir"
    return {
        str(path.relative_to(root)): _sha256(path.read_bytes())
        for path in sorted(package.rglob("*.py"))
    }


def _validate_manifest(value: Any, root: Path) -> list[dict[str, str]]:
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("unsupported source-mutation manifest")
    if value.get("strategy") != "temporary_source_patch":
        raise ValueError("mutation strategy must be temporary_source_patch")
    raw_mutations = value.get("mutations")
    if not isinstance(raw_mutations, list):
        raise ValueError("mutation manifest records must be a list")
    if any(not isinstance(record, dict) for record in raw_mutations):
        raise ValueError("mutation manifest record must be an object")
    mutations = [dict(record) for record in raw_mutations]
    if any(set(record) != REQUIRED_FIELDS for record in mutations):
        raise ValueError("source mutation record fields are incomplete or unknown")
    mutation_ids = {record["mutation_id"] for record in mutations}
    categories = {record["category"] for record in mutations}
    if mutation_ids != EXPECTED_MUTATIONS:
        raise ValueError("predeclared mutation set is incomplete or has drifted")
    if categories != EXPECTED_CATEGORIES or set(
        value.get("required_categories", [])
    ) != EXPECTED_CATEGORIES:
        raise ValueError("predeclared mutation categories are incomplete")
    for record in mutations:
        source = root / record["source"]
        probe = record["probe"].split("::", 1)[0]
        if not source.is_file() or not (root / probe).is_file():
            raise ValueError(
                f"source mutation path is missing for {record['mutation_id']}"
            )
        if not record["find"] or record["find"] == record["replace"]:
            raise ValueError(
                f"source mutation is empty for {record['mutation_id']}"
            )
    return mutations


def _run_one(root: Path, temporary: Path, record: dict[str, str]) -> dict[str, Any]:
    mutation_id = record["mutation_id"]
    mutation_root = temporary / mutation_id
    package_root = mutation_root / "src/rapids_choir"
    shutil.copytree(root / "src/rapids_choir", package_root)
    relative = Path(record["source"]).relative_to("src/rapids_choir")
    source = package_root / relative
    before = source.read_bytes()
    text = before.decode("utf-8")
    if text.count(record["find"]) != 1:
        raise ValueError(
            f"mutation anchor must occur exactly once for {mutation_id}: "
            f"found {text.count(record['find'])}"
        )
    mutated = text.replace(record["find"], record["replace"], 1).encode("utf-8")
    source.write_bytes(mutated)
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    paths = [str(mutation_root / "src"), str(root)]
    if existing:
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            record["probe"],
            "--disable-warnings",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    # Pytest exit code 1 means the selected probe ran and exposed the fault.
    # Collection, usage, import, or internal errors must not count as kills.
    detected = completed.returncode == 1
    return {
        "mutation_id": mutation_id,
        "category": record["category"],
        "target": record["target"],
        "source": record["source"],
        "source_sha256_before": _sha256(before),
        "source_sha256_after": _sha256(mutated),
        "probe": record["probe"],
        "probe_returncode": completed.returncode,
        "detected": detected,
        "stdout_sha256": _sha256(completed.stdout.encode("utf-8")),
        "stderr_sha256": _sha256(completed.stderr.encode("utf-8")),
    }


def run_controls(manifest_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (
        Path(__file__).resolve().parents[2] if root is None else root.resolve()
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutations = _validate_manifest(manifest, repository)
    source_hashes_before = _source_hashes(repository)
    with tempfile.TemporaryDirectory(prefix="rapids-choir-source-mutations-") as raw:
        temporary = Path(raw)
        results = [_run_one(repository, temporary, record) for record in mutations]
    source_hashes_after = _source_hashes(repository)
    source_tree_unchanged = source_hashes_before == source_hashes_after
    detected = sum(bool(result["detected"]) for result in results)
    return {
        "schema_version": 2,
        "strategy": "temporary_source_patch",
        "status": (
            "PASS"
            if detected == len(results) and source_tree_unchanged
            else "FAIL"
        ),
        "mutation_count": len(results),
        "detected_count": detected,
        "source_tree_unchanged": source_tree_unchanged,
        "source_tree_sha256_before": _sha256(
            json.dumps(source_hashes_before, sort_keys=True).encode("utf-8")
        ),
        "source_tree_sha256_after": _sha256(
            json.dumps(source_hashes_after, sort_keys=True).encode("utf-8")
        ),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("validation/contracts/parity_mutations_v1.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_controls(arguments.manifest, root=arguments.root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        immutable_json(arguments.output, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
