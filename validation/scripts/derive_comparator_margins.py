#!/usr/bin/env python3
"""Freeze development R-vs-R variability margins before profile ranking."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from gate2_common import file_hash, immutable_json, validate_candidate_manifest

from rapids_choir._calibration import bounded_metric_margin


def _revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_run(value: str) -> tuple[str, int, Path, int, Path]:
    fields = value.split("\t")
    if len(fields) != 5:
        raise ValueError(
            "--run requires DATASET<TAB>SEED_A<TAB>PATH_A<TAB>SEED_B<TAB>PATH_B"
        )
    return fields[0], int(fields[1]), Path(fields[2]), int(fields[3]), Path(fields[4])


def _results_by_pair(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise RuntimeError("reference run contains no results")
    indexed = {str(record["pair_id"]): record for record in results}
    if len(indexed) != len(results):
        raise RuntimeError("reference run contains duplicate pair identifiers")
    return indexed


def derive_margins(
    *,
    root: Path,
    contract_path: Path,
    runs: list[tuple[str, int, Path, int, Path]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    variability = contract["reference_variability"]
    expected_pairs = [tuple(pair) for pair in variability["seed_pairs"]]
    if len(expected_pairs) != 10 or len(set(expected_pairs)) != 10:
        raise RuntimeError("comparator contract requires ten unique seed pairs")
    expected_datasets = list(contract["roles"]["development"])
    expected_grid = {
        (dataset, int(first), int(second))
        for dataset in expected_datasets
        for first, second in expected_pairs
    }
    observed_grid = {(dataset, first, second) for dataset, first, _, second, _ in runs}
    if observed_grid != expected_grid or len(runs) != len(expected_grid):
        raise RuntimeError("reference-variability run grid is incomplete or unexpected")

    deterministic_fields = list(variability["deterministic_fields"])
    bounded_fields = list(variability["bounded_fields"])
    quantile = float(variability["quantile"])
    maximum = float(variability["maximum_margin"])
    deterministic_values: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    bounded_differences: dict[str, list[float]] = defaultdict(list)
    distributions: dict[tuple[str, str], list[float]] = defaultdict(list)
    provenance: list[dict[str, Any]] = []
    pair_sets: dict[str, set[str]] = {}
    dataset_inputs: dict[str, tuple[str, str, str]] = {}
    expected_candidate = {
        "commit": candidate["candidate_commit"],
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": candidate["manifest_sha256"],
        "dirty_state": candidate["dirty_state"],
    }

    for dataset, seed_a, path_a, seed_b, path_b in sorted(
        runs, key=lambda value: (value[0], value[1], value[3])
    ):
        payloads = []
        for seed, path in ((seed_a, path_a), (seed_b, path_b)):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS":
                raise RuntimeError(f"reference run is not PASS: {path}")
            if payload.get("dataset_id") != dataset or payload.get("seed") != seed:
                raise RuntimeError(f"reference run identity mismatch: {path}")
            if payload.get("candidate") != expected_candidate:
                raise RuntimeError(f"reference run candidate mismatch: {path}")
            input_identity = (
                str(payload.get("h5ad_sha256")),
                str(payload.get("pair_manifest_sha256")),
                str(payload.get("reference_source_sha256")),
            )
            if dataset in dataset_inputs and dataset_inputs[dataset] != input_identity:
                raise RuntimeError(f"reference input drift across seeds for {dataset}")
            dataset_inputs[dataset] = input_identity
            payloads.append(payload)
        first_results = _results_by_pair(payloads[0])
        second_results = _results_by_pair(payloads[1])
        if set(first_results) != set(second_results):
            raise RuntimeError(f"reference pair-set drift for {dataset}")
        if dataset in pair_sets and pair_sets[dataset] != set(first_results):
            raise RuntimeError(f"reference pair-set drift across seeds for {dataset}")
        pair_sets[dataset] = set(first_results)

        for pair_id in sorted(first_results):
            left = first_results[pair_id]
            right = second_results[pair_id]
            for field in deterministic_fields:
                deterministic_values[(dataset, pair_id, field)].update(
                    (
                        json.dumps(left.get(field), sort_keys=True),
                        json.dumps(right.get(field), sort_keys=True),
                    )
                )
            for field in bounded_fields:
                left_value = left.get(field)
                right_value = right.get(field)
                if left_value is None and right_value is None:
                    continue
                if left_value is None or right_value is None:
                    deterministic_values[(dataset, pair_id, f"missing:{field}")].update(
                        (json.dumps(left_value), json.dumps(right_value))
                    )
                    continue
                bounded_differences[field].append(
                    abs(float(left_value) - float(right_value))
                )
            for record in (left, right):
                value = record.get("mean_observed_accuracy")
                if value is not None:
                    distributions[(dataset, pair_id)].append(float(value))
        provenance.append(
            {
                "dataset_id": dataset,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "run_a": str(path_a.resolve()),
                "run_a_sha256": file_hash(path_a),
                "run_b": str(path_b.resolve()),
                "run_b_sha256": file_hash(path_b),
            }
        )

    deterministic_checks = {
        f"{dataset}/{pair_id}/{field}": len(values) == 1
        for (dataset, pair_id, field), values in sorted(deterministic_values.items())
    }
    margin_records: dict[str, dict[str, float | bool]] = {}
    for field in bounded_fields:
        differences = bounded_differences.get(field, [])
        if differences:
            margin_records[field] = bounded_metric_margin(
                differences,
                quantile=quantile,
                maximum=maximum,
            )
        else:
            margin_records[field] = {
                "raw_margin": 0.0,
                "margin": 0.0,
                "eligible": True,
            }
    eligible = all(deterministic_checks.values()) and all(
        bool(record["eligible"]) for record in margin_records.values()
    )
    reference_pairs = []
    for dataset in expected_datasets:
        for pair_id in sorted(pair_sets[dataset]):
            values = distributions.get((dataset, pair_id), [])
            reference_pairs.append(
                {
                    "dataset_id": dataset,
                    "pair_id": pair_id,
                    "mean_observed_accuracy_distribution": values,
                    "runs_with_numeric_accuracy": len(values),
                }
            )
    return {
        "schema_version": 1,
        "status": "FROZEN" if eligible else "INELIGIBLE",
        "eligible": eligible,
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_commit": _revision(root),
        "candidate": {
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": candidate["manifest_sha256"],
            "dirty_state": candidate["dirty_state"],
        },
        "contract": str(contract_path),
        "contract_sha256": file_hash(contract_path),
        "quantile": quantile,
        "maximum_margin": maximum,
        "deterministic_checks": deterministic_checks,
        "bounded_metric_margins": margin_records,
        "reference_distributions": reference_pairs,
        "runs": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = validate_candidate_manifest(root, args.candidate_manifest)
    candidate = {
        **candidate,
        "manifest_sha256": file_hash(args.candidate_manifest),
    }
    payload = derive_margins(
        root=root,
        contract_path=args.contract,
        runs=[parse_run(value) for value in args.run],
        candidate=candidate,
    )
    immutable_json(args.output, payload)


if __name__ == "__main__":
    main()
