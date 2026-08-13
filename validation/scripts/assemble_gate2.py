#!/usr/bin/env python3
"""Aggregate sealed holdout repeats and evaluate every Gate 2 threshold."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import yaml
from gate2_common import atomic_json, file_hash

from rapids_choir._calibration import (
    comparator_acceptance_checks,
    majority_decision,
    wilson_lower_bound,
)


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def aggregate_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (dataset_id, pair_id), group in frame.groupby(
        ["dataset_id", "pair_id"], sort=True
    ):
        if sorted(group["repeat"].tolist()) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"incomplete holdout repeats for {dataset_id}/{pair_id}")
        reference_decisions = set(group["reference_decision"])
        reference_accuracies = set(group["reference_mean_observed_accuracy"])
        if len(reference_decisions) != 1 or len(reference_accuracies) != 1:
            raise RuntimeError(
                f"reference drift across repeats for {dataset_id}/{pair_id}"
            )
        gpu_accuracy = float(median(group["gpu_mean_observed_accuracy"]))
        reference_accuracy = float(next(iter(reference_accuracies)))
        reference_decision = str(next(iter(reference_decisions)))
        gpu_decision = majority_decision(group["gpu_decision"].tolist())
        records.append(
            {
                "dataset_id": dataset_id,
                "pair_id": pair_id,
                "pair_kind": group["pair_kind"].iloc[0],
                "reference_decision": reference_decision,
                "gpu_majority_decision": gpu_decision,
                "decision_concordant": gpu_decision == reference_decision,
                "reference_mean_observed_accuracy": reference_accuracy,
                "gpu_median_observed_accuracy": gpu_accuracy,
                "abs_observed_accuracy_difference": abs(
                    gpu_accuracy - reference_accuracy
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset-spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--profile-seal", required=True, type=Path)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--run-parquet", required=True, type=Path)
    parser.add_argument("--aggregate-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec = yaml.safe_load(args.dataset_spec.read_text(encoding="utf-8"))
    thresholds = spec["acceptance"]["comparator"]
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    seal = json.loads(args.profile_seal.read_text(encoding="utf-8"))
    summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    if profile.get("status") != "FROZEN" or seal.get("status") != "PASS":
        raise RuntimeError("holdout evidence is not protected by a frozen profile seal")
    profile_hash = file_hash(args.profile)
    if seal["profile_sha256"] != profile_hash:
        raise RuntimeError("sealed profile checksum does not match holdout run")
    if summary.get("profile_evidence", {}).get("sha256") != profile_hash:
        raise RuntimeError("holdout GPU run did not use the sealed profile")
    if file_hash(args.run_parquet) != summary["results_parquet_sha256"]:
        raise RuntimeError("holdout result Parquet checksum mismatch")
    if "L40S" not in summary["device"]["name"]:
        raise RuntimeError("Gate 2 acceptance must be measured on a typed L40S")

    raw = pd.read_parquet(args.run_parquet)
    aggregated = aggregate_pairs(raw)
    pair_count = len(aggregated)
    successes = int(aggregated["decision_concordant"].sum())
    overall = successes / pair_count
    per_dataset = {
        str(dataset_id): float(group["decision_concordant"].mean())
        for dataset_id, group in aggregated.groupby("dataset_id", sort=True)
    }
    differences = aggregated["abs_observed_accuracy_difference"].to_numpy(
        dtype=float
    )
    median_difference = float(np.median(differences))
    p95_difference = float(np.quantile(differences, 0.95, method="linear"))
    maximum_transfer = int(
        raw["unapproved_post_ingest_transfer_bytes"].max()
    )
    free = raw["free_bytes_at_preflight"].to_numpy(dtype=float)
    estimated = raw["estimated_device_bytes"].to_numpy(dtype=float)
    if np.any(free <= 0):
        raise RuntimeError("invalid free-memory evidence")
    minimum_headroom = float(np.min((free - estimated) / free))
    wilson = wilson_lower_bound(successes, pair_count)
    checks = comparator_acceptance_checks(
        overall_concordance=overall,
        per_dataset_concordance=per_dataset,
        wilson_lower_95=wilson,
        median_accuracy_difference=median_difference,
        p95_accuracy_difference=p95_difference,
        maximum_unapproved_post_ingest_transfer_bytes=maximum_transfer,
        minimum_vram_headroom=minimum_headroom,
        thresholds=thresholds,
    )
    checks["typed_l40s"] = "L40S" in summary["device"]["name"]
    checks["five_repeats_per_pair"] = bool(
        (raw.groupby(["dataset_id", "pair_id"]).size() == 5).all()
    )
    status = "PASS" if all(checks.values()) else "FAIL"

    args.aggregate_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.aggregate_parquet.with_suffix(
        args.aggregate_parquet.suffix + ".tmp"
    )
    aggregated.to_parquet(temporary, index=False)
    temporary.replace(args.aggregate_parquet)
    payload = {
        "schema_version": 1,
        "gate": 2,
        "scope": "sealed_holdout_comparator_acceptance",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": revision(args.root),
        "rf_profile": {
            "sha256": profile_hash,
            "profile_commit": seal["profile_commit"],
            "winner": profile["winner"],
        },
        "holdout_pair_count": pair_count,
        "metrics": {
            "overall_decision_concordance": overall,
            "per_dataset_decision_concordance": per_dataset,
            "overall_wilson_lower_95": wilson,
            "median_abs_observed_accuracy_difference": median_difference,
            "p95_abs_observed_accuracy_difference": p95_difference,
            "maximum_unapproved_post_ingest_transfer_bytes": maximum_transfer,
            "minimum_vram_headroom": minimum_headroom,
        },
        "thresholds": thresholds,
        "checks": checks,
        "failure_reasons": sorted(
            name for name, passed in checks.items() if not passed
        ),
        "detailed_results": {
            "path": str(args.aggregate_parquet.resolve()),
            "rows": len(aggregated),
            "sha256": file_hash(args.aggregate_parquet),
        },
        "raw_gpu_results": {
            "path": str(args.run_parquet.resolve()),
            "rows": len(raw),
            "sha256": file_hash(args.run_parquet),
        },
        "inputs": summary["inputs"],
        "slurm": summary["slurm"],
        "device": summary["device"],
        "environment": summary["environment"],
        "promotion": (
            "Gate 3 planning permitted; hierarchy implementation not started."
            if status == "PASS"
            else "Gate 2 stopped; no retuning or Gate 3 work is permitted."
        ),
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
