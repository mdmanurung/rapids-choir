#!/usr/bin/env python3
"""Rank all 12 development profiles and freeze the unique winner."""

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
from gate2_common import atomic_json, file_hash

from rapids_choir._calibration import (
    macro_decision_concordance,
    majority_decision,
    rank_profiles,
)


def source_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def aggregate_profile(frame: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    aggregated = []
    group_columns = ["dataset_id", "pair_id"]
    for (dataset_id, pair_id), group in frame.groupby(group_columns, sort=True):
        if sorted(group["repeat"].tolist()) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"incomplete repeats for {dataset_id}/{pair_id}")
        aggregated.append(
            {
                "dataset_id": dataset_id,
                "pair_id": pair_id,
                "reference": group["reference_decision"].iloc[0],
                "predicted": majority_decision(group["gpu_decision"].tolist()),
                "reference_accuracy": float(
                    group["reference_mean_observed_accuracy"].iloc[0]
                ),
                "gpu_accuracy": float(median(group["gpu_mean_observed_accuracy"])),
            }
        )
    reference = [row["reference"] for row in aggregated]
    predicted = [row["predicted"] for row in aggregated]
    differences = np.asarray(
        [
            abs(row["gpu_accuracy"] - row["reference_accuracy"])
            for row in aggregated
        ],
        dtype=float,
    )
    per_dataset = {}
    for dataset_id in sorted({row["dataset_id"] for row in aggregated}):
        selected = [row for row in aggregated if row["dataset_id"] == dataset_id]
        per_dataset[dataset_id] = {
            "pair_count": len(selected),
            "decision_concordance": sum(
                row["reference"] == row["predicted"] for row in selected
            )
            / len(selected),
        }
    peak = frame["peak_device_bytes"].fillna(0).astype(int)
    estimated = frame["estimated_device_bytes"].fillna(0).astype(int)
    memory = np.maximum(peak.to_numpy(), estimated.to_numpy())
    return {
        "profile_id": frame["profile_id"].iloc[0],
        "profile": profile,
        "pair_count": len(aggregated),
        "macro_decision_concordance": macro_decision_concordance(
            reference, predicted
        ),
        "overall_decision_concordance": sum(
            left == right for left, right in zip(reference, predicted, strict=True)
        )
        / len(reference),
        "median_abs_observed_accuracy_difference": float(np.median(differences)),
        "p95_abs_observed_accuracy_difference": float(
            np.quantile(differences, 0.95, method="linear")
        ),
        "median_runtime_seconds": float(np.median(frame["runtime_seconds"])),
        "median_peak_device_bytes": int(np.median(memory)),
        "per_dataset": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--conda-lock", required=True, type=Path)
    parser.add_argument("--pip-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aggregate-parquet", required=True, type=Path)
    args = parser.parse_args()

    if len(args.run) != 12:
        raise RuntimeError("calibration requires exactly 12 RF profile runs")
    summaries = []
    frames = []
    run_provenance = []
    all_inputs: dict[str, dict[str, Any]] = {}
    for value in args.run:
        summary_path_text, parquet_path_text = value.split("\t")
        summary_path = Path(summary_path_text)
        parquet_path = Path(parquet_path_text)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError(f"profile run is not PASS: {summary_path}")
        if file_hash(parquet_path) != summary["results_parquet_sha256"]:
            raise RuntimeError(f"profile Parquet checksum mismatch: {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        frames.append(frame)
        summaries.append(aggregate_profile(frame, summary["profile"]))
        run_provenance.append(
            {
                "profile_id": summary["profile_id"],
                "summary_sha256": file_hash(summary_path),
                "results_parquet_sha256": file_hash(parquet_path),
                "source_commit": summary["source_commit"],
                "slurm": summary["slurm"],
                "device": summary["device"],
                "environment": summary["environment"],
            }
        )
        for record in summary["inputs"]:
            all_inputs[record["dataset_id"]] = record
    ranked = rank_profiles(summaries)
    profile_ids = {summary["profile_id"] for summary in ranked}
    if len(profile_ids) != 12:
        raise RuntimeError("calibration profile identifiers are not unique")
    commits = {record["source_commit"] for record in run_provenance}
    current_commit = source_commit(args.root)
    if commits != {current_commit}:
        raise RuntimeError(
            f"profile runs do not share current source commit: {sorted(commits)}"
        )

    combined = pd.concat(frames, ignore_index=True)
    args.aggregate_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.aggregate_parquet.with_suffix(
        args.aggregate_parquet.suffix + ".tmp"
    )
    combined.to_parquet(temporary, index=False)
    temporary.replace(args.aggregate_parquet)
    payload = {
        "schema_version": 1,
        "gate": 2,
        "scope": "development_rf_calibration",
        "status": "FROZEN",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": current_commit,
        "rules": "choir_v0_3_0",
        "seeds": {"reference_rf": 1, "gpu_repeats": [1, 1, 1, 1, 1]},
        "ranking_order": [
            "macro_decision_concordance_desc",
            "median_abs_observed_accuracy_difference_asc",
            "p95_abs_observed_accuracy_difference_asc",
            "median_runtime_seconds_asc",
            "median_peak_device_bytes_asc",
        ],
        "winner": ranked[0],
        "rankings": ranked,
        "inputs": [all_inputs[key] for key in sorted(all_inputs)],
        "environment_locks": {
            args.conda_lock.as_posix(): file_hash(args.conda_lock),
            args.pip_lock.as_posix(): file_hash(args.pip_lock),
        },
        "calibration_results": {
            "path": str(args.aggregate_parquet.resolve()),
            "rows": len(combined),
            "sha256": file_hash(args.aggregate_parquet),
        },
        "runs": run_provenance,
    }
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
