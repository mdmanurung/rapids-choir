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
from gate2_common import file_hash, immutable_json

from rapids_choir._calibration import (
    accuracy_distribution_error,
    macro_decision_concordance,
    macro_decision_reason_concordance,
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


def aggregate_profile(
    frame: pd.DataFrame,
    profile: dict[str, Any],
    reference_distributions: dict[tuple[str, str], list[float]],
) -> dict[str, Any]:
    aggregated = []
    group_columns = ["dataset_id", "pair_id"]
    for (dataset_id, pair_id), group in frame.groupby(group_columns, sort=True):
        if sorted(group["repeat"].tolist()) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"incomplete repeats for {dataset_id}/{pair_id}")
        if len(set(group["reference_decision"])) != 1 or len(
            set(group["reference_reason"])
        ) != 1:
            raise RuntimeError(f"reference outcome drift for {dataset_id}/{pair_id}")
        reference_value = group["reference_mean_observed_accuracy"].iloc[0]
        reference_accuracy = (
            None if pd.isna(reference_value) else float(reference_value)
        )
        gpu_values = (
            group["gpu_mean_observed_accuracy"].dropna().astype(float).tolist()
        )
        gpu_accuracy = float(median(gpu_values)) if gpu_values else None
        distribution = reference_distributions[(str(dataset_id), str(pair_id))]
        if bool(gpu_values) != bool(distribution):
            raise RuntimeError(
                f"numeric accuracy presence drift for {dataset_id}/{pair_id}"
            )
        aggregated.append(
            {
                "dataset_id": dataset_id,
                "pair_id": pair_id,
                "reference_accuracy": reference_accuracy,
                "gpu_accuracy": gpu_accuracy,
                "accuracy_distribution_error": (
                    accuracy_distribution_error(distribution, gpu_values)
                    if gpu_values
                    else None
                ),
            }
        )
    # Every stochastic repeat contributes to the primary fidelity criterion.
    # A majority vote here would conceal unstable profiles before selection.
    reference = frame["reference_decision"].astype(str).tolist()
    predicted = frame["gpu_decision"].astype(str).tolist()
    reference_reasons = frame["reference_reason"].astype(str).tolist()
    predicted_reasons = frame["gpu_reference_reason"].astype(str).tolist()
    differences = np.asarray(
        [
            abs(row["gpu_accuracy"] - row["reference_accuracy"])
            for row in aggregated
            if row["gpu_accuracy"] is not None
            and row["reference_accuracy"] is not None
        ],
        dtype=float,
    )
    if differences.size == 0:
        raise RuntimeError("profile has no numeric pair accuracy differences")
    per_dataset = {}
    for dataset_id in sorted({row["dataset_id"] for row in aggregated}):
        selected = frame[frame["dataset_id"].astype(str) == str(dataset_id)]
        per_dataset[dataset_id] = {
            "pair_count": sum(
                row["dataset_id"] == dataset_id for row in aggregated
            ),
            "repeat_record_count": len(selected),
            "decision_concordance": float(
                selected["reference_decision"]
                .astype(str)
                .eq(selected["gpu_decision"].astype(str))
                .mean()
            ),
        }
    peak = frame["peak_device_bytes"].fillna(0).astype(int)
    estimated = frame["estimated_device_bytes"].fillna(0).astype(int)
    memory = np.maximum(peak.to_numpy(), estimated.to_numpy())
    distribution_errors = [
        row["accuracy_distribution_error"]
        for row in aggregated
        if row["accuracy_distribution_error"] is not None
    ]
    if not distribution_errors:
        raise RuntimeError("profile has no numeric accuracy distributions")
    return {
        "profile_id": frame["profile_id"].iloc[0],
        "profile": profile,
        "pair_count": len(aggregated),
        "repeat_record_count": len(frame),
        "repeat_evaluation": "every_repeat",
        "macro_decision_concordance": macro_decision_concordance(
            reference, predicted
        ),
        "macro_decision_reason_concordance": (
            macro_decision_reason_concordance(
                reference,
                predicted,
                reference_reasons,
                predicted_reasons,
            )
        ),
        "overall_decision_concordance": sum(
            left == right for left, right in zip(reference, predicted, strict=True)
        )
        / len(reference),
        "median_abs_observed_accuracy_difference": float(np.median(differences)),
        "p95_abs_observed_accuracy_difference": float(
            np.quantile(differences, 0.95, method="linear")
        ),
        "median_accuracy_distribution_error": float(
            np.median(distribution_errors)
        ),
        "median_runtime_seconds": float(np.median(frame["runtime_seconds"])),
        "median_peak_device_bytes": int(np.median(memory)),
        "per_dataset": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--margin-contract", required=True, type=Path)
    parser.add_argument("--comparator-contract", required=True, type=Path)
    parser.add_argument("--conda-lock", required=True, type=Path)
    parser.add_argument("--pip-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--aggregate-parquet", required=True, type=Path)
    args = parser.parse_args()

    if len(args.run) != 12:
        raise RuntimeError("calibration requires exactly 12 RF profile runs")
    margin_contract = json.loads(args.margin_contract.read_text(encoding="utf-8"))
    if margin_contract.get("status") != "FROZEN" or not margin_contract.get(
        "eligible"
    ):
        raise RuntimeError("reference variability makes every RF profile ineligible")
    if margin_contract.get("contract_sha256") != file_hash(
        args.comparator_contract
    ):
        raise RuntimeError("margin and comparator contract checksums do not agree")
    reference_distributions = {
        (str(record["dataset_id"]), str(record["pair_id"])): list(
            record["mean_observed_accuracy_distribution"]
        )
        for record in margin_contract["reference_distributions"]
    }

    summaries = []
    frames = []
    run_provenance = []
    calibration_candidates: list[dict[str, Any]] = []
    resource_reservations: list[dict[str, Any]] = []
    all_inputs: dict[str, dict[str, Any]] = {}
    for value in args.run:
        summary_path_text, parquet_path_text = value.split("\t")
        summary_path = Path(summary_path_text)
        parquet_path = Path(parquet_path_text)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError(f"profile run is not PASS: {summary_path}")
        candidate = summary.get("candidate")
        if not isinstance(candidate, dict):
            raise RuntimeError(f"profile run lacks candidate identity: {summary_path}")
        calibration_candidates.append(candidate)
        reservation = summary.get("resource_reservation")
        if not isinstance(reservation, dict):
            raise RuntimeError(
                f"profile run lacks resource reservation: {summary_path}"
            )
        resource_reservations.append(reservation)
        if file_hash(parquet_path) != summary["results_parquet_sha256"]:
            raise RuntimeError(f"profile Parquet checksum mismatch: {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        frames.append(frame)
        profile_summary = aggregate_profile(
            frame,
            summary["profile"],
            reference_distributions,
        )
        profile_summary["eligible"] = True
        summaries.append(profile_summary)
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
    if margin_contract.get("candidate_commit") != current_commit:
        raise RuntimeError("reference-variability candidate does not match HEAD")
    if commits != {current_commit}:
        raise RuntimeError(
            f"profile runs do not share current source commit: {sorted(commits)}"
        )
    candidate_records = {
        json.dumps(record, sort_keys=True) for record in calibration_candidates
    }
    if len(candidate_records) != 1:
        raise RuntimeError("profile runs do not share one candidate manifest")
    calibration_candidate = calibration_candidates[0]
    reservation_records = {
        json.dumps(record, sort_keys=True) for record in resource_reservations
    }
    if len(reservation_records) != 1:
        raise RuntimeError("profile runs do not share one resource reservation")
    if calibration_candidate.get("runtime_tree_sha256") != margin_contract.get(
        "candidate", {}
    ).get("runtime_tree_sha256"):
        raise RuntimeError("profile runs and reference margins have runtime drift")
    if calibration_candidate.get("manifest_sha256") != margin_contract.get(
        "candidate", {}
    ).get("manifest_sha256"):
        raise RuntimeError("profile runs and reference margins have candidate drift")

    combined = pd.concat(frames, ignore_index=True)
    if args.aggregate_parquet.exists():
        raise RuntimeError(
            f"immutable calibration aggregate already exists: {args.aggregate_parquet}"
        )
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
        "calibration_candidate": calibration_candidate,
        "resource_reservation": resource_reservations[0],
        "rules": "choir_v0_3_0",
        "seeds": {"reference_rf": 1, "gpu_repeats": [1, 1, 1, 1, 1]},
        "ranking_order": [
            "macro_decision_reason_concordance_desc",
            "median_accuracy_distribution_error_asc",
            "median_runtime_seconds_asc",
            "n_bins_asc",
            "max_depth_16_before_unlimited",
            "n_streams_asc",
        ],
        "winner": ranked[0],
        "rankings": ranked,
        "inputs": [all_inputs[key] for key in sorted(all_inputs)],
        "environment_locks": {
            args.conda_lock.as_posix(): file_hash(args.conda_lock),
            args.pip_lock.as_posix(): file_hash(args.pip_lock),
        },
        "comparator_contract": {
            "path": args.comparator_contract.as_posix(),
            "sha256": file_hash(args.comparator_contract),
        },
        "reference_variability": {
            "path": str(args.margin_contract.resolve()),
            "sha256": file_hash(args.margin_contract),
            "status": margin_contract["status"],
            "eligible": margin_contract["eligible"],
            "candidate_commit": margin_contract["candidate_commit"],
            "candidate": margin_contract["candidate"],
            "contract_sha256": margin_contract["contract_sha256"],
            "maximum_margin": margin_contract["maximum_margin"],
            "bounded_metric_margins": margin_contract[
                "bounded_metric_margins"
            ],
        },
        "calibration_results": {
            "path": str(args.aggregate_parquet.resolve()),
            "rows": len(combined),
            "sha256": file_hash(args.aggregate_parquet),
        },
        "runs": run_provenance,
    }
    immutable_json(args.output, payload)


if __name__ == "__main__":
    main()
