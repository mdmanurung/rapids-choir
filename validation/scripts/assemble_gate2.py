#!/usr/bin/env python3
"""Evaluate sealed comparator confirmation against the frozen v2 contract."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import yaml
from gate2_common import file_hash, immutable_json, install_immutable_file

from rapids_choir._calibration import wilson_lower_bound


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _single_string(group: pd.DataFrame, column: str, identity: str) -> str:
    values = set(group[column].astype(str))
    if len(values) != 1:
        raise RuntimeError(f"{column} drift across repeats for {identity}")
    return next(iter(values))


def _numeric_values(group: pd.DataFrame, column: str) -> list[float | None]:
    values: list[float | None] = []
    for value in group[column]:
        if pd.isna(value):
            values.append(None)
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            raise RuntimeError(f"nonfinite bounded metric in {column}")
        values.append(numeric)
    return values


def aggregate_pairs(
    frame: pd.DataFrame,
    bounded_metric_margins: dict[str, float],
) -> pd.DataFrame:
    """Require every repeat to meet exact and bounded-record fidelity."""

    required = {
        "dataset_id",
        "pair_id",
        "repeat",
        "pair_kind",
        "expected_decision",
        "reference_pair_kind",
        "reference_expected_decision",
        "reference_decision",
        "reference_reason",
        "gpu_decision",
        "gpu_reference_reason",
    }
    for field in bounded_metric_margins:
        required.update((f"reference_{field}", f"gpu_{field}"))
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"holdout results lack required v2 fields: {missing}")

    records: list[dict[str, Any]] = []
    for (dataset_id, pair_id), group in frame.groupby(
        ["dataset_id", "pair_id"], sort=True
    ):
        identity = f"{dataset_id}/{pair_id}"
        if sorted(group["repeat"].astype(int).tolist()) != [1, 2, 3, 4, 5]:
            raise RuntimeError(f"incomplete holdout repeats for {identity}")

        pair_kind = _single_string(group, "pair_kind", identity)
        expected_decision = _single_string(group, "expected_decision", identity)
        reference_pair_kind = _single_string(
            group, "reference_pair_kind", identity
        )
        reference_expected_decision = _single_string(
            group, "reference_expected_decision", identity
        )
        reference_decision = _single_string(
            group, "reference_decision", identity
        )
        reference_reason = _single_string(group, "reference_reason", identity)
        metadata_exact = (
            pair_kind == reference_pair_kind
            and expected_decision == reference_expected_decision
        )
        outcome_exact_by_repeat = (
            group["gpu_decision"].astype(str).eq(reference_decision)
            & group["gpu_reference_reason"].astype(str).eq(reference_reason)
        )
        outcome_exact = bool(outcome_exact_by_repeat.all())
        record: dict[str, Any] = {
            "dataset_id": str(dataset_id),
            "pair_id": str(pair_id),
            "pair_kind": pair_kind,
            "expected_decision": expected_decision,
            "reference_decision": reference_decision,
            "reference_reason": reference_reason,
            "metadata_exact": metadata_exact,
            "outcome_exact_all_repeats": outcome_exact,
            "outcome_exact_repeat_count": int(outcome_exact_by_repeat.sum()),
        }

        bounded_checks: list[bool] = []
        for field, margin in bounded_metric_margins.items():
            reference_values = _numeric_values(group, f"reference_{field}")
            if len(set(reference_values)) != 1:
                raise RuntimeError(
                    f"reference {field} drift across repeats for {identity}"
                )
            reference_value = reference_values[0]
            gpu_values = _numeric_values(group, f"gpu_{field}")
            presence_exact = all(
                (value is None) == (reference_value is None)
                for value in gpu_values
            )
            if reference_value is None:
                within_margin = presence_exact
                maximum_difference = None
                gpu_median = None
            elif not presence_exact:
                within_margin = False
                maximum_difference = None
                gpu_median = None
            else:
                numeric_gpu = [value for value in gpu_values if value is not None]
                differences = [
                    abs(value - reference_value) for value in numeric_gpu
                ]
                maximum_difference = max(differences)
                gpu_median = float(median(numeric_gpu))
                within_margin = maximum_difference <= margin
            bounded_checks.append(within_margin)
            record.update(
                {
                    f"reference_{field}": reference_value,
                    f"gpu_median_{field}": gpu_median,
                    f"max_abs_difference_{field}": maximum_difference,
                    f"margin_{field}": margin,
                    f"presence_exact_{field}": presence_exact,
                    f"within_margin_{field}": within_margin,
                }
            )
        record["all_bounded_metrics_within_margin"] = all(bounded_checks)
        record["pair_pass"] = bool(
            metadata_exact and outcome_exact and all(bounded_checks)
        )
        records.append(record)
    if not records:
        raise RuntimeError("holdout results contain no comparator pairs")
    return pd.DataFrame.from_records(records)


def _load_margins(
    contract: dict[str, Any], payload: dict[str, Any]
) -> dict[str, float]:
    if payload.get("status") != "FROZEN" or not payload.get("eligible"):
        raise RuntimeError("development reference variability is not eligible")
    fields = list(contract["reference_variability"]["bounded_fields"])
    records = payload.get("bounded_metric_margins", {})
    if set(records) != set(fields):
        raise RuntimeError("development margin field set drift")
    maximum = float(contract["holdout"]["maximum_margin"])
    margins: dict[str, float] = {}
    for field in fields:
        record = records[field]
        margin = float(record["margin"])
        if not record.get("eligible") or not 0 <= margin <= maximum:
            raise RuntimeError(f"ineligible development margin for {field}")
        margins[field] = margin
    return margins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset-spec", required=True, type=Path)
    parser.add_argument("--comparator-contract", required=True, type=Path)
    parser.add_argument("--authorization-seal", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--profile-reproduction", required=True, type=Path)
    parser.add_argument("--profile-seal", required=True, type=Path)
    parser.add_argument("--run-summary", required=True, type=Path)
    parser.add_argument("--run-parquet", required=True, type=Path)
    parser.add_argument("--aggregate-parquet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec = yaml.safe_load(args.dataset_spec.read_text(encoding="utf-8"))
    operational_thresholds = spec["acceptance"]["comparator"]
    contract = yaml.safe_load(
        args.comparator_contract.read_text(encoding="utf-8")
    )
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    profile_reproduction = json.loads(
        args.profile_reproduction.read_text(encoding="utf-8")
    )
    seal = json.loads(args.profile_seal.read_text(encoding="utf-8"))
    authorization = json.loads(
        args.authorization_seal.read_text(encoding="utf-8")
    )
    summary = json.loads(args.run_summary.read_text(encoding="utf-8"))
    current_commit = revision(args.root)

    if profile.get("status") != "FROZEN" or seal.get("status") != "PASS":
        raise RuntimeError("holdout evidence lacks a frozen profile seal")
    if seal.get("checked_at_head") != current_commit:
        raise RuntimeError("profile seal was not checked at current HEAD")
    if authorization.get("status") != "AUTHORIZED" or authorization.get(
        "target"
    ) != "comparator_confirm":
        raise RuntimeError("comparator confirmation is not authorized")
    if authorization.get("candidate_commit") != current_commit:
        raise RuntimeError("authorization candidate does not match HEAD")
    contract_hash = file_hash(args.comparator_contract)
    if profile.get("comparator_contract", {}).get("sha256") != contract_hash:
        raise RuntimeError("profile and comparator contract checksums differ")
    margin_payload = profile.get("reference_variability", {})
    if margin_payload.get("contract_sha256") != contract_hash:
        raise RuntimeError("profile margins and comparator contract checksums differ")
    calibration_candidate = profile.get("calibration_candidate")
    if not isinstance(calibration_candidate, dict) or not calibration_candidate.get(
        "runtime_tree_sha256"
    ):
        raise RuntimeError("profile lacks its calibration runtime identity")
    reproduction_candidate = profile_reproduction.get("candidate")
    profile_hash = file_hash(args.profile)
    if (
        profile_reproduction.get("status") != "REPRODUCED"
        or profile_reproduction.get("exact_winning_profile") is not True
        or profile_reproduction.get("exact_reference_variability_margins") is not True
        or profile_reproduction.get("protocol_checks")
        != {
            "ranking_order": True,
            "comparator_contract_sha256": True,
            "environment_lock_sha256": True,
        }
        or profile_reproduction.get("frozen", {}).get("sha256") != profile_hash
        or not isinstance(reproduction_candidate, dict)
        or reproduction_candidate.get("commit") != current_commit
    ):
        raise RuntimeError(
            "final candidate did not reproduce the frozen RF profile"
        )
    reproduction_hash = file_hash(args.profile_reproduction)
    if (
        authorization.get("rf_profile_reproduction_sha256")
        != reproduction_hash
    ):
        raise RuntimeError("authorization does not bind profile reproduction")
    reproduced_reference = profile_reproduction.get("reproduced", {})
    reproduced_path = Path(str(reproduced_reference.get("path")))
    if (
        not reproduced_path.is_file()
        or reproduced_reference.get("sha256") != file_hash(reproduced_path)
    ):
        raise RuntimeError("final-candidate RF-profile reproduction is stale")
    confirmation_runtime = authorization.get("runtime_tree_sha256")
    if reproduction_candidate.get("runtime_tree_sha256") != confirmation_runtime:
        raise RuntimeError("profile reproduction and authorization runtime differ")
    if any(
        seal.get("candidate", {}).get(key) != reproduction_candidate.get(key)
        for key in ("commit", "attempt", "runtime_tree_sha256", "manifest_sha256")
    ):
        raise RuntimeError("profile seal and reproduction candidate differ")
    if seal.get("profile_sha256") != profile_hash:
        raise RuntimeError("sealed profile checksum does not match profile")
    if summary.get("profile_evidence", {}).get("sha256") != profile_hash:
        raise RuntimeError("holdout GPU run did not use the sealed profile")
    if summary.get("comparator_contract", {}).get("sha256") != contract_hash:
        raise RuntimeError("holdout GPU run used a different comparator contract")
    if summary.get("authorization", {}).get("sha256") != file_hash(
        args.authorization_seal
    ):
        raise RuntimeError("holdout GPU run used a different authorization seal")
    if summary.get("source_commit") != current_commit:
        raise RuntimeError("holdout GPU run does not match HEAD")
    if summary.get("candidate") != reproduction_candidate:
        raise RuntimeError("holdout GPU candidate differs from profile reproduction")
    if summary.get("candidate", {}).get("attempt") != authorization.get(
        "attempt"
    ):
        raise RuntimeError("holdout GPU attempt differs from authorization")
    if summary.get("resource_reservation", {}).get(
        "reservation_id"
    ) is None:
        raise RuntimeError("holdout GPU run lacks a resource reservation")
    if file_hash(args.run_parquet) != summary.get("results_parquet_sha256"):
        raise RuntimeError("holdout result Parquet checksum mismatch")
    if "L40S" not in summary.get("device", {}).get("name", ""):
        raise RuntimeError("Gate 2 acceptance requires a typed L40S")

    margins = _load_margins(contract, margin_payload)
    raw = pd.read_parquet(args.run_parquet)
    aggregated = aggregate_pairs(raw, margins)
    expected_datasets = set(contract["roles"]["sealed_holdout"])
    observed_datasets = set(aggregated["dataset_id"].astype(str))
    if observed_datasets != expected_datasets:
        raise RuntimeError(
            "holdout dataset set drift: "
            f"missing={sorted(expected_datasets - observed_datasets)}, "
            f"extra={sorted(observed_datasets - expected_datasets)}"
        )

    pair_count = len(aggregated)
    exact_successes = int(aggregated["outcome_exact_all_repeats"].sum())
    pair_successes = int(aggregated["pair_pass"].sum())
    per_dataset: dict[str, dict[str, Any]] = {}
    for dataset_id, group in aggregated.groupby("dataset_id", sort=True):
        per_dataset[str(dataset_id)] = {
            "pair_count": len(group),
            "outcome_exact_fraction": float(
                group["outcome_exact_all_repeats"].mean()
            ),
            "bounded_metric_pass_fraction": float(
                group["all_bounded_metrics_within_margin"].mean()
            ),
            "passed": bool(group["pair_pass"].all()),
        }

    maximum_transfer = int(raw["unapproved_post_ingest_transfer_bytes"].max())
    free = raw["free_bytes_at_preflight"].to_numpy(dtype=float)
    estimated = raw["estimated_device_bytes"].to_numpy(dtype=float)
    if np.any(free <= 0):
        raise RuntimeError("invalid free-memory evidence")
    minimum_headroom = float(np.min((free - estimated) / free))
    checks = {
        "deterministic_fields_exact_every_repeat": bool(
            aggregated["metadata_exact"].all()
            and aggregated["outcome_exact_all_repeats"].all()
        ),
        "bounded_metrics_within_development_margins_every_repeat": bool(
            aggregated["all_bounded_metrics_within_margin"].all()
        ),
        "each_holdout_dataset_passes": all(
            record["passed"] for record in per_dataset.values()
        ),
        "typed_l40s": "L40S" in summary["device"]["name"],
        "five_repeats_per_pair": bool(
            (raw.groupby(["dataset_id", "pair_id"]).size() == 5).all()
        ),
        "unapproved_transfer_within_limit": maximum_transfer
        <= int(operational_thresholds["unapproved_post_ingest_transfer_bytes_max"]),
        "vram_headroom_within_limit": minimum_headroom
        >= float(operational_thresholds["vram_headroom_min"]),
    }
    status = "PASS" if all(checks.values()) else "FAIL"

    args.aggregate_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.aggregate_parquet.with_suffix(
        args.aggregate_parquet.suffix + f".tmp.{os.getpid()}"
    )
    aggregated.to_parquet(temporary, index=False)
    install_immutable_file(temporary, args.aggregate_parquet)
    payload = {
        "schema_version": 2,
        "gate": 2,
        "scope": "sealed_holdout_comparator_component_fidelity",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": current_commit,
        "authorization": {
            "sha256": file_hash(args.authorization_seal),
            "attempt": authorization["attempt"],
            "operator": authorization["operator"],
        },
        "resource_reservation": summary["resource_reservation"],
        "rf_profile": {
            "sha256": profile_hash,
            "profile_commit": seal["profile_commit"],
            "winner": profile["winner"],
            "final_candidate_reproduction_sha256": reproduction_hash,
        },
        "comparator_contract": {
            "sha256": contract_hash,
            "contract_id": contract["contract_id"],
        },
        "development_margins": {
            "source_sha256": margin_payload["sha256"],
            "source_candidate_commit": margin_payload["candidate_commit"],
            "bounded_metrics": margins,
        },
        "holdout_pair_count": pair_count,
        "metrics": {
            "exact_outcome_pairs": exact_successes,
            "fully_passing_pairs": pair_successes,
            "exact_outcome_fraction": exact_successes / pair_count,
            "exact_outcome_wilson_lower_95_informational": wilson_lower_bound(
                exact_successes, pair_count
            ),
            "per_dataset": per_dataset,
            "maximum_unapproved_post_ingest_transfer_bytes": maximum_transfer,
            "minimum_vram_headroom": minimum_headroom,
        },
        "operational_thresholds": {
            "unapproved_post_ingest_transfer_bytes_max": operational_thresholds[
                "unapproved_post_ingest_transfer_bytes_max"
            ],
            "vram_headroom_min": operational_thresholds["vram_headroom_min"],
        },
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
            "Comparator component confirmed for the unchanged final candidate."
            if status == "PASS"
            else "Candidate stopped; no retuning or hierarchy work is permitted."
        ),
    }
    immutable_json(args.output, payload)


if __name__ == "__main__":
    main()
