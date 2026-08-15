#!/usr/bin/env python3
"""Freeze hierarchy fidelity margins from role-exclusive R-vs-R development runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from validation.scripts.gate2_common import (
    file_hash,
    immutable_json,
    validate_candidate_manifest,
)

ENDPOINTS = {
    "one_minus_ari",
    "pairwise_false_split_rate",
    "pairwise_false_merge_rate",
    "record_discordance",
}
DETERMINISTIC_BOUNDED = {"adjusted_alpha", "previous_alpha"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive(
    contract_path: Path,
    calibration_paths: list[Path],
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("status") != "DESIGN_FROZEN":
        raise ValueError("hierarchy validation design is not frozen")
    objects = contract.get("development", {}).get("objects", [])
    expected = {record["dataset_id"]: record["role"] for record in objects}
    variability = contract.get("development", {}).get(
        "reference_variability", {}
    )
    expected_repeat_pair = variability.get("repeat_pair")
    expected_calibration_count = variability.get(
        "expected_calibration_record_count"
    )
    seeds = variability.get("seeds")
    if (
        expected_repeat_pair != "seed104729-vs-seed104759"
        or seeds != [104729, 104759]
        or expected_calibration_count != len(expected)
    ):
        raise ValueError("hierarchy R-vs-R calibration design has drifted")
    if len(expected) != contract["development"].get("expected_object_count"):
        raise ValueError("hierarchy development object count has drifted")
    observed: dict[str, set[str]] = {}
    endpoint_values: dict[str, list[float]] = {name: [] for name in ENDPOINTS}
    bounded_values: dict[str, list[float]] = {}
    exact_pass = True
    candidate_context: dict[str, Any] | None = None
    sources: list[dict[str, Any]] = []
    for path in calibration_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "rapids_choir.hierarchy-calibration/v1":
            raise ValueError(f"unsupported hierarchy calibration schema: {path}")
        dataset_id = str(payload.get("dataset_id"))
        role = str(payload.get("role"))
        repeat_pair = str(payload.get("repeat_pair"))
        if expected.get(dataset_id) != role:
            raise ValueError(f"hierarchy calibration role drift: {dataset_id}")
        if repeat_pair != expected_repeat_pair:
            raise ValueError(f"hierarchy calibration seed-pair drift: {dataset_id}")
        if repeat_pair in observed.setdefault(dataset_id, set()):
            raise ValueError(f"duplicate hierarchy calibration repeat: {dataset_id}")
        observed[dataset_id].add(repeat_pair)
        candidate = payload.get("candidate")
        if not isinstance(candidate, dict):
            raise ValueError(f"hierarchy calibration lacks candidate identity: {path}")
        if candidate_context is None:
            candidate_context = candidate
        elif candidate != candidate_context:
            raise ValueError("hierarchy calibration candidate drift")
        exact_pass = exact_pass and payload.get("exact_fields_pass") is True
        endpoints = payload.get("endpoint_differences")
        if not isinstance(endpoints, dict) or set(endpoints) != ENDPOINTS:
            raise ValueError(f"hierarchy endpoint calibration is incomplete: {path}")
        for name, raw_value in endpoints.items():
            value = float(raw_value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid hierarchy endpoint difference: {name}")
            endpoint_values[name].append(value)
        bounded = payload.get("bounded_differences")
        if not isinstance(bounded, dict):
            raise ValueError(f"bounded hierarchy calibration is invalid: {path}")
        for name, raw_value in bounded.items():
            if name in DETERMINISTIC_BOUNDED:
                raise ValueError(
                    "deterministic bounded field entered stochastic calibration: "
                    f"{name}"
                )
            value = float(raw_value)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid bounded hierarchy difference: {name}")
            bounded_values.setdefault(str(name), []).append(value)
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "dataset_id": dataset_id,
                "role": role,
                "repeat_pair": repeat_pair,
            }
        )
    if set(observed) != set(expected):
        raise ValueError(
            "hierarchy development role set is incomplete: "
            f"missing={sorted(set(expected) - set(observed))}"
        )
    if len(sources) != expected_calibration_count:
        raise ValueError("hierarchy calibration record count has drifted")
    if not bounded_values:
        raise ValueError("hierarchy calibration contains no bounded metrics")
    cap = float(contract["metric_policy"]["maximum_margin"])
    quantile = float(contract["metric_policy"]["quantile"])
    quantile_method = contract["metric_policy"].get("quantile_method")
    if cap != 0.03 or quantile != 0.975 or quantile_method != "linear":
        raise ValueError("hierarchy margin policy has drifted")

    def margin_record(values: list[float]) -> dict[str, Any]:
        observed_quantile = float(
            np.quantile(values, quantile, method=quantile_method)
        )
        maximum_observed = max(values)
        eligible = maximum_observed <= cap and observed_quantile <= cap
        return {
            "margin": min(observed_quantile, cap),
            "observed_quantile": observed_quantile,
            "maximum_observed": maximum_observed,
            "eligible": eligible,
            "observation_count": len(values),
        }

    endpoint_margins = {
        name: margin_record(values)
        for name, values in sorted(endpoint_values.items())
    }
    stochastic_margins = {
        name: margin_record(values)
        for name, values in sorted(bounded_values.items())
    }
    eligible = (
        exact_pass
        and all(record["eligible"] for record in endpoint_margins.values())
        and all(record["eligible"] for record in stochastic_margins.values())
    )
    return {
        "schema_version": 1,
        "contract_id": "hierarchy-fidelity-margins-v1",
        "status": "FROZEN" if eligible else "INELIGIBLE",
        "eligible": eligible,
        "maximum_margin": cap,
        "quantile": quantile,
        "quantile_method": quantile_method,
        "deterministic_float": {"rtol": 1e-7, "atol": 1e-10},
        "deterministic_bounded_fields": sorted(DETERMINISTIC_BOUNDED),
        "stochastic_margins": stochastic_margins,
        "hierarchy_endpoint_margins": endpoint_margins,
        "exact_fields_pass": exact_pass,
        "candidate": candidate_context,
        "development_object_count": len(observed),
        "calibration_record_count": len(sources),
        "validation_contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument(
        "--calibration", required=True, action="append", type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = derive(arguments.contract, arguments.calibration)
    candidate = validate_candidate_manifest(
        arguments.root.resolve(), arguments.candidate_manifest
    )
    expected_candidate = {
        "commit": candidate["candidate_commit"],
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": file_hash(arguments.candidate_manifest),
    }
    if report["candidate"] != expected_candidate:
        raise RuntimeError("hierarchy calibration candidate drift")
    immutable_json(arguments.output, report)
    if not report["eligible"]:
        immutable_json(
            arguments.output.with_name(
                f"{arguments.output.stem}.INELIGIBLE.json"
            ),
            report,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
