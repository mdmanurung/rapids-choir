#!/usr/bin/env python3
"""Aggregate hierarchy evidence without pooling roles, datasets, or metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import (
    file_hash,
    immutable_json,
    validate_candidate_manifest,
)

AGGREGATE_SCHEMA = "rapids_choir.hierarchy-fidelity-aggregate/v1"
ENDPOINT_METRICS = {
    "one_minus_ari",
    "pairwise_false_split_rate",
    "pairwise_false_merge_rate",
    "record_discordance",
}
REPORT_CHECKS = {
    "candidate_context_exact",
    "schemas_and_topology_invariants",
    "provenance_contract_exact",
    "semantic_trace_exact_requirements",
    "endpoint_metrics_within_development_margins",
    "role_contract_bound",
    "role_requirements_satisfied",
    "confirmation_opening_contract",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"hierarchy evidence must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_artifact_reference(value: Any, *, role: str) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{role} hierarchy evidence reference is incomplete")
    path = Path(str(value["path"]))
    if not path.is_file() or _sha256(path) != value["sha256"]:
        raise ValueError(f"{role} hierarchy evidence reference is stale")


def _index_reports(
    paths: list[Path],
    expected: set[str],
    *,
    phase: str,
    candidate: dict[str, Any],
    margin_path: Path,
    expected_margins: dict[str, float],
    expected_roles: dict[str, str],
    role_contract_path: Path,
) -> dict[str, tuple[Path, dict[str, Any]]]:
    observed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        report = _read(path)
        if report.get("schema_version") != 1:
            raise ValueError(f"unsupported {phase} hierarchy report: {path}")
        dataset_id = str(report.get("dataset_id"))
        if dataset_id not in expected:
            raise ValueError(f"unexpected {phase} hierarchy dataset: {dataset_id}")
        if dataset_id in observed:
            raise ValueError(f"duplicate {phase} hierarchy dataset: {dataset_id}")
        if report.get("status") not in {"PASS", "FAIL"}:
            raise ValueError(f"invalid {phase} hierarchy status: {dataset_id}")
        if report.get("candidate") != candidate:
            raise ValueError(f"{phase} hierarchy candidate drift: {dataset_id}")
        if report.get("role") != expected_roles[dataset_id]:
            raise ValueError(f"{phase} hierarchy role drift: {dataset_id}")
        checks = report.get("checks")
        metric_checks = report.get("metric_checks")
        metrics = report.get("metrics")
        report_margins = report.get("margins")
        if (
            not isinstance(checks, dict)
            or set(checks) != REPORT_CHECKS
            or not all(isinstance(value, bool) for value in checks.values())
            or not isinstance(metric_checks, dict)
            or set(metric_checks) != ENDPOINT_METRICS
            or not all(isinstance(value, bool) for value in metric_checks.values())
            or not isinstance(metrics, dict)
            or set(metrics) != ENDPOINT_METRICS
            or not isinstance(report_margins, dict)
            or report_margins != expected_margins
        ):
            raise ValueError(f"malformed {phase} hierarchy report: {dataset_id}")
        passed = all(checks.values()) and all(metric_checks.values())
        if (report["status"] == "PASS") is not passed:
            raise ValueError(f"inconsistent {phase} hierarchy status: {dataset_id}")
        evidence = report.get("evidence")
        expected_evidence = {
            "reference_result",
            "candidate_result",
            "margin_contract",
            "role_contract",
        }
        if phase == "confirmation":
            expected_evidence.add("opening")
        if not isinstance(evidence, dict) or set(evidence) != expected_evidence:
            raise ValueError(f"incomplete {phase} hierarchy custody: {dataset_id}")
        for role in ("reference_result", "candidate_result", "margin_contract"):
            _validate_artifact_reference(evidence[role], role=role)
        _validate_artifact_reference(
            evidence["role_contract"], role="role_contract"
        )
        if phase == "confirmation":
            _validate_artifact_reference(evidence["opening"], role="opening")
            opening = report.get("opening")
            if (
                not isinstance(opening, dict)
                or opening.get("opening_count") != 1
                or not isinstance(opening.get("attempt_sha256"), str)
                or not isinstance(opening.get("authorization_sha256"), str)
            ):
                raise ValueError(
                    f"invalid confirmation opening custody: {dataset_id}"
                )
        elif report.get("opening") is not None:
            raise ValueError(
                f"development report has confirmation opening: {dataset_id}"
            )
        if (
            Path(str(evidence["margin_contract"]["path"])).resolve()
            != margin_path.resolve()
            or evidence["margin_contract"]["sha256"] != _sha256(margin_path)
        ):
            raise ValueError(f"{phase} hierarchy margin drift: {dataset_id}")
        if (
            Path(str(evidence["role_contract"]["path"])).resolve()
            != role_contract_path.resolve()
            or evidence["role_contract"]["sha256"]
            != _sha256(role_contract_path)
        ):
            raise ValueError(f"{phase} hierarchy role-contract drift: {dataset_id}")
        observed[dataset_id] = (path, report)
    if set(observed) != expected:
        raise ValueError(
            f"incomplete {phase} hierarchy evidence: "
            f"missing={sorted(expected - set(observed))}"
        )
    return observed


def _evidence_record(
    dataset_id: str,
    role: str,
    path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "role": role,
        "status": report["status"],
        "run_id": report.get("run_id"),
        "checks": report.get("checks"),
        "metrics": report.get("metrics"),
        "margins": report.get("margins"),
        "opening": report.get("opening"),
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def aggregate(
    design_path: Path,
    margin_path: Path,
    development_role_contract_path: Path,
    confirmation_role_contract_path: Path,
    development_paths: list[Path],
    confirmation_paths: list[Path],
) -> dict[str, Any]:
    design = _read(design_path)
    if design.get("schema_version") != 1 or design.get("status") != "DESIGN_FROZEN":
        raise ValueError("hierarchy validation design is not frozen")
    margins = _read(margin_path)
    if margins.get("status") != "FROZEN" or margins.get("eligible") is not True:
        raise ValueError("hierarchy fidelity margins are not frozen and eligible")
    if float(margins.get("maximum_margin", -1)) != 0.03:
        raise ValueError("hierarchy fidelity margin cap has drifted")
    candidate = margins.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("hierarchy fidelity margins lack candidate identity")
    raw_endpoint_margins = margins.get("hierarchy_endpoint_margins")
    if not isinstance(raw_endpoint_margins, dict) or set(
        raw_endpoint_margins
    ) != ENDPOINT_METRICS:
        raise ValueError("hierarchy fidelity endpoint margins are incomplete")
    expected_margins = {
        name: float(record["margin"])
        for name, record in raw_endpoint_margins.items()
        if isinstance(record, dict) and "margin" in record
    }
    if set(expected_margins) != ENDPOINT_METRICS:
        raise ValueError("hierarchy fidelity endpoint margins are malformed")

    development_records = design.get("development", {}).get("objects", [])
    development_roles = {
        str(record["dataset_id"]): str(record["role"])
        for record in development_records
    }
    expected_count = design.get("development", {}).get("expected_object_count")
    if (
        len(development_roles) != expected_count
        or len(set(development_roles.values())) != expected_count
        or design.get("development", {}).get("role_exclusive") is not True
    ):
        raise ValueError("hierarchy development role design has drifted")
    confirmation_records = design.get("confirmation", {}).get("datasets", [])
    confirmation_roles = {
        str(record["dataset_id"]): "hierarchy_parity_confirmation"
        for record in confirmation_records
    }
    if (
        set(confirmation_roles) != {"Arigoni", "Chen"}
        or any(record.get("open_once") is not True for record in confirmation_records)
        or any(record.get("opened") is not False for record in confirmation_records)
        or design.get("confirmation", {}).get("each_dataset_must_pass_independently")
        is not True
    ):
        raise ValueError("hierarchy confirmation design has drifted")

    development = _index_reports(
        development_paths,
        set(development_roles),
        phase="development",
        candidate=candidate,
        margin_path=margin_path,
        expected_margins=expected_margins,
        expected_roles=development_roles,
        role_contract_path=development_role_contract_path,
    )
    confirmation = _index_reports(
        confirmation_paths,
        set(confirmation_roles),
        phase="confirmation",
        candidate=candidate,
        margin_path=margin_path,
        expected_margins=expected_margins,
        expected_roles=confirmation_roles,
        role_contract_path=confirmation_role_contract_path,
    )
    development_evidence = [
        _evidence_record(
            dataset_id,
            development_roles[dataset_id],
            *development[dataset_id],
        )
        for dataset_id in sorted(development)
    ]
    confirmation_evidence = [
        _evidence_record(
            dataset_id,
            confirmation_roles[dataset_id],
            *confirmation[dataset_id],
        )
        for dataset_id in ("Arigoni", "Chen")
    ]
    checks = {
        "all_21_role_exclusive_development_objects_pass": all(
            record["status"] == "PASS" for record in development_evidence
        ),
        "arigoni_passes_independently": confirmation_evidence[0]["status"]
        == "PASS",
        "chen_passes_independently": confirmation_evidence[1]["status"]
        == "PASS",
        "confirmation_opened_once_each": all(
            record.get("opening", {}).get("opening_count") == 1
            for record in confirmation_evidence
        ),
        "no_development_confirmation_pooling": True,
    }
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "candidate": candidate,
        "checks": checks,
        "development": development_evidence,
        "confirmation": confirmation_evidence,
        "acceptance_policy": {
            "per_dataset": True,
            "per_repeat": True,
            "pooled_acceptance": False,
            "maximum_margin": 0.03,
        },
        "design": {
            "path": str(design_path.resolve()),
            "sha256": _sha256(design_path),
        },
        "margins": {
            "path": str(margin_path.resolve()),
            "sha256": _sha256(margin_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--design", required=True, type=Path)
    parser.add_argument("--margins", required=True, type=Path)
    parser.add_argument(
        "--development-role-contract", required=True, type=Path
    )
    parser.add_argument(
        "--confirmation-role-contract", required=True, type=Path
    )
    parser.add_argument("--development", action="append", required=True, type=Path)
    parser.add_argument("--confirmation", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = aggregate(
        arguments.design,
        arguments.margins,
        arguments.development_role_contract,
        arguments.confirmation_role_contract,
        arguments.development,
        arguments.confirmation,
    )
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
        raise RuntimeError("hierarchy aggregate candidate drift")
    immutable_json(arguments.output, report)
    if report["status"] != "PASS":
        immutable_json(
            arguments.output.with_name(f"{arguments.output.stem}.FAILED.json"),
            report,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
