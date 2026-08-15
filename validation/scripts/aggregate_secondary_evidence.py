#!/usr/bin/env python3
"""Aggregate construct-specific corrected/compatibility case-study reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import file_hash, immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"secondary evidence must be an object: {path}")
    return value


def aggregate(contract_path: Path, report_paths: list[Path]) -> dict[str, Any]:
    contract = _read(contract_path)
    expected = {
        record["dataset_id"]: set(record["constructs"])
        for record in contract.get("datasets", [])
    }
    if (
        contract.get("schema_version") != 1
        or set(expected) != {"Stoeckius", "Kang", "pancreas"}
        or contract.get("modes") != ["compatibility", "corrected"]
        or contract.get("reporting", {}).get(
            "unfavorable_valid_results_retained"
        )
        is not True
        or contract.get("reporting", {}).get(
            "pooled_biological_validity_claim"
        )
        is not False
    ):
        raise ValueError("secondary-evidence contract has drifted")
    reports: dict[str, dict[str, Any]] = {}
    candidate: dict[str, Any] | None = None
    for path in report_paths:
        report = _read(path)
        dataset_id = report.get("dataset_id")
        if dataset_id not in expected or dataset_id in reports:
            raise ValueError(f"unexpected or duplicate case study: {dataset_id}")
        if (
            report.get("schema_version") != 1
            or report.get("status") != "VALID"
            or report.get("runtime_schema_status") != "PASS"
            or report.get("modes") != ["compatibility", "corrected"]
            or set(report.get("constructs", {})) != expected[dataset_id]
        ):
            raise ValueError(f"invalid case-study report: {dataset_id}")
        for construct, by_mode in report["constructs"].items():
            if not isinstance(by_mode, dict) or set(by_mode) != {
                "compatibility",
                "corrected",
            }:
                raise ValueError(
                    f"case-study construct is not mode-separated: {construct}"
                )
            if any(value is None for value in by_mode.values()):
                raise ValueError(f"case-study construct is missing: {construct}")
        report_candidate = report.get("candidate")
        if not isinstance(report_candidate, dict):
            raise ValueError(f"case-study candidate is missing: {dataset_id}")
        if candidate is None:
            candidate = report_candidate
        elif report_candidate != candidate:
            raise ValueError("secondary-evidence candidate drift")
        reports[str(dataset_id)] = {
            "status": "VALID",
            "constructs": report["constructs"],
            "path": str(path.resolve()),
            "sha256": file_hash(path),
        }
    if set(reports) != set(expected):
        raise ValueError(
            "secondary evidence is incomplete: "
            f"missing={sorted(set(expected) - set(reports))}"
        )
    return {
        "schema_version": 1,
        "status": "VALID",
        "scope": "construct_specific_secondary_evidence",
        "candidate": candidate,
        "datasets": [reports[name] | {"dataset_id": name} for name in expected],
        "unfavorable_valid_results_retained": True,
        "pooled_biological_validity_claim": False,
        "may_determine_publication_eligibility": False,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": file_hash(contract_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--report", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    immutable_json(
        arguments.output,
        aggregate(arguments.contract, arguments.report),
    )


if __name__ == "__main__":
    main()
