#!/usr/bin/env python3
"""Account for every in-scope Sant reproduction analysis without hiding failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import file_hash, immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Sant reproduction record must be an object: {path}")
    return value


def aggregate(contract_path: Path, record_paths: list[Path]) -> dict[str, Any]:
    contract = _read(contract_path)
    expected = contract.get("analyses")
    allowed = set(contract.get("allowed_classifications", []))
    if (
        contract.get("schema_version") != 1
        or not isinstance(expected, list)
        or len(expected) != 4
        or allowed != {"reproduced", "partial", "unverifiable", "failed"}
        or contract.get("failure_blocks_primary_claim") is not False
    ):
        raise ValueError("Sant reproduction contract has drifted")
    observed: dict[str, dict[str, Any]] = {}
    candidate: dict[str, Any] | None = None
    for path in record_paths:
        record = _read(path)
        analysis_id = record.get("analysis_id")
        classification = record.get("classification")
        if analysis_id not in expected or analysis_id in observed:
            raise ValueError(f"unexpected or duplicate Sant analysis: {analysis_id}")
        if (
            record.get("schema_version") != 1
            or classification not in allowed
            or not isinstance(record.get("attempts"), list)
            or not record["attempts"]
        ):
            raise ValueError(f"invalid Sant reproduction record: {analysis_id}")
        record_candidate = record.get("candidate")
        if not isinstance(record_candidate, dict):
            raise ValueError(f"Sant candidate is missing: {analysis_id}")
        if candidate is None:
            candidate = record_candidate
        elif record_candidate != candidate:
            raise ValueError("Sant reproduction candidate drift")
        observed[str(analysis_id)] = {
            "analysis_id": analysis_id,
            "classification": classification,
            "attempt_count": len(record["attempts"]),
            "path": str(path.resolve()),
            "sha256": file_hash(path),
        }
    if set(observed) != set(expected):
        raise ValueError(
            "Sant reproduction accounting is incomplete: "
            f"missing={sorted(set(expected) - set(observed))}"
        )
    complete = all(
        record["classification"] == "reproduced" for record in observed.values()
    )
    return {
        "schema_version": 1,
        "status": "ACCOUNTED",
        "candidate": candidate,
        "analyses": [observed[name] for name in expected],
        "completeness_claim": "ELIGIBLE" if complete else "WITHHELD",
        "primary_claim_blocked": False,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": file_hash(contract_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--record", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    immutable_json(
        arguments.output,
        aggregate(arguments.contract, arguments.record),
    )


if __name__ == "__main__":
    main()
