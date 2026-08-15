#!/usr/bin/env python3
"""Require the final candidate to reproduce the promoted RF-profile winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import file_hash, immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"RF-profile evidence must be an object: {path}")
    return value


def _selection(value: dict[str, Any], *, role: str) -> dict[str, Any]:
    rankings = value.get("rankings")
    winner = value.get("winner")
    reference = value.get("reference_variability")
    if (
        value.get("schema_version") != 1
        or value.get("scope") != "development_rf_calibration"
        or value.get("status") != "FROZEN"
        or not isinstance(rankings, list)
        or len(rankings) != 12
        or not isinstance(winner, dict)
        or winner.get("profile_id") != rankings[0].get("profile_id")
        or winner.get("profile") != rankings[0].get("profile")
        or not isinstance(reference, dict)
        or reference.get("maximum_margin") != 0.03
    ):
        raise ValueError(f"{role} RF-profile selection is invalid")
    profile_ids = [record.get("profile_id") for record in rankings]
    if len(set(profile_ids)) != 12 or None in profile_ids:
        raise ValueError(f"{role} RF-profile ranking is incomplete")
    candidate = value.get("calibration_candidate")
    if not isinstance(candidate, dict):
        raise ValueError(f"{role} RF-profile lacks candidate identity")
    return {
        "winner_profile_id": winner["profile_id"],
        "winner_profile": winner["profile"],
        "ranking_order": value.get("ranking_order"),
        "comparator_contract_sha256": value.get("comparator_contract", {}).get(
            "sha256"
        ),
        "environment_locks": value.get("environment_locks"),
        "reference_variability": {
            "maximum_margin": reference.get("maximum_margin"),
            "bounded_metric_margins": reference.get("bounded_metric_margins"),
        },
    }


def verify(frozen_path: Path, reproduced_path: Path) -> dict[str, Any]:
    frozen = _read(frozen_path)
    reproduced = _read(reproduced_path)
    frozen_selection = _selection(frozen, role="frozen")
    reproduced_selection = _selection(reproduced, role="reproduced")
    winner_exact = (
        reproduced_selection["winner_profile_id"]
        == frozen_selection["winner_profile_id"]
        and reproduced_selection["winner_profile"]
        == frozen_selection["winner_profile"]
    )
    protocol_exact = {
        key: reproduced_selection[key] == frozen_selection[key]
        for key in (
            "ranking_order",
            "comparator_contract_sha256",
            "environment_locks",
        )
    }
    protocol_exact["environment_lock_sha256"] = protocol_exact.pop(
        "environment_locks"
    )
    margins_exact = (
        reproduced_selection["reference_variability"]
        == frozen_selection["reference_variability"]
    )
    reproduced_candidate = reproduced["calibration_candidate"]
    exact = winner_exact and margins_exact and all(protocol_exact.values())
    return {
        "schema_version": 1,
        "status": "REPRODUCED" if exact else "DRIFTED",
        "candidate": reproduced_candidate,
        "exact_winning_profile": winner_exact,
        "exact_reference_variability_margins": margins_exact,
        "protocol_checks": protocol_exact,
        "frozen": {
            "path": str(frozen_path.resolve()),
            "sha256": file_hash(frozen_path),
            "calibration_candidate": frozen["calibration_candidate"],
        },
        "reproduced": {
            "path": str(reproduced_path.resolve()),
            "sha256": file_hash(reproduced_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--reproduced", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = verify(arguments.frozen, arguments.reproduced)
    immutable_json(arguments.output, report)
    if report["status"] != "REPRODUCED":
        immutable_json(
            arguments.output.with_name(
                f"{arguments.output.stem}.DRIFTED.json"
            ),
            report,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
