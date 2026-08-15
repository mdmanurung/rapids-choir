#!/usr/bin/env python3
"""Require final-candidate hierarchy margins to reproduce the frozen contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import file_hash, immutable_json

MARGIN_PAYLOAD_FIELDS = {
    "maximum_margin",
    "quantile",
    "quantile_method",
    "deterministic_float",
    "deterministic_bounded_fields",
    "stochastic_margins",
    "hierarchy_endpoint_margins",
    "exact_fields_pass",
    "development_object_count",
    "calibration_record_count",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"hierarchy margin evidence must be an object: {path}")
    return value


def _projection(value: dict[str, Any]) -> dict[str, Any]:
    missing = sorted(MARGIN_PAYLOAD_FIELDS - value.keys())
    if missing:
        raise ValueError(f"hierarchy margin payload is incomplete: {missing}")
    return {name: value[name] for name in sorted(MARGIN_PAYLOAD_FIELDS)}


def verify(frozen_path: Path, reproduced_path: Path) -> dict[str, Any]:
    frozen = _read(frozen_path)
    reproduced = _read(reproduced_path)
    for name, value in (("frozen", frozen), ("reproduced", reproduced)):
        if (
            value.get("schema_version") != 1
            or value.get("contract_id") != "hierarchy-fidelity-margins-v1"
            or value.get("status") != "FROZEN"
            or value.get("eligible") is not True
            or float(value.get("maximum_margin", -1)) != 0.03
        ):
            raise ValueError(f"{name} hierarchy margins are not eligible and frozen")
    frozen_payload = _projection(frozen)
    reproduced_payload = _projection(reproduced)
    exact = reproduced_payload == frozen_payload
    candidate = reproduced.get("candidate")
    if not isinstance(candidate, dict):
        raise ValueError("reproduced hierarchy margins lack candidate identity")
    return {
        "schema_version": 1,
        "status": "REPRODUCED" if exact else "DRIFTED",
        "candidate": candidate,
        "exact_margin_payload": exact,
        "maximum_margin": 0.03,
        "frozen": {
            "path": str(frozen_path.resolve()),
            "sha256": file_hash(frozen_path),
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
