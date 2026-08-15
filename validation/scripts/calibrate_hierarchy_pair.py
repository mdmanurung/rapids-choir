#!/usr/bin/env python3
"""Summarize one R-vs-R hierarchy repeat pair before margins are frozen."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from rapids_choir._semantic_trace import SemanticEvent
from rapids_choir._trace_compare import compare_semantic_traces
from validation.scripts.evaluate_hierarchy_fidelity import (
    EVENT_INDEX,
    _load_json,
    _mandatory_trace_difference,
    _pair_metrics,
    _trace,
    _validate_result,
)
from validation.scripts.gate2_common import immutable_json

DETERMINISTIC_BOUNDED = {"adjusted_alpha", "previous_alpha"}
REPEAT_PAIR = re.compile(r"^seed(-?\d+)-vs-seed(-?\d+)$")


def _normalize_seed_payload(value: Any, *, field: str | None = None) -> Any:
    """Erase seed values, but retain their type, cardinality, and ordering slots."""

    seed_field = field is not None and (
        field == "random_state" or field.endswith("seed") or field.endswith("seeds")
    )
    if seed_field:
        if value is None:
            return None
        if isinstance(value, list):
            return [0 for _ in value]
        if isinstance(value, tuple):
            return tuple(0 for _ in value)
        if isinstance(value, int) and not isinstance(value, bool):
            return 0
    if isinstance(value, dict):
        return {
            key: _normalize_seed_payload(item, field=str(key))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_seed_payload(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_seed_payload(item) for item in value)
    return value


def _calibration_trace(
    events: tuple[SemanticEvent, ...],
) -> tuple[SemanticEvent, ...]:
    """Make different planned root seeds comparable without hiding structure."""

    return tuple(
        replace(
            event,
            run_id="hierarchy-r-vs-r-calibration",
            seed=None if event.seed is None else 0,
            parameters=_normalize_seed_payload(event.parameters),
            topology_context=_normalize_seed_payload(event.topology_context),
            payload=_normalize_seed_payload(event.payload),
        )
        for event in events
    )


def calibrate(
    first_path: Path,
    second_path: Path,
    *,
    role: str,
    repeat_pair: str,
) -> dict[str, object]:
    seed_match = REPEAT_PAIR.fullmatch(repeat_pair)
    if seed_match is None:
        raise ValueError("R-vs-R calibration repeat pair is not canonical")
    first_seed, second_seed = (int(value) for value in seed_match.groups())
    first = _load_json(first_path)
    second = _load_json(second_path)
    _validate_result(first_path, first)
    _validate_result(second_path, second)
    if first["dataset_id"] != second["dataset_id"]:
        raise ValueError("R-vs-R calibration dataset identities differ")
    if first["cell_ids"] != second["cell_ids"]:
        raise ValueError("R-vs-R calibration cell identities or order differ")
    if first["candidate"] != second["candidate"]:
        raise ValueError("R-vs-R calibration candidate identities differ")
    if (
        first["provenance"].get("random_state") != first_seed
        or second["provenance"].get("random_state") != second_seed
    ):
        raise ValueError("R-vs-R calibration seeds disagree with the repeat pair")
    first_trace = _trace(first_path, first)
    second_trace = _trace(second_path, second)
    bounded_fields = {
        name for event in first_trace for name in event.bounded_values
    } | {name for event in second_trace for name in event.bounded_values}
    stochastic = bounded_fields - DETERMINISTIC_BOUNDED
    temporary_contract = {
        "schema_version": 1,
        "status": "FROZEN",
        "eligible": True,
        "maximum_margin": 0.03,
        "deterministic_float": {"rtol": 1e-7, "atol": 1e-10},
        "deterministic_bounded_fields": sorted(
            bounded_fields & DETERMINISTIC_BOUNDED
        ),
        "stochastic_margins": {
            name: {"margin": 0.03, "eligible": True}
            for name in sorted(stochastic)
        },
    }
    comparison = compare_semantic_traces(
        _calibration_trace(first_trace),
        _calibration_trace(second_trace),
        temporary_contract,
    )
    exact_failures = [
        difference
        for difference in comparison.differences
        if _mandatory_trace_difference(
            difference.path, difference.rule, first_trace
        )
        and not difference.rule.startswith("stochastic_margin<=")
    ]
    discordant_events = {
        int(match.group(1))
        for difference in comparison.differences
        if difference not in exact_failures
        and ".bounded_values." not in difference.path
        and (match := EVENT_INDEX.match(difference.path)) is not None
    }
    endpoint_differences = _pair_metrics(first["labels"], second["labels"])
    endpoint_differences["record_discordance"] = len(discordant_events) / max(
        1, comparison.event_count
    )
    return {
        "schema_version": "rapids_choir.hierarchy-calibration/v1",
        "dataset_id": first["dataset_id"],
        "role": role,
        "repeat_pair": repeat_pair,
        "candidate": first["candidate"],
        "exact_fields_pass": not exact_failures,
        "exact_failures": [
            {"path": difference.path, "rule": difference.rule}
            for difference in exact_failures
        ],
        "endpoint_differences": endpoint_differences,
        "bounded_differences": {
            name: value
            for name, value in comparison.maximum_bounded_differences.items()
            if name in stochastic
        },
        "first": str(first_path.resolve()),
        "second": str(second_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True, type=Path)
    parser.add_argument("--second", required=True, type=Path)
    parser.add_argument("--role", required=True)
    parser.add_argument("--repeat-pair", required=True)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = calibrate(
        arguments.first,
        arguments.second,
        role=arguments.role,
        repeat_pair=arguments.repeat_pair,
    )
    immutable_json(arguments.output, report)
    if not report["exact_fields_pass"]:
        immutable_json(
            arguments.output.with_name(f"{arguments.output.stem}.FAILED.json"),
            report,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
