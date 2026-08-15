#!/usr/bin/env python3
"""Evaluate one R/Python hierarchy pair without pooling datasets or repeats."""

from __future__ import annotations

import argparse
import json
import re
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from rapids_choir._semantic_trace import read_trace_reference
from rapids_choir._trace_compare import compare_semantic_traces
from rapids_choir.tree import HierarchyTree
from validation.scripts.gate2_common import file_hash, immutable_json

RESULT_SCHEMA = "rapids_choir.hierarchy-result/v1"
RESULT_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts/hierarchy_result_v1.schema.json"
)
ENDPOINT_METRICS = {
    "one_minus_ari",
    "pairwise_false_split_rate",
    "pairwise_false_merge_rate",
    "record_discordance",
}
EVENT_INDEX = re.compile(r"^events\[(\d+)\]")
CONFIRMATION_ARTIFACTS = {
    "h5ad",
    "recipe",
    "python_config",
    "custody_record",
}


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"hierarchy result must be an object: {path}")
    return value


def _trace(result_path: Path, value: dict[str, Any]) -> tuple[Any, ...]:
    raw = value.get("semantic_trace")
    if not isinstance(raw, dict):
        raise ValueError("semantic_trace must be a content-bound reference")
    return read_trace_reference(raw, relative_to=result_path.parent)


def _validate_result(path: Path, value: dict[str, Any]) -> HierarchyTree:
    schema = _load_json(RESULT_SCHEMA_PATH)
    validation_errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if validation_errors:
        first = validation_errors[0]
        location = ".".join(str(item) for item in first.absolute_path) or "<root>"
        raise ValueError(
            f"hierarchy result schema violation at {location}: {first.message}"
        )
    required = {
        "schema_version",
        "candidate",
        "dataset_id",
        "run_id",
        "cell_ids",
        "labels",
        "tree",
        "provenance",
        "semantic_trace",
    }
    if set(value) != required or value.get("schema_version") != RESULT_SCHEMA:
        raise ValueError(f"hierarchy result schema is not canonical: {path}")
    cell_ids = value["cell_ids"]
    labels = value["labels"]
    if not isinstance(cell_ids, list) or not isinstance(labels, list):
        raise ValueError("hierarchy result cell_ids and labels must be arrays")
    if (
        not cell_ids
        or len(cell_ids) != len(labels)
        or len(set(cell_ids)) != len(cell_ids)
    ):
        raise ValueError("hierarchy result cell identity is empty or misaligned")
    tree = HierarchyTree.from_record(value["tree"])
    if tree.cell_ids != tuple(str(cell) for cell in cell_ids):
        raise ValueError("hierarchy result tree is not aligned to cell_ids")
    provenance = value["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("hierarchy result provenance must be an object")
    for field in (
        "schema",
        "run_id",
        "input_sha256",
        "input_contract",
        "configuration",
        "rules",
        "multiplicity",
        "random_state",
    ):
        if field not in provenance:
            raise ValueError(f"hierarchy result provenance lacks {field}")
    if value["run_id"] != provenance["run_id"]:
        raise ValueError("hierarchy result run ID and provenance disagree")
    trace_reference = value["semantic_trace"]
    if (
        not isinstance(trace_reference, dict)
        or trace_reference.get("run_id") != value["run_id"]
    ):
        raise ValueError("hierarchy result trace and run IDs disagree")
    return tree


def _factorize(values: list[Any]) -> np.ndarray[Any, np.dtype[np.int64]]:
    mapping: dict[Any, int] = {}
    labels = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        mapping.setdefault(key, len(mapping))
        labels[index] = mapping[key]
    return labels


def _pair_metrics(reference: list[Any], candidate: list[Any]) -> dict[str, float]:
    left = _factorize(reference)
    right = _factorize(candidate)
    n_cells = left.size
    if n_cells < 2:
        return {
            "one_minus_ari": 0.0,
            "pairwise_false_split_rate": 0.0,
            "pairwise_false_merge_rate": 0.0,
        }
    contingency: dict[tuple[int, int], int] = {}
    left_counts: dict[int, int] = {}
    right_counts: dict[int, int] = {}
    for left_value, right_value in zip(left.tolist(), right.tolist(), strict=True):
        contingency[(left_value, right_value)] = (
            contingency.get((left_value, right_value), 0) + 1
        )
        left_counts[left_value] = left_counts.get(left_value, 0) + 1
        right_counts[right_value] = right_counts.get(right_value, 0) + 1
    same_both = sum(comb(count, 2) for count in contingency.values())
    same_reference = sum(comb(count, 2) for count in left_counts.values())
    same_candidate = sum(comb(count, 2) for count in right_counts.values())
    total_pairs = comb(n_cells, 2)
    expected = same_reference * same_candidate / total_pairs
    maximum = 0.5 * (same_reference + same_candidate)
    ari = 1.0 if maximum == expected else (same_both - expected) / (maximum - expected)
    false_splits = same_reference - same_both
    false_merges = same_candidate - same_both
    reference_different = total_pairs - same_reference
    return {
        "one_minus_ari": float(1.0 - ari),
        "pairwise_false_split_rate": (
            0.0 if same_reference == 0 else false_splits / same_reference
        ),
        "pairwise_false_merge_rate": (
            0.0 if reference_different == 0 else false_merges / reference_different
        ),
    }


def _endpoint_margins(contract: dict[str, Any]) -> dict[str, float]:
    maximum = float(contract.get("maximum_margin", -1))
    if contract.get("status") != "FROZEN" or contract.get("eligible") is not True:
        raise ValueError("hierarchy fidelity margins are not frozen and eligible")
    if not 0 <= maximum <= 0.03:
        raise ValueError("hierarchy fidelity margin cap must lie in [0, 0.03]")
    records = contract.get("hierarchy_endpoint_margins")
    if not isinstance(records, dict) or set(records) != ENDPOINT_METRICS:
        raise ValueError("hierarchy endpoint margin set is incomplete")
    margins: dict[str, float] = {}
    for name, record in records.items():
        if not isinstance(record, dict) or record.get("eligible") is not True:
            raise ValueError(f"hierarchy endpoint margin {name!r} is ineligible")
        margin = float(record.get("margin", -1))
        if not 0 <= margin <= maximum:
            raise ValueError(f"hierarchy endpoint margin {name!r} exceeds cap")
        margins[name] = margin
    return margins


def _role_requirement(
    contract: dict[str, Any], dataset_id: str
) -> tuple[str, dict[str, Any] | None]:
    object_matches = [
        record
        for record in contract.get("objects", [])
        if record.get("dataset_id") == dataset_id
    ]
    role_matches = [
        record
        for record in contract.get("roles", [])
        if dataset_id in record.get("datasets", [])
    ]
    matches = [*object_matches, *role_matches]
    if len(matches) != 1:
        raise ValueError(
            "dataset must have exactly one role in its role contract: "
            f"{dataset_id}"
        )
    record = matches[0]
    role = record.get("role", record.get("role_id"))
    if not isinstance(role, str) or not role:
        raise ValueError(f"dataset role is invalid: {dataset_id}")
    required_event = record.get("required_event")
    if required_event is not None and not isinstance(required_event, dict):
        raise ValueError(f"required role event is invalid: {dataset_id}")
    return role, required_event


def _subset_matches(observed: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(
            key in observed and _subset_matches(observed[key], value)
            for key, value in expected.items()
        )
    return observed == expected


def _event_matches(event: Any, requirement: dict[str, Any]) -> bool:
    record = event.to_record()
    return _subset_matches(record, requirement)


def _validate_confirmation_opening(
    path: Path,
    *,
    dataset_id: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    opening = _load_json(path)
    if (
        opening.get("schema_version") != 1
        or opening.get("status") != "SEALED"
        or opening.get("scope")
        != "one_time_hierarchy_confirmation_opening"
        or opening.get("dataset_id") != dataset_id
        or opening.get("role")
        != "computational_parity_confirmation_only"
        or opening.get("opening_count") != 1
        or opening.get("independent_hierarchy_validation") is not False
        or opening.get("candidate") != candidate
    ):
        raise ValueError("hierarchy confirmation opening is invalid or drifted")
    attempt = opening.get("opening_attempt")
    if not isinstance(attempt, dict) or set(attempt) != {"path", "sha256"}:
        raise ValueError("hierarchy confirmation attempt marker is absent")
    attempt_path = Path(str(attempt["path"]))
    if (
        not attempt_path.is_file()
        or file_hash(attempt_path) != attempt["sha256"]
    ):
        raise ValueError("hierarchy confirmation attempt marker is stale")
    attempt_record = _load_json(attempt_path)
    if (
        attempt_record.get("status") != "OPENING_STARTED"
        or attempt_record.get("scope")
        != "one_time_hierarchy_confirmation_opening"
        or attempt_record.get("dataset_id") != dataset_id
        or attempt_record.get("candidate") != candidate
    ):
        raise ValueError("hierarchy confirmation attempt marker has drifted")
    authorization = opening.get("authorization")
    if not isinstance(authorization, dict) or set(authorization) != {
        "path",
        "sha256",
        "authorization_id",
    }:
        raise ValueError("hierarchy confirmation authorization is absent")
    authorization_path = Path(str(authorization["path"]))
    if (
        not authorization_path.is_file()
        or file_hash(authorization_path) != authorization["sha256"]
        or attempt_record.get("authorization_sha256")
        != authorization["sha256"]
    ):
        raise ValueError("hierarchy confirmation authorization is stale")
    artifacts = opening.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != CONFIRMATION_ARTIFACTS:
        raise ValueError("hierarchy confirmation sealed artifacts are incomplete")
    for name, artifact in artifacts.items():
        if not isinstance(artifact, dict) or set(artifact) != {
            "source_path",
            "sealed_path",
            "bytes",
            "sha256",
        }:
            raise ValueError(
                f"hierarchy confirmation artifact is malformed: {name}"
            )
        sealed = Path(str(artifact["sealed_path"]))
        if (
            not sealed.is_file()
            or sealed.stat().st_size != artifact["bytes"]
            or file_hash(sealed) != artifact["sha256"]
        ):
            raise ValueError(f"hierarchy confirmation artifact is stale: {name}")
    return opening


def _mandatory_trace_difference(
    path: str,
    rule: str,
    reference_events: tuple[Any, ...],
) -> bool:
    if path == "events" or rule in {
        "deterministic_float",
        "unclassified_bounded_metric",
    } or rule.startswith("stochastic_margin<="):
        return True
    if path.endswith(".bounded_values") and rule == "exact_keys":
        return True
    match = EVENT_INDEX.match(path)
    if match is None:
        return True
    event = reference_events[int(match.group(1))]
    if any(
        path.endswith(f".{field}")
        for field in (
            "schema_version",
            "operation_index",
            "event_type",
            "phase",
            "seed",
            "decision",
            "reason",
        )
    ):
        return True
    if event.event_type in {
        "provenance",
        "preprocessing",
        "prepared_input_validation",
    }:
        return True
    return event.event_type == "comparator_call" and (
        ".parameters.pair_kind" in path
    )


def evaluate(
    reference_path: Path,
    candidate_path: Path,
    margin_path: Path,
    role_contract_path: Path,
    expected_role: str,
    opening_path: Path | None = None,
) -> dict[str, Any]:
    reference = _load_json(reference_path)
    candidate = _load_json(candidate_path)
    reference_tree = _validate_result(reference_path, reference)
    candidate_tree = _validate_result(candidate_path, candidate)
    if reference["dataset_id"] != candidate["dataset_id"]:
        raise ValueError("hierarchy result dataset identities differ")
    if reference["run_id"] != candidate["run_id"]:
        raise ValueError("hierarchy result run identities differ")
    if reference["candidate"] != candidate["candidate"]:
        raise ValueError("hierarchy result candidate identities differ")
    if reference["cell_ids"] != candidate["cell_ids"]:
        raise ValueError("hierarchy result cell identities or order differ")
    reference_provenance = reference["provenance"]
    candidate_provenance = candidate["provenance"]
    provenance_fields = {
        "schema",
        "run_id",
        "input_sha256",
        "input_contract",
        "configuration",
        "rules",
        "multiplicity",
        "random_state",
    }
    provenance_exact = all(
        reference_provenance.get(field) == candidate_provenance.get(field)
        for field in provenance_fields
    )
    margin_contract = _load_json(margin_path)
    margins = _endpoint_margins(margin_contract)
    reference_trace = _trace(reference_path, reference)
    candidate_trace = _trace(candidate_path, candidate)
    role_contract = _load_json(role_contract_path)
    role, required_event = _role_requirement(
        role_contract, str(reference["dataset_id"])
    )
    if role != expected_role:
        raise ValueError(
            f"hierarchy role drift: expected={expected_role}, observed={role}"
        )
    confirmation_role = role == "hierarchy_parity_confirmation"
    if confirmation_role != (opening_path is not None):
        raise ValueError(
            "hierarchy confirmation role requires exactly one opening seal"
        )
    opening = (
        _validate_confirmation_opening(
            opening_path,
            dataset_id=str(reference["dataset_id"]),
            candidate=reference["candidate"],
        )
        if opening_path is not None
        else None
    )
    reference_role_event = required_event is None or any(
        _event_matches(event, required_event) for event in reference_trace
    )
    candidate_role_event = required_event is None or any(
        _event_matches(event, required_event) for event in candidate_trace
    )
    trace_comparison = compare_semantic_traces(
        reference_trace,
        candidate_trace,
        margin_contract,
    )
    metrics = _pair_metrics(reference["labels"], candidate["labels"])
    mandatory_trace_failures = [
        difference
        for difference in trace_comparison.differences
        if _mandatory_trace_difference(
            difference.path,
            difference.rule,
            reference_trace,
        )
    ]
    discordant_events = {
        int(match.group(1))
        for difference in trace_comparison.differences
        if difference not in mandatory_trace_failures
        and (match := EVENT_INDEX.match(difference.path)) is not None
    }
    event_denominator = max(1, trace_comparison.event_count)
    metrics["record_discordance"] = len(discordant_events) / event_denominator
    metric_checks = {
        name: value <= margins[name] for name, value in metrics.items()
    }
    checks = {
        "candidate_context_exact": reference["candidate"] == candidate["candidate"],
        "schemas_and_topology_invariants": (
            reference_tree.n_cells == candidate_tree.n_cells
        ),
        "provenance_contract_exact": provenance_exact,
        "semantic_trace_exact_requirements": not mandatory_trace_failures,
        "endpoint_metrics_within_development_margins": all(
            metric_checks.values()
        ),
        "role_contract_bound": True,
        "role_requirements_satisfied": (
            reference_role_event and candidate_role_event
        ),
        "confirmation_opening_contract": (
            not confirmation_role or opening is not None
        ),
    }
    evidence = {
        "reference_result": {
            "path": str(reference_path.resolve()),
            "sha256": file_hash(reference_path),
        },
        "candidate_result": {
            "path": str(candidate_path.resolve()),
            "sha256": file_hash(candidate_path),
        },
        "margin_contract": {
            "path": str(margin_path.resolve()),
            "sha256": file_hash(margin_path),
        },
        "role_contract": {
            "path": str(role_contract_path.resolve()),
            "sha256": file_hash(role_contract_path),
        },
    }
    opening_summary = None
    if opening_path is not None and opening is not None:
        evidence["opening"] = {
            "path": str(opening_path.resolve()),
            "sha256": file_hash(opening_path),
        }
        opening_summary = {
            "opening_count": opening["opening_count"],
            "attempt_sha256": opening["opening_attempt"]["sha256"],
            "authorization_sha256": opening["authorization"]["sha256"],
        }
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "dataset_id": reference["dataset_id"],
        "role": role,
        "opening": opening_summary,
        "required_event": required_event,
        "run_id": reference["run_id"],
        "candidate": reference["candidate"],
        "checks": checks,
        "metrics": metrics,
        "margins": margins,
        "metric_checks": metric_checks,
        "trace_comparison": trace_comparison.to_record(),
        "mandatory_trace_failures": [
            {
                "path": difference.path,
                "rule": difference.rule,
            }
            for difference in mandatory_trace_failures
        ],
        "evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--margins", required=True, type=Path)
    parser.add_argument("--role-contract", required=True, type=Path)
    parser.add_argument("--expected-role", required=True)
    parser.add_argument("--opening", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = evaluate(
        arguments.reference,
        arguments.candidate,
        arguments.margins,
        arguments.role_contract,
        arguments.expected_role,
        arguments.opening,
    )
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        immutable_json(arguments.output, report)
        if report["status"] != "PASS":
            immutable_json(
                arguments.output.with_name(
                    f"{arguments.output.stem}.FAILED.json"
                ),
                report,
            )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
