"""Fail-closed cross-language semantic trace comparison."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any

from ._semantic_trace import SemanticEvent


@dataclass(frozen=True, slots=True)
class TraceDifference:
    path: str
    reference: Any
    candidate: Any
    rule: str


@dataclass(frozen=True, slots=True)
class TraceComparison:
    passed: bool
    event_count: int
    differences: tuple[TraceDifference, ...]
    maximum_bounded_differences: dict[str, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "event_count": self.event_count,
            "differences": [
                {
                    "path": difference.path,
                    "reference": difference.reference,
                    "candidate": difference.candidate,
                    "rule": difference.rule,
                }
                for difference in self.differences
            ],
            "maximum_bounded_differences": self.maximum_bounded_differences,
        }


def _validate_contract(
    contract: dict[str, Any],
) -> tuple[float, float, dict[str, float], set[str]]:
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported fidelity margin contract")
    if contract.get("status") != "FROZEN" or contract.get("eligible") is not True:
        raise ValueError("fidelity margins are not frozen and eligible")
    maximum = float(contract.get("maximum_margin", -1))
    if not 0 <= maximum <= 0.03:
        raise ValueError("maximum fidelity margin must lie in [0, 0.03]")
    deterministic = contract.get("deterministic_float", {})
    rtol = float(deterministic.get("rtol", -1))
    atol = float(deterministic.get("atol", -1))
    if rtol != 1e-7 or atol != 1e-10:
        raise ValueError("deterministic float tolerances are not locked")
    raw_margins = contract.get("stochastic_margins", {})
    if not isinstance(raw_margins, dict):
        raise ValueError("stochastic_margins must be an object")
    margins: dict[str, float] = {}
    for name, record in raw_margins.items():
        if not isinstance(record, dict) or record.get("eligible") is not True:
            raise ValueError(f"stochastic margin {name!r} is ineligible")
        margin = float(record.get("margin", -1))
        if not 0 <= margin <= maximum:
            raise ValueError(f"stochastic margin {name!r} exceeds the cap")
        margins[str(name)] = margin
    raw_deterministic = contract.get("deterministic_bounded_fields", [])
    if not isinstance(raw_deterministic, list) or not all(
        isinstance(name, str) for name in raw_deterministic
    ):
        raise ValueError("deterministic_bounded_fields must be a string list")
    deterministic_fields = set(raw_deterministic)
    if deterministic_fields & set(margins):
        raise ValueError("bounded fields cannot be both deterministic and stochastic")
    return rtol, atol, margins, deterministic_fields


def _compare_recursive(
    reference: Any,
    candidate: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
    differences: list[TraceDifference],
) -> None:
    if isinstance(reference, bool) or isinstance(candidate, bool):
        if type(reference) is not type(candidate) or reference != candidate:
            differences.append(TraceDifference(path, reference, candidate, "exact"))
        return
    numeric = (int, float)
    if isinstance(reference, numeric) and isinstance(candidate, numeric):
        if isinstance(reference, int) and isinstance(candidate, int):
            if reference != candidate:
                differences.append(
                    TraceDifference(path, reference, candidate, "exact")
                )
            return
        if not isclose(float(reference), float(candidate), rel_tol=rtol, abs_tol=atol):
            differences.append(
                TraceDifference(path, reference, candidate, "deterministic_float")
            )
        return
    if isinstance(reference, dict) and isinstance(candidate, dict):
        if set(reference) != set(candidate):
            differences.append(
                TraceDifference(
                    path,
                    sorted(reference),
                    sorted(candidate),
                    "exact_keys",
                )
            )
            return
        for key in sorted(reference):
            _compare_recursive(
                reference[key],
                candidate[key],
                path=f"{path}.{key}",
                rtol=rtol,
                atol=atol,
                differences=differences,
            )
        return
    if isinstance(reference, (list, tuple)) and isinstance(candidate, (list, tuple)):
        if len(reference) != len(candidate):
            differences.append(
                TraceDifference(path, len(reference), len(candidate), "exact_length")
            )
            return
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            _compare_recursive(
                left,
                right,
                path=f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                differences=differences,
            )
        return
    if type(reference) is not type(candidate) or reference != candidate:
        differences.append(TraceDifference(path, reference, candidate, "exact"))


def compare_semantic_traces(
    reference: tuple[SemanticEvent, ...],
    candidate: tuple[SemanticEvent, ...],
    contract: dict[str, Any],
) -> TraceComparison:
    """Compare one R run to one Python run without repeat or dataset pooling."""

    rtol, atol, margins, deterministic_fields = _validate_contract(contract)
    differences: list[TraceDifference] = []
    maxima: dict[str, float] = {}
    if len(reference) != len(candidate):
        differences.append(
            TraceDifference(
                "events", len(reference), len(candidate), "exact_event_count"
            )
        )
    for index, (left, right) in enumerate(
        zip(reference, candidate, strict=False)
    ):
        prefix = f"events[{index}]"
        for field in (
            "schema_version",
            "run_id",
            "operation_index",
            "event_type",
            "phase",
            "seed",
            "input_membership_hashes",
            "output_membership_hashes",
            "decision",
            "reason",
        ):
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if type(left_value) is not type(right_value) or left_value != right_value:
                differences.append(
                    TraceDifference(
                        f"{prefix}.{field}", left_value, right_value, "exact"
                    )
                )
        for field in ("parameters", "topology_context", "payload"):
            _compare_recursive(
                getattr(left, field),
                getattr(right, field),
                path=f"{prefix}.{field}",
                rtol=rtol,
                atol=atol,
                differences=differences,
            )
        if set(left.bounded_values) != set(right.bounded_values):
            differences.append(
                TraceDifference(
                    f"{prefix}.bounded_values",
                    sorted(left.bounded_values),
                    sorted(right.bounded_values),
                    "exact_keys",
                )
            )
            continue
        for name in sorted(left.bounded_values):
            left_value = left.bounded_values[name]
            right_value = right.bounded_values[name]
            delta = abs(left_value - right_value)
            maxima[name] = max(maxima.get(name, 0.0), delta)
            path = f"{prefix}.bounded_values.{name}"
            if name in margins:
                if delta > margins[name]:
                    differences.append(
                        TraceDifference(
                            path,
                            left_value,
                            right_value,
                            f"stochastic_margin<={margins[name]}",
                        )
                    )
            elif name in deterministic_fields:
                if not isclose(
                    left_value, right_value, rel_tol=rtol, abs_tol=atol
                ):
                    differences.append(
                        TraceDifference(
                            path,
                            left_value,
                            right_value,
                            "deterministic_float",
                        )
                    )
            else:
                differences.append(
                    TraceDifference(
                        path,
                        left_value,
                        right_value,
                        "unclassified_bounded_metric",
                    )
                )
    return TraceComparison(
        passed=not differences,
        event_count=min(len(reference), len(candidate)),
        differences=tuple(differences),
        maximum_bounded_differences=maxima,
    )
