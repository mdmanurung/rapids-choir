from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from rapids_choir._semantic_trace import (
    SemanticEvent,
    TraceRecorder,
    read_trace,
    read_trace_parquet,
    read_trace_reference,
    semantic_trace_reference,
)
from rapids_choir._trace_compare import compare_semantic_traces
from rapids_choir.exceptions import ArtifactCollisionError

LEFT = "a" * 64
RIGHT = "b" * 64


def _event(**values: object) -> SemanticEvent:
    defaults: dict[str, object] = {
        "run_id": "run-fixture",
        "operation_index": 0,
        "event_type": "comparator_call",
        "phase": "pruning",
        "parameters": {"alpha": 0.05, "pair_kind": "sibling"},
        "seed": 7,
        "input_membership_hashes": (LEFT, RIGHT),
        "output_membership_hashes": (),
        "topology_context": {"level": 2, "parent": "P0"},
        "decision": "split",
        "reason": "split",
        "bounded_values": {"mean_observed_accuracy": 0.8},
        "payload": {"round_seeds": [8, 9]},
    }
    defaults.update(values)
    return SemanticEvent(**defaults)  # type: ignore[arg-type]


def _contract(*, margin: float = 0.01) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "FROZEN",
        "eligible": True,
        "maximum_margin": 0.03,
        "deterministic_float": {"rtol": 1e-7, "atol": 1e-10},
        "stochastic_margins": {
            "mean_observed_accuracy": {
                "margin": margin,
                "eligible": True,
            }
        },
        "deterministic_bounded_fields": ["adjusted_alpha"],
    }


def test_jsonl_and_parquet_traces_round_trip_immutably(tmp_path: Path) -> None:
    recorder = TraceRecorder("run-fixture")
    recorder.append(
        "comparator_call",
        "pruning",
        parameters={"alpha": 0.05, "pair_kind": "sibling"},
        seed=7,
        input_membership_hashes=(LEFT, RIGHT),
        topology_context={"level": 2, "parent": "P0"},
        decision="split",
        reason="split",
        bounded_values={"mean_observed_accuracy": 0.8},
        payload={"round_seeds": [8, 9]},
    )
    jsonl = tmp_path / "trace.jsonl"
    recorder.write_jsonl(jsonl)
    assert read_trace(jsonl) == recorder.events
    with pytest.raises(ArtifactCollisionError):
        recorder.write_jsonl(jsonl)

    parquet = tmp_path / "trace.parquet"
    manifest = recorder.write_parquet(parquet)
    assert read_trace_parquet(manifest) == recorder.events
    changed = asdict(manifest)
    changed["run_id"] = "run-changed"
    with pytest.raises(ValueError, match="metadata mismatch"):
        read_trace_parquet(changed)


def test_trace_reference_binds_checksum_run_id_and_row_count(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder("run-fixture")
    recorder.append("provenance", "initialization", decision="validated")
    path = tmp_path / "trace.jsonl"
    recorder.write_jsonl(path)
    reference = semantic_trace_reference(
        path,
        format="jsonl",
        relative_to=tmp_path,
    )
    assert read_trace_reference(reference, relative_to=tmp_path) == recorder.events
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        read_trace_reference(reference, relative_to=tmp_path)


def test_trace_reader_rejects_unknown_fields_and_wrong_scalar_types(
    tmp_path: Path,
) -> None:
    recorder = TraceRecorder("run-fixture")
    recorder.append("provenance", "initialization")
    record = recorder.events[0].to_record()

    unknown_path = tmp_path / "unknown.jsonl"
    unknown_path.write_text(
        json.dumps(dict(record, unexpected=True)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="incomplete or unknown"):
        read_trace(unknown_path)

    wrong_type_path = tmp_path / "wrong-type.jsonl"
    wrong_type_path.write_text(
        json.dumps(dict(record, operation_index=True)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="operation_index"):
        read_trace(wrong_type_path)


def test_trace_comparison_accepts_only_prespecified_numeric_differences() -> None:
    reference = (_event(),)
    within = (
        replace(
            reference[0],
            parameters={
                "alpha": 0.050000000001,
                "pair_kind": "sibling",
                "whole_float": 1.0,
            },
            bounded_values={"mean_observed_accuracy": 0.809},
        ),
    )
    reference = (
        replace(
            reference[0],
            parameters={
                "alpha": 0.05,
                "pair_kind": "sibling",
                "whole_float": 1,
            },
        ),
    )
    outside = (
        replace(
            reference[0],
            bounded_values={"mean_observed_accuracy": 0.811},
        ),
    )
    assert compare_semantic_traces(reference, within, _contract()).passed
    failed = compare_semantic_traces(reference, outside, _contract())
    assert not failed.passed
    assert failed.differences[0].rule == "stochastic_margin<=0.01"


@pytest.mark.parametrize(
    "mutation",
    [
        {"decision": "merge"},
        {"reason": "merge"},
        {"seed": 8},
        {"input_membership_hashes": ("c" * 64, RIGHT)},
        {"topology_context": {"level": 3, "parent": "P0"}},
        {"parameters": {"alpha": 0.05, "pair_kind": "bridge"}},
    ],
)
def test_trace_comparison_rejects_discrete_mutations(
    mutation: dict[str, object],
) -> None:
    reference = (_event(),)
    candidate = (replace(reference[0], **mutation),)
    assert not compare_semantic_traces(reference, candidate, _contract()).passed


def test_trace_comparison_rejects_unclassified_metrics() -> None:
    event = _event(bounded_values={"new_metric": 0.2})
    comparison = compare_semantic_traces((event,), (event,), _contract())
    assert not comparison.passed
    assert comparison.differences[0].rule == "unclassified_bounded_metric"


def test_trace_comparison_rejects_unfrozen_or_over_cap_contracts() -> None:
    unfrozen = _contract()
    unfrozen["status"] = "DEVELOPMENT_REQUIRED"
    with pytest.raises(ValueError, match="not frozen"):
        compare_semantic_traces((_event(),), (_event(),), unfrozen)
    with pytest.raises(ValueError, match="exceeds"):
        compare_semantic_traces((_event(),), (_event(),), _contract(margin=0.031))
