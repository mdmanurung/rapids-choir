from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "assemble_gate2", SCRIPTS / "assemble_gate2.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

MARGINS = {
    "mean_observed_accuracy": 0.02,
    "variance_observed_accuracy": 0.01,
}


def result_frame() -> pd.DataFrame:
    records = []
    for repeat in range(1, 6):
        records.append(
            {
                "dataset_id": "Simulated Dataset 5",
                "pair_id": "pair-1",
                "repeat": repeat,
                "pair_kind": "null",
                "expected_decision": "merge",
                "reference_pair_kind": "null",
                "reference_expected_decision": "merge",
                "reference_decision": "merge",
                "reference_reason": "not_distinct",
                "gpu_decision": "merge",
                "gpu_reference_reason": "not_distinct",
                "reference_mean_observed_accuracy": 0.50,
                "gpu_mean_observed_accuracy": 0.51,
                "reference_variance_observed_accuracy": None,
                "gpu_variance_observed_accuracy": None,
            }
        )
    return pd.DataFrame.from_records(records)


def test_gate2_requires_every_repeat_and_all_bounded_fields() -> None:
    aggregated = MODULE.aggregate_pairs(result_frame(), MARGINS)

    assert len(aggregated) == 1
    assert bool(aggregated.loc[0, "outcome_exact_all_repeats"])
    assert bool(aggregated.loc[0, "all_bounded_metrics_within_margin"])
    assert bool(aggregated.loc[0, "pair_pass"])
    assert aggregated.loc[0, "max_abs_difference_mean_observed_accuracy"] == (
        pytest.approx(0.01)
    )


def test_gate2_does_not_majority_vote_deterministic_failures() -> None:
    frame = result_frame()
    frame.loc[0, "gpu_reference_reason"] = "distinct"

    aggregated = MODULE.aggregate_pairs(frame, MARGINS)

    assert not bool(aggregated.loc[0, "outcome_exact_all_repeats"])
    assert aggregated.loc[0, "outcome_exact_repeat_count"] == 4
    assert not bool(aggregated.loc[0, "pair_pass"])


def test_gate2_fails_one_repeat_outside_development_margin() -> None:
    frame = result_frame()
    frame.loc[4, "gpu_mean_observed_accuracy"] = 0.521

    aggregated = MODULE.aggregate_pairs(frame, MARGINS)

    assert not bool(
        aggregated.loc[0, "within_margin_mean_observed_accuracy"]
    )
    assert not bool(aggregated.loc[0, "pair_pass"])


def test_gate2_requires_reference_and_gpu_columns_for_every_metric() -> None:
    frame = result_frame().drop(columns=["gpu_variance_observed_accuracy"])

    with pytest.raises(RuntimeError, match="lack required v2 fields"):
        MODULE.aggregate_pairs(frame, MARGINS)
