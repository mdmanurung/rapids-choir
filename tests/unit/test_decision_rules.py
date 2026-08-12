from __future__ import annotations

import json
from pathlib import Path

import pytest

from rapids_choir._decision import decide_comparison
from rapids_choir._enums import REFERENCE_REASON_ALIASES, Decision
from rapids_choir.config import ComparisonConfig
from rapids_choir.results import ComparisonMetrics

ROOT = Path(__file__).parents[2]


def _metrics(record: dict[str, float]) -> ComparisonMetrics:
    return ComparisonMetrics(
        mean_observed_accuracy=record["mean_accuracy"],
        variance_observed_accuracy=0.01,
        mean_null_accuracy=0.5,
        variance_null_accuracy=0.01,
        accuracy_reference_tail_probability=record["accuracy_tail"],
        variance_reference_tail_probability=record["variance_tail"],
        mean_modified_accuracy=record["mean_accuracy"],
        variance_modified_accuracy=0.01,
        modified_accuracy_reference_tail_probability=record["modified_accuracy_tail"],
        modified_variance_reference_tail_probability=record["modified_variance_tail"],
        repeated_error_cells_cluster1=int(record["n_repeat_errors1"]),
        repeated_error_cells_cluster2=int(record["n_repeat_errors2"]),
    )


@pytest.mark.reference
def test_decision_order_matches_frozen_r_fixture() -> None:
    fixture = json.loads(
        (ROOT / "validation/fixtures/generated/decision_rules.json").read_text()
    )
    config = ComparisonConfig(rules="choir_v0_3_0")
    for case, inputs in fixture["inputs"].items():
        outcome = decide_comparison(
            _metrics(inputs), config, cluster1_size=20, cluster2_size=20
        )
        expected_decision, expected_reason = fixture["outputs"][case]
        assert outcome.decision is Decision(expected_decision)
        assert REFERENCE_REASON_ALIASES[outcome.reason_code] == expected_reason


def test_tail_threshold_is_strict() -> None:
    metrics = _metrics(
        {
            "mean_accuracy": 0.8,
            "accuracy_tail": 0.05,
            "variance_tail": 0.01,
            "modified_accuracy_tail": 0.05,
            "modified_variance_tail": 0.01,
            "n_repeat_errors1": 0,
            "n_repeat_errors2": 0,
        }
    )
    outcome = decide_comparison(
        metrics, ComparisonConfig(), cluster1_size=100, cluster2_size=100
    )
    assert outcome.decision is Decision.MERGE
