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
    "freeze_rf_profile", SCRIPTS / "freeze_rf_profile.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_profile_fidelity_scores_every_repeat_without_majority_vote() -> None:
    records = []
    for repeat in range(1, 6):
        records.append(
            {
                "dataset_id": "Simulated Dataset 1",
                "pair_id": "pair-1",
                "repeat": repeat,
                "profile_id": "bins128_depth16_streams1",
                "reference_decision": "merge",
                "reference_reason": "not_distinct",
                "gpu_decision": "split" if repeat == 1 else "merge",
                "gpu_reference_reason": (
                    "distinct" if repeat == 1 else "not_distinct"
                ),
                "reference_mean_observed_accuracy": 0.5,
                "gpu_mean_observed_accuracy": 0.5,
                "runtime_seconds": 1.0,
                "peak_device_bytes": 100,
                "estimated_device_bytes": 80,
            }
        )

    summary = MODULE.aggregate_profile(
        pd.DataFrame.from_records(records),
        {"n_bins": 128, "max_depth": 16, "n_streams": 1},
        {("Simulated Dataset 1", "pair-1"): [0.5] * 5},
    )

    assert summary["repeat_evaluation"] == "every_repeat"
    assert summary["macro_decision_reason_concordance"] == pytest.approx(0.8)
    assert summary["overall_decision_concordance"] == pytest.approx(0.8)
