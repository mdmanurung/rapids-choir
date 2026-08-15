from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from validation.scripts.verify_frozen_rf_profile import verify

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


def test_final_candidate_must_reproduce_promoted_profile_winner(
    tmp_path: Path,
) -> None:
    def profile(candidate_marker: str) -> dict[str, Any]:
        rankings = [
            {
                "profile_id": f"profile-{index:02d}",
                "profile": {
                    "n_bins": 128 if index == 0 else 256,
                    "max_depth": 16,
                    "n_streams": 1,
                },
            }
            for index in range(12)
        ]
        return {
            "schema_version": 1,
            "scope": "development_rf_calibration",
            "status": "FROZEN",
            "calibration_candidate": {
                "commit": candidate_marker * 40,
                "attempt": 1,
                "runtime_tree_sha256": candidate_marker * 64,
                "manifest_sha256": "c" * 64,
            },
            "ranking_order": ["fidelity", "runtime"],
            "rankings": rankings,
            "winner": rankings[0],
            "comparator_contract": {"sha256": "d" * 64},
            "environment_locks": {"conda": "e" * 64, "pip": "f" * 64},
            "reference_variability": {
                "maximum_margin": 0.03,
                "bounded_metric_margins": {
                    "mean_observed_accuracy": {
                        "margin": 0.01,
                        "eligible": True,
                    }
                },
            },
        }

    frozen = tmp_path / "frozen.json"
    reproduced = tmp_path / "reproduced.json"
    frozen.write_text(json.dumps(profile("a")), encoding="utf-8")
    reproduced.write_text(json.dumps(profile("b")), encoding="utf-8")
    report = verify(frozen, reproduced)
    assert report["status"] == "REPRODUCED"
    assert report["candidate"]["commit"] == "b" * 40

    drifted = profile("b")
    drifted["rankings"] = [
        drifted["rankings"][1],
        drifted["rankings"][0],
        *drifted["rankings"][2:],
    ]
    drifted["winner"] = drifted["rankings"][0]
    reproduced.write_text(json.dumps(drifted), encoding="utf-8")
    assert verify(frozen, reproduced)["status"] == "DRIFTED"

    lock_drifted = profile("b")
    lock_drifted["environment_locks"] = {
        "conda": "f" * 64,
        "pip": "e" * 64,
    }
    reproduced.write_text(json.dumps(lock_drifted), encoding="utf-8")
    report = verify(frozen, reproduced)
    assert report["status"] == "DRIFTED"
    assert report["protocol_checks"]["environment_lock_sha256"] is False
