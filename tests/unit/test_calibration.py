from __future__ import annotations

import pytest

from rapids_choir._calibration import (
    comparator_acceptance_checks,
    macro_decision_concordance,
    majority_decision,
    rank_profiles,
    wilson_lower_bound,
)


def test_wilson_lower_bound_matches_known_interval() -> None:
    assert wilson_lower_bound(95, 100) == pytest.approx(0.8882495, abs=1e-7)
    with pytest.raises(ValueError):
        wilson_lower_bound(1, 0)


def test_majority_and_macro_concordance() -> None:
    assert majority_decision(["split", "merge", "split", "split", "merge"]) == "split"
    assert macro_decision_concordance(
        ["merge", "merge", "split", "split"],
        ["merge", "split", "split", "split"],
    ) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="odd"):
        majority_decision(["merge", "split"])


def summary(
    *,
    profile_id: str,
    concordance: float,
    median_error: float,
    p95_error: float,
    runtime: float,
    memory: int,
) -> dict[str, object]:
    return {
        "profile_id": profile_id,
        "profile": {"n_bins": 128, "max_depth": 16, "n_streams": 1},
        "macro_decision_concordance": concordance,
        "median_abs_observed_accuracy_difference": median_error,
        "p95_abs_observed_accuracy_difference": p95_error,
        "median_runtime_seconds": runtime,
        "median_peak_device_bytes": memory,
    }


def test_profile_ranking_is_frozen_lexicographic_order() -> None:
    ranked = rank_profiles(
        [
            summary(
                profile_id="fast_but_wrong",
                concordance=0.9,
                median_error=0.01,
                p95_error=0.01,
                runtime=1,
                memory=1,
            ),
            summary(
                profile_id="accurate_slow",
                concordance=1.0,
                median_error=0.02,
                p95_error=0.03,
                runtime=10,
                memory=10,
            ),
            summary(
                profile_id="accurate_precise",
                concordance=1.0,
                median_error=0.01,
                p95_error=0.04,
                runtime=20,
                memory=20,
            ),
        ]
    )
    assert [record["profile_id"] for record in ranked] == [
        "accurate_precise",
        "accurate_slow",
        "fast_but_wrong",
    ]
    assert [record["rank"] for record in ranked] == [1, 2, 3]


def test_acceptance_requires_every_threshold() -> None:
    thresholds = {
        "overall_decision_concordance_min": 0.95,
        "per_dataset_decision_concordance_min": 0.90,
        "overall_wilson_lower_95_min": 0.92,
        "median_abs_observed_accuracy_difference_max": 0.05,
        "p95_abs_observed_accuracy_difference_max": 0.10,
        "unapproved_post_ingest_transfer_bytes_max": 1_048_576,
        "vram_headroom_min": 0.20,
    }
    checks = comparator_acceptance_checks(
        overall_concordance=0.96,
        per_dataset_concordance={"a": 0.91, "b": 0.95},
        wilson_lower_95=0.93,
        median_accuracy_difference=0.04,
        p95_accuracy_difference=0.09,
        maximum_unapproved_post_ingest_transfer_bytes=1_048_576,
        minimum_vram_headroom=0.20,
        thresholds=thresholds,
    )
    assert all(checks.values())
    checks["vram_headroom"] = comparator_acceptance_checks(
        overall_concordance=0.96,
        per_dataset_concordance={"a": 0.91},
        wilson_lower_95=0.93,
        median_accuracy_difference=0.04,
        p95_accuracy_difference=0.09,
        maximum_unapproved_post_ingest_transfer_bytes=0,
        minimum_vram_headroom=0.199,
        thresholds=thresholds,
    )["vram_headroom"]
    assert not checks["vram_headroom"]
