from __future__ import annotations

import pytest

from rapids_choir._calibration import (
    accuracy_distribution_error,
    bounded_metric_margin,
    comparator_acceptance_checks,
    macro_decision_concordance,
    macro_decision_reason_concordance,
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


def test_joint_decision_reason_concordance_and_distribution_error() -> None:
    assert macro_decision_reason_concordance(
        ["merge", "merge", "split", "split"],
        ["merge", "merge", "split", "split"],
        ["default", "min_accuracy", "default", "default"],
        ["default", "default", "default", "default"],
    ) == pytest.approx(2 / 3)
    assert accuracy_distribution_error([0.4, 0.6], [0.4, 0.6]) == 0
    assert accuracy_distribution_error([0.4, 0.6], [0.5, 0.7]) == pytest.approx(
        0.1
    )


def test_reference_variability_margin_fails_closed_above_cap() -> None:
    passing = bounded_metric_margin([0.0, 0.01, 0.02], maximum=0.03)
    assert passing["eligible"] is True
    assert passing["margin"] <= 0.03
    failing = bounded_metric_margin([0.01, 0.04], quantile=0.975, maximum=0.03)
    assert failing["eligible"] is False
    assert failing["margin"] == 0.03


def summary(
    *,
    profile_id: str,
    concordance: float,
    distribution_error: float,
    runtime: float,
    n_bins: int = 128,
    max_depth: int | None = 16,
    n_streams: int = 1,
) -> dict[str, object]:
    return {
        "profile_id": profile_id,
        "profile": {
            "n_bins": n_bins,
            "max_depth": max_depth,
            "n_streams": n_streams,
        },
        "macro_decision_reason_concordance": concordance,
        "median_accuracy_distribution_error": distribution_error,
        "median_runtime_seconds": runtime,
    }


def test_profile_ranking_is_frozen_lexicographic_order() -> None:
    ranked = rank_profiles(
        [
            summary(
                profile_id="fast_but_wrong",
                concordance=0.9,
                distribution_error=0.01,
                runtime=1,
            ),
            summary(
                profile_id="accurate_slow",
                concordance=1.0,
                distribution_error=0.02,
                runtime=10,
            ),
            summary(
                profile_id="accurate_precise",
                concordance=1.0,
                distribution_error=0.01,
                runtime=20,
            ),
        ]
    )
    assert [record["profile_id"] for record in ranked] == [
        "accurate_precise",
        "accurate_slow",
        "fast_but_wrong",
    ]
    assert [record["rank"] for record in ranked] == [1, 2, 3]


def test_profile_ties_prefer_smaller_bins_depth_16_and_fewer_streams() -> None:
    tied = [
        summary(
            profile_id="unlimited",
            concordance=1,
            distribution_error=0,
            runtime=1,
            max_depth=None,
        ),
        summary(
            profile_id="more_streams",
            concordance=1,
            distribution_error=0,
            runtime=1,
            n_streams=4,
        ),
        summary(
            profile_id="larger_bins",
            concordance=1,
            distribution_error=0,
            runtime=1,
            n_bins=256,
        ),
        summary(
            profile_id="winner",
            concordance=1,
            distribution_error=0,
            runtime=1,
        ),
    ]
    assert [record["profile_id"] for record in rank_profiles(tied)] == [
        "winner",
        "more_streams",
        "unlimited",
        "larger_bins",
    ]


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
