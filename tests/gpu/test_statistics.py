from __future__ import annotations

import json
from pathlib import Path

import pytest

cp = pytest.importorskip("cupy")

from rapids_choir._stats import (  # noqa: E402
    benjamini_hochberg,
    bh_dynamic_threshold,
    bonferroni_threshold,
    tail_probabilities,
)

ROOT = Path(__file__).parents[2]


@pytest.mark.gpu
@pytest.mark.reference
def test_kde_cdf_and_fixed_bootstrap_match_r_fixture() -> None:
    fixture = json.loads(
        (ROOT / "validation/fixtures/generated/kde_bootstrap.json").read_text()
    )
    observed = cp.asarray(fixture["observed_accuracies"], dtype=cp.float64)
    null = cp.asarray(fixture["null_accuracies"], dtype=cp.float64)
    indices = cp.asarray(fixture["fixed_bootstrap_indices_1_based"], dtype=cp.int64) - 1
    accuracy_tail, variance_tail, bootstrapped = tail_probabilities(
        observed,
        null,
        bootstrap_seed=1,
        bootstrap_indices=indices,
    )
    cp.testing.assert_allclose(
        bootstrapped,
        cp.asarray(fixture["fixed_bootstrap_variances"]),
        rtol=1e-6,
        atol=1e-8,
    )
    assert float(accuracy_tail.item()) == pytest.approx(
        fixture["accuracy_upper_tail"], abs=1e-3
    )
    assert float(variance_tail.item()) == pytest.approx(
        fixture["variance_lower_tail"], abs=1e-3
    )


@pytest.mark.gpu
@pytest.mark.reference
def test_multiplicity_arithmetic_matches_r_fixture() -> None:
    fixture = json.loads(
        (ROOT / "validation/fixtures/generated/multiplicity.json").read_text()
    )
    accuracy = cp.asarray(fixture["accuracy_tail_scores"], dtype=cp.float64)
    variance = cp.asarray(fixture["variance_tail_scores"], dtype=cp.float64)
    cp.testing.assert_allclose(
        benjamini_hochberg(accuracy),
        cp.asarray(fixture["bh_adjusted_accuracy"]),
        rtol=1e-7,
        atol=1e-9,
    )
    cp.testing.assert_allclose(
        benjamini_hochberg(variance),
        cp.asarray(fixture["bh_adjusted_variance"]),
        rtol=1e-7,
        atol=1e-9,
    )
    assert bonferroni_threshold(fixture["alpha"], len(accuracy)) == pytest.approx(
        fixture["bonferroni_threshold"], rel=1e-7, abs=1e-9
    )
    for rules, key in (
        ("choir_v0_3_0", "choir_v0_3_0_fdr_threshold"),
        ("corrected", "corrected_fdr_threshold"),
    ):
        threshold = bh_dynamic_threshold(
            accuracy,
            variance,
            alpha=fixture["alpha"],
            previous_threshold=fixture["alpha"],
            rules=rules,
        )
        assert threshold == pytest.approx(fixture[key], rel=1e-7, abs=1e-9)
