from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import rapids_choir._sampling as sampling_module
from rapids_choir._enums import Decision, ReasonCode, RuleSet
from rapids_choir._sampling import balanced_split_round, eligible_batch_codes
from rapids_choir.results import (
    ComparisonMetrics,
    ComparisonResult,
    FeatureCounts,
    MemoryRecord,
    SampleCounts,
    SeedRecord,
    TimingRecord,
)
from rapids_choir.tl._pruning import PruningCluster, prune_hierarchy
from rapids_choir.tree import hierarchy_from_partitions


def test_batch_eligibility_is_an_intersection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampling_module, "_cupy", lambda: np)
    batches = np.asarray([0, 0, 1, 1, 1, 1, 2, 2], dtype=np.int64)
    left = np.asarray([0, 1, 2, 3], dtype=np.int64)
    right = np.asarray([4, 5, 6, 7], dtype=np.int64)
    assert eligible_batch_codes(batches, left, right).tolist() == [1]


def test_compatibility_sampling_preserves_r_call_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sampling_module, "_cupy", lambda: np)
    left = np.arange(0, 8, dtype=np.int64)
    right = np.arange(20, 28, dtype=np.int64)
    observed = balanced_split_round(
        left,
        right,
        n_per_half=2,
        seed=17,
        rules=RuleSet.CHOIR_V0_3_0,
    )
    generator = np.random.RandomState(17)
    expected_right = generator.permutation(right)[:4]
    expected_left = generator.permutation(left)[:4]
    assert observed.row_indices.tolist() == np.concatenate(
        (expected_left, expected_right)
    ).tolist()


def _split_result() -> ComparisonResult:
    metrics = ComparisonMetrics(
        mean_observed_accuracy=0.8,
        variance_observed_accuracy=0.01,
        mean_null_accuracy=0.5,
        variance_null_accuracy=0.01,
        accuracy_reference_tail_probability=0.01,
        variance_reference_tail_probability=0.01,
    )
    return ComparisonResult(
        decision=Decision.SPLIT,
        reason_code=ReasonCode.SPLIT,
        rules=RuleSet.CORRECTED,
        cluster1_id="left",
        cluster2_id="right",
        connectivity=1.0,
        metrics=metrics,
        samples=SampleCounts(4, 4, 2, 2, 2, 2, 8),
        features=FeatureCounts(10, 5),
        seeds=SeedRecord(1, 2),
        timings=TimingRecord(0.0),
        memory=MemoryRecord(0, 0, 0, 0),
    )


def test_bridge_rejects_the_more_distant_partner() -> None:
    tree = hierarchy_from_partitions(
        [[0, 0, 1, 1, 2, 2]],
        cell_ids=[f"cell-{index}" for index in range(6)],
        resolutions=[0.5],
    )

    def compare(
        left: PruningCluster,
        right: PruningCluster,
        *,
        pair_kind: str,
        alpha: float,
    ) -> ComparisonResult:
        del alpha
        if pair_kind == "bridge":
            return _split_result()
        pair = frozenset((frozenset(left.members), frozenset(right.members)))
        separated = frozenset((frozenset({2, 3}), frozenset({4, 5})))
        result = _split_result()
        if pair != separated:
            return replace(
                result,
                decision=Decision.MERGE,
                reason_code=ReasonCode.MERGE,
            )
        return result

    def distance(left: PruningCluster, right: PruningCluster) -> float:
        pair = frozenset((frozenset(left.members), frozenset(right.members)))
        distant = frozenset((frozenset({0, 1}), frozenset({2, 3})))
        return 10.0 if pair == distant else 1.0

    result = prune_hierarchy(
        tree,
        compare,
        alpha=0.05,
        multiplicity="none",
        rules="corrected",
        use_variance=True,
        distance=distance,
    )
    assert result.labels[0] == result.labels[4]
    assert result.labels[0] != result.labels[2]
