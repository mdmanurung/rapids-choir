from __future__ import annotations

import numpy as np
import pytest

from rapids_choir._enums import Decision, Multiplicity, ReasonCode, RuleSet
from rapids_choir.results import (
    ComparisonMetrics,
    ComparisonResult,
    FeatureCounts,
    MemoryRecord,
    SampleCounts,
    SeedRecord,
    TimingRecord,
)
from rapids_choir.tl._pruning import (
    PruningCluster,
    _dynamic_threshold,
    _split_survives,
    prune_hierarchy,
)
from rapids_choir.tree import HierarchyTree, hierarchy_from_partitions


def _result(
    decision: Decision,
    *,
    accuracy: float = 0.01,
    variance: float = 0.01,
    rules: RuleSet = RuleSet.CORRECTED,
) -> ComparisonResult:
    metrics = ComparisonMetrics(
        mean_observed_accuracy=0.8,
        variance_observed_accuracy=0.01,
        mean_null_accuracy=0.5,
        variance_null_accuracy=0.01,
        accuracy_reference_tail_probability=accuracy,
        variance_reference_tail_probability=variance,
        mean_modified_accuracy=0.8,
        variance_modified_accuracy=0.01,
        modified_accuracy_reference_tail_probability=accuracy,
        modified_variance_reference_tail_probability=variance,
    )
    return ComparisonResult(
        decision=decision,
        reason_code=(
            ReasonCode.SPLIT
            if decision is Decision.SPLIT
            else ReasonCode.MERGE
        ),
        rules=rules,
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


def _tree(partition: list[int]) -> HierarchyTree:
    return hierarchy_from_partitions(
        [partition],
        cell_ids=[f"cell-{index}" for index in range(len(partition))],
        resolutions=[0.5],
    )


def test_split_and_merge_produce_expected_final_labels() -> None:
    tree = _tree([0, 0, 1, 1])

    def compare_split(*args: object, **kwargs: object) -> ComparisonResult:
        return _result(Decision.SPLIT)

    split = prune_hierarchy(
        tree,
        compare_split,
        alpha=0.05,
        multiplicity="none",
        rules="corrected",
        use_variance=True,
    )
    assert split.labels[0] == split.labels[1]
    assert split.labels[2] == split.labels[3]
    assert split.labels[0] != split.labels[2]
    assert len(split.comparisons) == 1

    def compare_merge(*args: object, **kwargs: object) -> ComparisonResult:
        return _result(Decision.MERGE)

    merged = prune_hierarchy(
        tree,
        compare_merge,
        alpha=0.05,
        multiplicity="none",
        rules="corrected",
        use_variance=True,
    )
    assert merged.labels.tolist() == [1, 1, 1, 1]


def test_singletons_merge_without_calling_comparator() -> None:
    tree = _tree([0, 1, 1, 1])

    def forbidden(*args: object, **kwargs: object) -> ComparisonResult:
        raise AssertionError("singleton branch must short circuit")

    result = prune_hierarchy(
        tree,
        forbidden,
        alpha=0.05,
        multiplicity="none",
        rules="choir_v0_3_0",
        use_variance=True,
    )
    assert result.labels.tolist() == [1, 1, 1, 1]
    assert result.comparisons == ()


def test_non_transitive_merges_trigger_two_ordered_bridge_calls() -> None:
    tree = _tree([0, 0, 1, 1, 2, 2])
    calls: list[tuple[str, frozenset[int], frozenset[int]]] = []

    def compare(
        left: PruningCluster,
        right: PruningCluster,
        *,
        pair_kind: str,
        alpha: float,
    ) -> ComparisonResult:
        del alpha
        left_set = frozenset(left.members)
        right_set = frozenset(right.members)
        calls.append((pair_kind, left_set, right_set))
        pair = frozenset((left_set, right_set))
        split_pair = frozenset((frozenset({2, 3}), frozenset({4, 5})))
        if pair_kind == "sibling":
            return _result(
                Decision.SPLIT if pair == split_pair else Decision.MERGE
            )
        bridge_split = (
            left_set == frozenset({0, 1, 2, 3})
            and right_set == frozenset({4, 5})
        )
        return _result(Decision.SPLIT if bridge_split else Decision.MERGE)

    result = prune_hierarchy(
        tree,
        compare,
        alpha=0.05,
        multiplicity="none",
        rules="corrected",
        use_variance=True,
    )
    assert [kind for kind, _, _ in calls[:3]] == ["sibling"] * 3
    assert [kind for kind, _, _ in calls[3:5]] == ["bridge"] * 2
    assert len(np.unique(result.labels)) == 2
    assert result.labels[0] == result.labels[2]
    assert result.labels[0] != result.labels[4]


def test_compatibility_fdr_uses_documented_variance_candidate_quirk() -> None:
    accuracy = [0.001, 0.002, 0.03, 0.5]
    variance = [0.001, 0.2, 0.3, 0.4]
    results = [
        _result(Decision.SPLIT, accuracy=a, variance=v)
        for a, v in zip(accuracy, variance, strict=True)
    ]
    compatibility = _dynamic_threshold(
        results,
        multiplicity=Multiplicity.BH,
        alpha=0.05,
        previous=0.05,
        rules=RuleSet.CHOIR_V0_3_0,
    )
    corrected = _dynamic_threshold(
        results,
        multiplicity=Multiplicity.BH,
        alpha=0.05,
        previous=0.05,
        rules=RuleSet.CORRECTED,
    )
    assert compatibility == 0.0125
    assert corrected == 0.037500000000000006


def test_variance_toggle_changes_multiplicity_survival() -> None:
    result = _result(Decision.SPLIT, accuracy=0.01, variance=0.08)
    assert _split_survives(result, 0.05, use_variance=False)
    assert not _split_survives(result, 0.05, use_variance=True)


def test_overlapping_multiplicity_corrections_merge_closest_pair() -> None:
    tree = _tree([0, 0, 1, 1, 2, 2])

    def split(*args: object, **kwargs: object) -> ComparisonResult:
        return _result(Decision.SPLIT, accuracy=0.02, variance=0.02)

    def distance(left: PruningCluster, right: PruningCluster) -> float:
        pair = frozenset((frozenset(left.members), frozenset(right.members)))
        closest = frozenset((frozenset({2, 3}), frozenset({4, 5})))
        return 1.0 if pair == closest else 10.0

    result = prune_hierarchy(
        tree,
        split,
        alpha=0.05,
        multiplicity="bonferroni",
        rules="choir_v0_3_0",
        use_variance=True,
        distance=distance,
    )
    assert result.adjusted_alpha == pytest.approx(0.05 / 3)
    assert result.labels[2] == result.labels[4]
    assert result.labels[0] != result.labels[2]
    assert all(record.root_distance is not None for record in result.comparisons)


def test_corrected_final_tie_order_is_membership_stable() -> None:
    tree = _tree([0, 0, 1, 1])

    def split(*args: object, **kwargs: object) -> ComparisonResult:
        return _result(Decision.SPLIT)

    first = prune_hierarchy(
        tree,
        split,
        alpha=0.05,
        multiplicity="none",
        rules="corrected",
        use_variance=True,
    )
    second = prune_hierarchy(
        tree,
        split,
        alpha=0.05,
        multiplicity="none",
        rules="corrected",
        use_variance=True,
    )
    assert np.array_equal(first.labels, second.labels)
