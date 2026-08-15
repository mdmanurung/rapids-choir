from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rapids_choir._enums import Decision, ReasonCode, RuleSet
from rapids_choir.results import (
    ComparisonMetrics,
    ComparisonResult,
    FeatureCounts,
    MemoryRecord,
    SampleCounts,
    SeedRecord,
    TimingRecord,
)
from rapids_choir.tl._hierarchy import (
    ClusteringFailure,
    build_automatic_hierarchy,
    build_overclustered_subtree,
    find_starting_resolution,
    optimize_hierarchy,
    relabel_singletons,
)
from rapids_choir.tl._pruning import PruningCluster
from rapids_choir.tree import HierarchyTree, hierarchy_from_partitions


def _comparison(decision: Decision) -> ComparisonResult:
    return ComparisonResult(
        decision=decision,
        reason_code=(
            ReasonCode.SPLIT if decision is Decision.SPLIT else ReasonCode.MERGE
        ),
        rules=RuleSet.CORRECTED,
        cluster1_id="left",
        cluster2_id="right",
        connectivity=1.0,
        metrics=ComparisonMetrics(0.8, 0.01, 0.5, 0.01, 0.01, 0.01),
        samples=SampleCounts(4, 4, 2, 2, 2, 2, 8),
        features=FeatureCounts(10, 5),
        seeds=SeedRecord(1, 2),
        timings=TimingRecord(0.0),
        memory=MemoryRecord(0, 0, 0, 0),
    )


def test_hierarchy_intersects_non_nested_partitions_and_round_trips() -> None:
    partitions = [
        np.asarray([0, 0, 0, 0, 0, 0]),
        np.asarray([0, 0, 0, 1, 1, 1]),
        np.asarray([0, 0, 1, 1, 1, 2]),
    ]
    tree = hierarchy_from_partitions(
        partitions,
        cell_ids=[f"cell-{index}" for index in range(6)],
        resolutions=[0.0, 0.5, 1.0],
    )
    assert tree.n_levels == 3
    assert tree.resolutions == (None, 0.5, 1.0)
    assert [node.size for node in tree.leaves] == [2, 1, 2, 1]
    record = tree.to_record()
    assert all(
        isinstance(node_id, str)
        for assignment in record["assignments"]
        for node_id in assignment
    )
    assert HierarchyTree.from_record(record) == tree


def test_node_identities_ignore_backend_label_spelling() -> None:
    first = hierarchy_from_partitions(
        [[0, 0, 1, 1]], cell_ids=list("abcd"), resolutions=[0.5]
    )
    second = hierarchy_from_partitions(
        [[9, 9, 4, 4]], cell_ids=list("abcd"), resolutions=[0.5]
    )
    assert first.assignments == second.assignments
    assert first.nodes == second.nodes


def test_singleton_relabelling_is_deterministic() -> None:
    values = relabel_singletons(["cluster", "singleton", "singleton"])
    assert len(np.unique(values)) == 3
    repeated = relabel_singletons(["cluster", "singleton", "singleton"])
    assert np.array_equal(values, repeated)


def test_starting_resolution_searches_upward() -> None:
    def partitioner(resolution: float, members: np.ndarray[Any, Any], seed: int) -> Any:
        del seed
        if resolution < 1.5:
            return np.zeros(members.size)
        return np.arange(members.size) % 2

    result = find_starting_resolution(
        partitioner, np.arange(8), random_state=7
    )
    assert result.resolution == 1.5
    assert [step.action for step in result.steps] == [
        "baseline",
        "probe",
        "probe_up",
    ]


def test_starting_resolution_searches_downward() -> None:
    def partitioner(resolution: float, members: np.ndarray[Any, Any], seed: int) -> Any:
        del seed
        if resolution >= 0.7:
            modulus = 3
        elif resolution >= 0.4:
            modulus = 2
        else:
            modulus = 1
        return np.arange(members.size) % modulus

    result = find_starting_resolution(
        partitioner, np.arange(12), random_state=7
    )
    assert result.resolution == 0.4
    assert [step.resolution for step in result.steps[-3:]] == [0.7, 0.4, 0.1]


def test_resolution_one_failure_uses_reference_fallback() -> None:
    def partitioner(resolution: float, members: np.ndarray[Any, Any], seed: int) -> Any:
        del seed
        if resolution == 1.0:
            raise ClusteringFailure("fixture failure")
        return np.zeros(members.size)

    result = find_starting_resolution(
        partitioner, np.arange(4), random_state=7
    )
    assert result.resolution == 0.1
    assert result.decimal_places == 2
    assert result.steps[-1].reason == "resolution_1_failed"


def test_hierarchy_rejects_misaligned_partitions() -> None:
    with pytest.raises(ValueError, match="align"):
        hierarchy_from_partitions([[0, 1]], cell_ids=["a", "b", "c"])


def test_tree_optimization_inserts_complete_linkage_levels_for_wide_split() -> None:
    partition = np.repeat(np.arange(5), 2)
    tree = hierarchy_from_partitions(
        [partition],
        cell_ids=[f"cell-{index}" for index in range(partition.size)],
        resolutions=[0.5],
    )
    representation = np.column_stack(
        (np.repeat([0.0, 1.0, 10.0, 11.0, 30.0], 2), np.arange(10))
    )
    optimized = optimize_hierarchy(
        tree, representation, rules="corrected"
    )
    assert optimized.n_levels == 5
    assert len(optimized.leaves) == 5
    assert {node.membership_sha256 for node in optimized.leaves} == {
        node.membership_sha256 for node in tree.leaves
    }
    assert all(
        len(set(optimized.assignments[index]))
        <= len(set(optimized.assignments[index + 1]))
        for index in range(optimized.n_levels - 1)
    )


def test_tree_optimization_leaves_three_way_split_unchanged() -> None:
    tree = hierarchy_from_partitions(
        [np.repeat(np.arange(3), 2)],
        cell_ids=[f"cell-{index}" for index in range(6)],
        resolutions=[0.5],
    )
    optimized = optimize_hierarchy(
        tree, np.arange(12, dtype=float).reshape(6, 2), rules="choir_v0_3_0"
    )
    assert optimized == tree


def test_overclustered_subtree_stops_on_furthest_neighbor_merge() -> None:
    def partitioner(resolution: float, members: np.ndarray[Any, Any], seed: int) -> Any:
        del seed
        if resolution < 0.4:
            return np.zeros(members.size, dtype=np.int64)
        return np.arange(members.size) % 2

    calls: list[str] = []

    def compare(
        left: PruningCluster,
        right: PruningCluster,
        *,
        pair_kind: str,
        alpha: float,
    ) -> ComparisonResult:
        del left, right, alpha
        calls.append(pair_kind)
        return _comparison(Decision.MERGE)

    result = build_overclustered_subtree(
        partitioner,
        members=np.arange(8),
        cell_ids=[f"cell-{index}" for index in range(8)],
        representation=np.arange(16, dtype=float).reshape(8, 2),
        compare=compare,
        min_cluster_depth=2,
        alpha=0.05,
        random_state=7,
    )
    assert calls == ["subtree_furthest"]
    assert result.stop_reasons[-1] == "furthest_neighbors_merge"
    assert len(result.tree.leaves) == 2
    assert len(result.comparisons) == 1


def test_automatic_hierarchy_uses_minimum_depth_subtree_path() -> None:
    def partitioner(resolution: float, members: np.ndarray[Any, Any], seed: int) -> Any:
        del seed
        if resolution < 0.4:
            return np.zeros(members.size, dtype=np.int64)
        if resolution < 2.0:
            return np.arange(members.size) % 2
        return np.arange(members.size) % max(1, members.size // 2)

    def compare(
        left: PruningCluster,
        right: PruningCluster,
        *,
        pair_kind: str,
        alpha: float,
    ) -> ComparisonResult:
        del left, right, pair_kind, alpha
        return _comparison(Decision.MERGE)

    result = build_automatic_hierarchy(
        partitioner,
        cell_ids=[f"cell-{index}" for index in range(8)],
        representation=np.arange(16, dtype=float).reshape(8, 2),
        silhouette=lambda labels, members: float(len(np.unique(labels))),
        compare=compare,
        max_clusters=None,
        min_cluster_depth=10,
        multiplicity="bonferroni",
        alpha=0.05,
        random_state=7,
        max_steps=64,
    )
    assert result.stop_reason == "automatic_subtrees_complete"
    assert len(result.tree.leaves) == 2
    assert result.tree.n_cells == 8
