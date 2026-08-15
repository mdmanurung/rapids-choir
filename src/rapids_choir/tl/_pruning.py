"""Bottom-up CHOIR pruning over an immutable hierarchy."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from .._enums import Decision, Multiplicity, ReasonCode, RuleSet
from .._hashing import membership_sha256
from .._semantic_trace import TraceRecorder
from ..results import ComparisonResult
from ..tree import HierarchyTree


@dataclass(frozen=True, slots=True)
class PruningCluster:
    label: str
    members: tuple[int, ...]
    membership_sha256: str

    @classmethod
    def create(cls, label: str, members: Any) -> PruningCluster:
        canonical = tuple(sorted(int(value) for value in members))
        return cls(label, canonical, membership_sha256(np.asarray(canonical)))


class CompareClusters(Protocol):
    def __call__(
        self,
        left: PruningCluster,
        right: PruningCluster,
        *,
        pair_kind: str,
        alpha: float,
    ) -> ComparisonResult: ...


Distance = Callable[[PruningCluster, PruningCluster], float]


@dataclass(frozen=True, slots=True)
class PruningResult:
    labels: np.ndarray[Any, np.dtype[np.int64]]
    record_labels: tuple[str, ...]
    comparisons: tuple[ComparisonResult, ...]
    adjusted_alpha: float
    stepwise_labels: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class _MultiplicityCorrection:
    result: ComparisonResult
    left_hash: str
    right_hash: str
    order: int


def _clusters_from_assignment(values: np.ndarray[Any, Any]) -> list[PruningCluster]:
    order = list(dict.fromkeys(str(value) for value in values.tolist()))
    return [
        PruningCluster.create(label, np.flatnonzero(values == label))
        for label in order
    ]


def _bounded_metrics(result: ComparisonResult) -> dict[str, float]:
    if result.metrics is None:
        return {}
    metrics = result.metrics
    values = {
        "mean_observed_accuracy": metrics.mean_observed_accuracy,
        "variance_observed_accuracy": metrics.variance_observed_accuracy,
        "mean_null_accuracy": metrics.mean_null_accuracy,
        "variance_null_accuracy": metrics.variance_null_accuracy,
        "accuracy_reference_tail_probability": (
            metrics.accuracy_reference_tail_probability
        ),
        "variance_reference_tail_probability": (
            metrics.variance_reference_tail_probability
        ),
    }
    optional = {
        "mean_modified_accuracy": metrics.mean_modified_accuracy,
        "variance_modified_accuracy": metrics.variance_modified_accuracy,
        "modified_accuracy_reference_tail_probability": (
            metrics.modified_accuracy_reference_tail_probability
        ),
        "modified_variance_reference_tail_probability": (
            metrics.modified_variance_reference_tail_probability
        ),
    }
    values.update(
        {name: value for name, value in optional.items() if value is not None}
    )
    return {
        name: min(1.0, max(0.0, float(value))) for name, value in values.items()
    }


def _connected_components(
    clusters: list[PruningCluster],
    decisions: dict[tuple[str, str], Decision],
) -> list[list[PruningCluster]]:
    by_hash = {cluster.membership_sha256: cluster for cluster in clusters}
    order = {
        cluster.membership_sha256: index
        for index, cluster in enumerate(clusters)
    }
    adjacency: dict[str, set[str]] = {key: set() for key in by_hash}
    for (left, right), decision in decisions.items():
        if decision is Decision.MERGE:
            adjacency[left].add(right)
            adjacency[right].add(left)
    components: list[list[PruningCluster]] = []
    seen: set[str] = set()
    for cluster in clusters:
        key = cluster.membership_sha256
        if key in seen:
            continue
        pending = [key]
        component: list[PruningCluster] = []
        seen.add(key)
        while pending:
            current = pending.pop()
            component.append(by_hash[current])
            for neighbor in sorted(adjacency[current], key=order.__getitem__):
                if neighbor not in seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
        component.sort(key=lambda cluster: order[cluster.membership_sha256])
        components.append(component)
    return components


def _root_prefix(labels: list[str]) -> str:
    roots = [match.group(0) for label in labels if (match := re.match(r"P\d+", label))]
    return roots[0] if roots and len(set(roots)) == 1 else "P0"


def _new_merge_label(
    clusters: list[PruningCluster],
    *,
    level: int,
    counters: dict[tuple[str, int], int],
) -> str:
    if len(clusters) == 1:
        return clusters[0].label
    root = _root_prefix([cluster.label for cluster in clusters])
    key = (root, level)
    counters[key] = counters.get(key, 0) + 1
    return f"{root}_L{level}_{counters[key]}"


def _comparison_key(
    left: PruningCluster, right: PruningCluster
) -> tuple[str, str]:
    first, second = sorted(
        (left.membership_sha256, right.membership_sha256)
    )
    return first, second


def _bh_candidate(scores: list[float], *, alpha: float, previous: float) -> float:
    if not scores:
        return previous
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    passing = ranked[adjusted < alpha]
    if passing.size == 0:
        return previous
    largest = float(passing.max())
    last_rank = int(np.flatnonzero(np.sort(values) == largest)[-1]) + 1
    return last_rank / values.size * alpha


def _dynamic_threshold(
    results: list[ComparisonResult],
    *,
    multiplicity: Multiplicity,
    alpha: float,
    previous: float,
    rules: RuleSet,
) -> float:
    accuracy = [
        result.metrics.accuracy_reference_tail_probability
        for result in results
        if result.metrics is not None
    ]
    if not accuracy:
        return previous
    if multiplicity is Multiplicity.BONFERRONI:
        return alpha / len(accuracy)
    variance = [
        result.metrics.variance_reference_tail_probability
        for result in results
        if result.metrics is not None
    ]
    accuracy_candidate = _bh_candidate(accuracy, alpha=alpha, previous=previous)
    variance_candidate = _bh_candidate(variance, alpha=alpha, previous=previous)
    if rules is RuleSet.CHOIR_V0_3_0:
        combined = variance_candidate
    else:
        combined = max(accuracy_candidate, variance_candidate)
    return min(previous, combined)


def _split_survives(
    result: ComparisonResult,
    threshold: float,
    *,
    use_variance: bool,
) -> bool:
    metrics = result.metrics
    if metrics is None or result.decision is not Decision.SPLIT:
        return False
    if result.reason_code is ReasonCode.SPLIT_REPEAT_ERROR:
        accuracy = metrics.modified_accuracy_reference_tail_probability
        variance = metrics.modified_variance_reference_tail_probability
    else:
        accuracy = metrics.accuracy_reference_tail_probability
        variance = metrics.variance_reference_tail_probability
    return bool(
        accuracy is not None
        and accuracy < threshold
        and (not use_variance or (variance is not None and variance < threshold))
    )


def _record_comparison(
    result: ComparisonResult,
    left: PruningCluster,
    right: PruningCluster,
    pair_kind: str,
    trace: TraceRecorder | None,
    *,
    parent_label: str,
    level: int,
    root_distance: float | None = None,
    subtree_distance: float | None = None,
) -> ComparisonResult:
    recorded = replace(
        result,
        pair_kind=pair_kind,
        input_membership_sha256=(
            left.membership_sha256,
            right.membership_sha256,
        ),
        root_distance=root_distance,
        subtree_distance=subtree_distance,
    )
    if trace is not None:
        trace.append(
            "comparator_call",
            "pruning",
            parameters={
                "cluster1": left.label,
                "cluster2": right.label,
                "pair_kind": pair_kind,
            },
            seed=result.seeds.operation,
            input_membership_hashes=(
                left.membership_sha256,
                right.membership_sha256,
            ),
            topology_context={"parent": parent_label, "level": level},
            decision=result.decision.value,
            reason=result.reason_code.value,
            bounded_values=_bounded_metrics(result),
            payload={
                "connectivity": result.connectivity,
                "root_distance": root_distance,
                "subtree_distance": subtree_distance,
                "round_seeds": list(result.seeds.round_seeds),
                "fit_seeds": list(result.seeds.fit_seeds),
            },
        )
    return recorded


def _select_multiplicity_correction(
    corrections: list[_MultiplicityCorrection],
    current_clusters: dict[str, PruningCluster],
    distance: Distance | None,
) -> _MultiplicityCorrection:
    """Select the R correction edge for an overlapping set of failed splits."""

    first = corrections[0]
    endpoints = [
        membership_hash
        for correction in corrections
        for membership_hash in (correction.left_hash, correction.right_hash)
    ]
    if endpoints.count(first.left_hash) == 1 and endpoints.count(first.right_hash) == 1:
        return first
    incident = [
        correction
        for correction in corrections
        if first.left_hash in (correction.left_hash, correction.right_hash)
        or first.right_hash in (correction.left_hash, correction.right_hash)
    ]
    if distance is not None:
        subtree_values = [
            correction.result.subtree_distance
            for correction in incident
            if correction.result.subtree_distance is not None
        ]
        if subtree_values:
            return min(
                incident,
                key=lambda correction: (
                    float("inf")
                    if correction.result.subtree_distance is None
                    else correction.result.subtree_distance,
                    correction.order,
                ),
            )
        root_values = [
            correction.result.root_distance
            for correction in incident
            if correction.result.root_distance is not None
        ]
        if root_values:
            return min(
                incident,
                key=lambda correction: (
                    float("inf")
                    if correction.result.root_distance is None
                    else correction.result.root_distance,
                    correction.order,
                ),
            )
        return min(
            incident,
            key=lambda correction: (
                distance(
                    current_clusters[correction.left_hash],
                    current_clusters[correction.right_hash],
                ),
                correction.order,
            ),
        )
    return min(
        incident,
        key=lambda correction: (
            float("inf")
            if correction.result.metrics is None
            else correction.result.metrics.mean_observed_accuracy,
            correction.order,
        ),
    )


def prune_hierarchy(
    tree: HierarchyTree,
    compare: CompareClusters,
    *,
    alpha: float,
    multiplicity: Multiplicity | str,
    rules: RuleSet | str,
    use_variance: bool,
    distance: Distance | None = None,
    trace: TraceRecorder | None = None,
) -> PruningResult:
    """Prune siblings bottom-up, including bridge and multiplicity transitions."""

    method = Multiplicity(multiplicity)
    rule_set = RuleSet(rules)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    node_map = tree.node_map
    current = np.asarray(tree.labels(-1), dtype=object)
    comparisons: list[ComparisonResult] = []
    cache: dict[tuple[str, str], ComparisonResult] = {}
    stepwise: list[tuple[str, ...]] = [tuple(str(value) for value in current)]
    counters: dict[tuple[str, int], int] = {}
    for node in tree.nodes:
        match = re.fullmatch(r"(P\d+)_L(\d+)_(\d+)", node.label)
        if match:
            counter_key = (match.group(1), int(match.group(2)))
            counters[counter_key] = max(
                counters.get(counter_key, 0), int(match.group(3))
            )
    adjusted_alpha = alpha

    for parent_level in range(tree.n_levels - 2, -1, -1):
        parent_assignment = tree.assignments[parent_level]
        parent_order = list(dict.fromkeys(parent_assignment))
        for parent_id in parent_order:
            parent = node_map[parent_id]
            parent_members = np.flatnonzero(
                np.asarray(parent_assignment, dtype=object) == parent_id
            )
            local_values = current[parent_members]
            local_labels = list(dict.fromkeys(str(value) for value in local_values))
            clusters = [
                PruningCluster.create(
                    label,
                    parent_members[np.flatnonzero(local_values == label)],
                )
                for label in local_labels
            ]
            if len(clusters) < 2:
                continue
            decisions: dict[tuple[str, str], Decision] = {}
            for left_index, left in enumerate(clusters[:-1]):
                for right in clusters[left_index + 1 :]:
                    pair_key = _comparison_key(left, right)
                    if len(left.members) == 1 or len(right.members) == 1:
                        decisions[pair_key] = Decision.MERGE
                        if trace is not None:
                            singleton = left if len(left.members) == 1 else right
                            trace.append(
                                "singleton",
                                "pruning",
                                parameters={
                                    "cluster": singleton.label,
                                    "pair_kind": "sibling",
                                },
                                input_membership_hashes=(
                                    left.membership_sha256,
                                    right.membership_sha256,
                                ),
                                topology_context={
                                    "parent": parent.label,
                                    "level": parent_level + 1,
                                },
                                decision="merge",
                                reason="singleton",
                            )
                        continue
                    result = cache.get(pair_key)
                    if result is None:
                        result = compare(
                            left,
                            right,
                            pair_kind="sibling",
                            alpha=adjusted_alpha,
                        )
                        result = _record_comparison(
                            result,
                            left,
                            right,
                            "sibling",
                            trace,
                            parent_label=parent.label,
                            level=parent_level + 1,
                            root_distance=(
                                None if distance is None else distance(left, right)
                            ),
                        )
                        comparisons.append(result)
                        cache[pair_key] = result
                    decisions[pair_key] = result.decision

            # Resolve non-transitive A-B merge, A-C merge, B-C split patterns
            # with the two bridge comparisons used by the R implementation.
            for child in clusters:
                partners = [
                    other
                    for other in clusters
                    if other is not child
                    and decisions[_comparison_key(child, other)] is Decision.MERGE
                ]
                if len(partners) < 2:
                    continue
                for first_index, first in enumerate(partners[:-1]):
                    for second in partners[first_index + 1 :]:
                        if (
                            decisions[_comparison_key(first, second)]
                            is not Decision.SPLIT
                        ):
                            continue
                        combined_first = PruningCluster.create(
                            f"{child.label}.{first.label}",
                            (*child.members, *first.members),
                        )
                        combined_second = PruningCluster.create(
                            f"{child.label}.{second.label}",
                            (*child.members, *second.members),
                        )
                        bridge_results: list[ComparisonResult] = []
                        for combined, other in (
                            (combined_first, second),
                            (combined_second, first),
                        ):
                            if len(other.members) == 1:
                                raise AssertionError(
                                    "a singleton cannot be the split side "
                                    "of a bridge comparison"
                                )
                            bridge_key = _comparison_key(combined, other)
                            result = cache.get(bridge_key)
                            if result is None:
                                result = compare(
                                    combined,
                                    other,
                                    pair_kind="bridge",
                                    alpha=adjusted_alpha,
                                )
                                result = _record_comparison(
                                    result,
                                    combined,
                                    other,
                                    "bridge",
                                    trace,
                                    parent_label=parent.label,
                                    level=parent_level + 1,
                                    root_distance=(
                                        None
                                        if distance is None
                                        else distance(combined, other)
                                    ),
                                )
                                comparisons.append(result)
                                cache[bridge_key] = result
                            bridge_results.append(result)
                        first_result, second_result = bridge_results
                        if (
                            first_result.decision is Decision.MERGE
                            and second_result.decision is Decision.SPLIT
                        ):
                            decisions[_comparison_key(child, first)] = Decision.SPLIT
                        elif (
                            first_result.decision is Decision.SPLIT
                            and second_result.decision is Decision.MERGE
                        ):
                            decisions[_comparison_key(child, second)] = Decision.SPLIT
                        elif (
                            first_result.decision is Decision.SPLIT
                            and second_result.decision is Decision.SPLIT
                        ):
                            if distance is not None:
                                first_score = distance(child, first)
                                second_score = distance(child, second)
                            else:
                                first_score = (
                                    first_result.metrics.mean_observed_accuracy
                                    if first_result.metrics is not None
                                    else 1.0
                                )
                                second_score = (
                                    second_result.metrics.mean_observed_accuracy
                                    if second_result.metrics is not None
                                    else 1.0
                                )
                            rejected = first if first_score > second_score else second
                            decisions[_comparison_key(child, rejected)] = Decision.SPLIT
                        if trace is not None:
                            trace.append(
                                "bridge",
                                "pruning",
                                parameters={
                                    "child": child.label,
                                    "partner1": first.label,
                                    "partner2": second.label,
                                },
                                input_membership_hashes=(
                                    child.membership_sha256,
                                    first.membership_sha256,
                                    second.membership_sha256,
                                ),
                                topology_context={
                                    "parent": parent.label,
                                    "level": parent_level + 1,
                                },
                                decision="resolved",
                                reason=(
                                    f"{first_result.decision.value}_"
                                    f"{second_result.decision.value}"
                                ),
                            )

            components = _connected_components(clusters, decisions)
            for component in components:
                if len(component) == 1:
                    continue
                members = sorted(
                    value for cluster in component for value in cluster.members
                )
                if len(members) == len(parent_members):
                    new_label = parent.label
                else:
                    new_label = _new_merge_label(
                        component,
                        level=parent_level + 1,
                        counters=counters,
                    )
                old_labels = [cluster.label for cluster in component]
                current[np.asarray(members, dtype=np.int64)] = new_label
                if trace is not None:
                    trace.append(
                        "merge",
                        "pruning",
                        parameters={
                            "clusters": old_labels,
                            "new_label": new_label,
                            "pair_kind": "connected_component",
                        },
                        input_membership_hashes=tuple(
                            cluster.membership_sha256 for cluster in component
                        ),
                        output_membership_hashes=(membership_sha256(members),),
                        topology_context={
                            "parent": parent.label,
                            "level": parent_level + 1,
                        },
                        decision="merge",
                    )

        split_count = sum(
            result.decision is Decision.SPLIT for result in comparisons
        )
        score_count = sum(result.metrics is not None for result in comparisons)
        top_reached = parent_level <= 1 and split_count >= 1
        development_threshold = (
            score_count >= (100 if method is Multiplicity.BH else 50)
            and split_count >= (5 if method is Multiplicity.BH else 1)
        )
        if method is not Multiplicity.NONE and (top_reached or development_threshold):
            previous = adjusted_alpha
            adjusted_alpha = _dynamic_threshold(
                comparisons,
                multiplicity=method,
                alpha=alpha,
                previous=adjusted_alpha,
                rules=rule_set,
            )
            if trace is not None:
                trace.append(
                    "multiplicity",
                    "pruning",
                    parameters={
                        "method": method.value,
                        "previous_alpha": previous,
                        "adjusted_alpha": adjusted_alpha,
                        "score_count": score_count,
                        "split_count": split_count,
                        "rules": rule_set.value,
                    },
                    topology_context={"level": parent_level + 1},
                    decision="adjust",
                    reason="top_reached" if top_reached else "minimum_counts_reached",
                    bounded_values={
                        "previous_alpha": previous,
                        "adjusted_alpha": adjusted_alpha,
                    },
                )

            # R resolves overlapping failed split corrections by selecting the
            # closest incident pair (or the lowest mean accuracy when distance
            # awareness is disabled). Rebuild membership state after every
            # merge so stale comparison-cache identities cannot be reused.
            corrections = [
                _MultiplicityCorrection(result, hashes[0], hashes[1], order)
                for order, result in enumerate(comparisons)
                if (hashes := result.input_membership_sha256) is not None
                and result.decision is Decision.SPLIT
                and not _split_survives(
                    result,
                    adjusted_alpha,
                    use_variance=use_variance,
                )
            ]
            while corrections:
                current_clusters = {
                    cluster.membership_sha256: cluster
                    for cluster in _clusters_from_assignment(current)
                }
                corrections = [
                    correction
                    for correction in corrections
                    if correction.left_hash in current_clusters
                    and correction.right_hash in current_clusters
                ]
                if not corrections:
                    break
                correction = _select_multiplicity_correction(
                    corrections, current_clusters, distance
                )
                left_hash = correction.left_hash
                right_hash = correction.right_hash
                left = current_clusters[left_hash]
                right = current_clusters[right_hash]
                merged_members = sorted((*left.members, *right.members))
                new_label = _new_merge_label(
                    [left, right], level=parent_level + 1, counters=counters
                )
                current[np.asarray(merged_members, dtype=np.int64)] = new_label
                if trace is not None:
                    trace.append(
                        "merge",
                        "pruning",
                        parameters={
                            "clusters": [left.label, right.label],
                            "new_label": new_label,
                            "pair_kind": "multiplicity_adjustment",
                        },
                        input_membership_hashes=(left_hash, right_hash),
                        output_membership_hashes=(membership_sha256(merged_members),),
                        topology_context={"level": parent_level + 1},
                        decision="merge",
                        reason=ReasonCode.MERGE_ADJUSTMENT.value,
                    )
                corrections = [
                    candidate
                    for candidate in corrections
                    if candidate is not correction
                ]
        stepwise.append(tuple(str(value) for value in current))
        if trace is not None:
            trace.append(
                "prune",
                "pruning",
                parameters={
                    "level": parent_level + 1,
                    "cluster_count": len(set(map(str, current.tolist()))),
                },
                output_membership_hashes=tuple(
                    cluster.membership_sha256
                    for cluster in _clusters_from_assignment(current)
                ),
                topology_context={"level": parent_level + 1},
                decision="level_complete",
            )

    final_clusters = _clusters_from_assignment(current)
    if rule_set is RuleSet.CHOIR_V0_3_0:
        final_clusters.sort(key=lambda cluster: (-len(cluster.members), cluster.label))
        final_ordering = "size_desc_then_record_label"
    else:
        final_clusters.sort(
            key=lambda cluster: (-len(cluster.members), cluster.membership_sha256)
        )
        final_ordering = "size_desc_then_membership_sha256"
    numeric = np.empty(tree.n_cells, dtype=np.int64)
    for final_label, cluster in enumerate(final_clusters, start=1):
        numeric[np.asarray(cluster.members, dtype=np.int64)] = final_label
    if trace is not None:
        trace.append(
            "final_relabel",
            "finalization",
            parameters={
                "ordering": final_ordering,
                "cluster_count": len(final_clusters),
                "labels": list(range(1, len(final_clusters) + 1)),
            },
            input_membership_hashes=tuple(
                cluster.membership_sha256 for cluster in final_clusters
            ),
            decision="relabel",
        )
    return PruningResult(
        labels=numeric,
        record_labels=tuple(str(value) for value in current.tolist()),
        comparisons=tuple(comparisons),
        adjusted_alpha=adjusted_alpha,
        stepwise_labels=tuple(stepwise),
    )
