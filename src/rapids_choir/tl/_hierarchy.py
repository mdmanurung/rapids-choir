"""Pure hierarchy construction state machine recovered from CHOIR v0.3.0."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .._enums import Decision, Multiplicity, RuleSet
from .._hashing import membership_sha256
from .._semantic_trace import TraceRecorder
from ..results import ComparisonResult
from ..tree import HierarchyTree, hierarchy_from_partitions
from ._pruning import CompareClusters, PruningCluster, _record_comparison


class ClusteringFailure(RuntimeError):
    """A partition backend could not cluster at the requested resolution."""


class Partitioner(Protocol):
    def __call__(
        self,
        resolution: float,
        members: np.ndarray[Any, np.dtype[np.int64]],
        seed: int,
    ) -> Any: ...


Silhouette = Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float]


@dataclass(frozen=True, slots=True)
class ResolutionStep:
    resolution: float
    cluster_count: int | None
    action: str
    gap: float | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class StartingResolution:
    resolution: float
    resolution_zero_partition: np.ndarray[Any, np.dtype[np.int64]]
    decimal_places: int
    steps: tuple[ResolutionStep, ...]


@dataclass(frozen=True, slots=True)
class HierarchyBuild:
    tree: HierarchyTree
    steps: tuple[ResolutionStep, ...]
    stop_reason: str
    accepted_resolutions: tuple[float, ...]
    comparisons: tuple[ComparisonResult, ...] = ()


@dataclass(frozen=True, slots=True)
class SubtreeBuild:
    tree: HierarchyTree
    steps: tuple[ResolutionStep, ...]
    stop_reasons: tuple[str, ...]
    accepted_resolutions: tuple[float, ...]
    comparisons: tuple[ComparisonResult, ...]


def _exact_linkage_cut(
    linkage: Any, n_leaves: int, n_groups: int
) -> np.ndarray[Any, Any]:
    """Cut a SciPy linkage after exactly ``n_leaves - n_groups`` merges."""

    if not 1 <= n_groups <= n_leaves:
        raise ValueError("complete-linkage cut count is outside the leaf range")
    active: dict[int, set[int]] = {index: {index} for index in range(n_leaves)}
    for merge_index, row in enumerate(np.asarray(linkage, dtype=np.float64)):
        if len(active) <= n_groups:
            break
        left_id = int(row[0])
        right_id = int(row[1])
        left = active.pop(left_id)
        right = active.pop(right_id)
        active[n_leaves + merge_index] = left | right
    ordered = sorted(active.values(), key=lambda members: min(members))
    labels = np.empty(n_leaves, dtype=np.int64)
    for label, members in enumerate(ordered):
        labels[np.fromiter(sorted(members), dtype=np.int64)] = label
    return labels


def _complete_linkage_partitions(
    representation: np.ndarray[Any, Any],
    child_codes: np.ndarray[Any, Any],
    child_order: list[int],
    *,
    rules: RuleSet,
) -> dict[int, np.ndarray[Any, Any]]:
    """Return exact-k complete-linkage cuts used by R ``.optimizeTree``."""

    from scipy.cluster.hierarchy import linkage  # type: ignore[import-untyped]
    from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

    if rules is RuleSet.CHOIR_V0_3_0:
        # CHOIR v0.3.0 computes ``reduction_parent`` but accidentally passes the
        # full reduction. R then recycles the parent-length logical selector.
        recycled = np.resize(child_codes, representation.shape[0])
        centroids = np.vstack(
            [representation[recycled == child].mean(axis=0) for child in child_order]
        )
    else:
        centroids = np.vstack(
            [representation[child_codes == child].mean(axis=0) for child in child_order]
        )
    if not np.isfinite(centroids).all():
        raise ClusteringFailure("tree optimization produced a non-finite centroid")
    hierarchy = linkage(pdist(centroids), method="complete")
    return {
        count: _exact_linkage_cut(hierarchy, len(child_order), count)
        for count in range(2, len(child_order) + 1)
    }


def optimize_hierarchy(
    tree: HierarchyTree,
    representation: Any,
    *,
    rules: RuleSet | str,
    trace: TraceRecorder | None = None,
) -> HierarchyTree:
    """Insert R-compatible complete-linkage levels for splits with >3 children."""

    coordinates = np.asarray(representation, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != tree.n_cells:
        raise ValueError("hierarchy optimization representation is misaligned")
    rule_set = RuleSet(rules)
    if tree.n_levels < 2:
        return tree
    partitions: list[np.ndarray[Any, Any]] = []
    resolutions: list[float | None] = []
    inserted = 0
    for level_index in range(1, tree.n_levels):
        previous = np.asarray(tree.assignments[level_index - 1], dtype=object)
        current = np.asarray(tree.assignments[level_index], dtype=object)
        parent_order = list(dict.fromkeys(int(value) for value in previous.tolist()))
        parent_details: list[
            tuple[np.ndarray[Any, Any], list[int], dict[int, np.ndarray[Any, Any]]]
        ] = []
        maximum_children = 0
        for parent_id in parent_order:
            members = np.flatnonzero(previous == parent_id)
            local = current[members]
            children = list(dict.fromkeys(int(value) for value in local.tolist()))
            cuts: dict[int, np.ndarray[Any, Any]] = {}
            if len(children) > 3:
                maximum_children = max(maximum_children, len(children))
                child_codes = np.asarray(
                    [children.index(int(value)) for value in local.tolist()],
                    dtype=np.int64,
                )
                child_order = list(range(len(children)))
                source = (
                    coordinates
                    if rule_set is RuleSet.CHOIR_V0_3_0
                    else coordinates[members]
                )
                cuts = _complete_linkage_partitions(
                    source,
                    child_codes,
                    child_order,
                    rules=rule_set,
                )
            parent_details.append((members, children, cuts))

        if maximum_children > 3:
            for group_count in range(2, maximum_children + 1):
                optimized = previous.copy()
                for parent_ordinal, (members, children, cuts) in enumerate(
                    parent_details, start=1
                ):
                    if not cuts:
                        continue
                    local_count = min(group_count, len(children))
                    local_cut = cuts[local_count]
                    child_for_cell = current[members]
                    child_positions = np.asarray(
                        [children.index(int(value)) for value in child_for_cell],
                        dtype=np.int64,
                    )
                    optimized[members] = np.asarray(
                        [
                            f"opt-{level_index}-{group_count}-{parent_ordinal}-"
                            f"{local_cut[position]}"
                            for position in child_positions
                        ],
                        dtype=object,
                    )
                partitions.append(optimized)
                resolutions.append(None)
                inserted += 1
        partitions.append(current)
        resolutions.append(tree.resolutions[level_index])

    optimized_tree = hierarchy_from_partitions(
        partitions,
        cell_ids=tree.cell_ids,
        resolutions=resolutions,
    )
    if trace is not None and inserted:
        trace.append(
            "resolution_transition",
            "hierarchy",
            parameters={
                "inserted_levels": inserted,
                "rules": rule_set.value,
            },
            input_membership_hashes=tuple(
                node.membership_sha256 for node in tree.leaves
            ),
            output_membership_hashes=tuple(
                node.membership_sha256 for node in optimized_tree.leaves
            ),
            decision="optimize",
            reason="complete_linkage_for_wide_split",
        )
    return optimized_tree


def _emit_hierarchy_trace(
    tree: HierarchyTree,
    *,
    random_state: int,
    stop_reason: str,
    accepted_resolutions: list[float] | tuple[float, ...],
    trace: TraceRecorder | None,
) -> None:
    if trace is None:
        return
    for node in tree.nodes:
        trace.append(
            "hierarchy_node",
            "hierarchy",
            parameters={
                "node_id": str(node.node_id),
                "label": node.label,
                "level": node.level,
                "ordinal": node.ordinal,
                "size": node.size,
            },
            seed=random_state,
            input_membership_hashes=(
                ()
                if node.parent_id is None
                else (tree.node_map[node.parent_id].membership_sha256,)
            ),
            output_membership_hashes=(node.membership_sha256,),
            topology_context={
                "parent_id": (
                    None if node.parent_id is None else str(node.parent_id)
                ),
                "child_ids": [str(value) for value in node.child_ids],
            },
            decision="create",
        )
    trace.append(
        "hierarchy_stop",
        "hierarchy",
        parameters={"accepted_resolutions": list(accepted_resolutions)},
        seed=random_state,
        output_membership_hashes=tuple(
            node.membership_sha256 for node in tree.leaves
        ),
        reason=stop_reason,
    )


def _canonical_labels(values: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError("partition backend must return a one-dimensional vector")
    mapping: dict[Any, int] = {}
    result = np.empty(labels.size, dtype=np.int64)
    for index, raw_value in enumerate(labels.tolist()):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if value not in mapping:
            mapping[value] = len(mapping)
        result[index] = mapping[value]
    return result


def relabel_singletons(values: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
    """Give every literal ``singleton`` label its own deterministic identity."""

    labels = np.asarray(values, dtype=object)
    if labels.ndim != 1:
        raise ValueError("singleton relabelling requires a one-dimensional vector")
    expanded = [
        f"singleton{index + 1}" if str(value) == "singleton" else value
        for index, value in enumerate(labels.tolist())
    ]
    # R factor levels are lexical; mapping them in sorted string order captures
    # the compatibility ordering without making the raw backend labels public.
    levels = {
        value: index
        for index, value in enumerate(sorted(set(map(str, expanded))))
    }
    return np.asarray([levels[str(value)] for value in expanded], dtype=np.int64)


def _partition(
    partitioner: Partitioner,
    resolution: float,
    members: np.ndarray[Any, np.dtype[np.int64]],
    seed: int,
    *,
    group_singletons: bool,
) -> np.ndarray[Any, np.dtype[np.int64]]:
    values = partitioner(resolution, members, seed)
    labels = (
        _canonical_labels(values)
        if group_singletons
        else relabel_singletons(values)
    )
    if labels.size != members.size:
        raise ValueError("partition result is not aligned to requested members")
    return labels


def _ascending_step(resolution: float) -> float:
    if resolution < 4:
        return 0.5
    if resolution < 10:
        return 1.0
    return 5.0


def _trace_resolution_transition(
    step: ResolutionStep,
    members: np.ndarray[Any, Any],
    *,
    random_state: int,
    trace: TraceRecorder | None,
) -> None:
    if trace is None:
        return
    trace.append(
        "resolution_transition",
        "hierarchy",
        parameters={
            "resolution": step.resolution,
            "cluster_count": step.cluster_count,
            "gap": step.gap,
        },
        seed=random_state,
        input_membership_hashes=(membership_sha256(members),),
        decision=step.action,
        reason=step.reason,
    )


def find_starting_resolution(
    partitioner: Partitioner,
    members: Any,
    *,
    random_state: int,
    group_singletons: bool = True,
    max_steps: int = 256,
    trace: TraceRecorder | None = None,
) -> StartingResolution:
    """Reproduce ``.getStartingResolution`` resolution transitions."""

    selected = np.asarray(members, dtype=np.int64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("starting-resolution members must be a non-empty vector")
    steps: list[ResolutionStep] = []
    res0 = _partition(
        partitioner,
        0.0,
        selected,
        random_state,
        group_singletons=group_singletons,
    )
    count0 = int(np.unique(res0).size)
    steps.append(ResolutionStep(0.0, count0, "baseline"))
    try:
        res1 = _partition(
            partitioner,
            1.0,
            selected,
            random_state,
            group_singletons=group_singletons,
        )
    except ClusteringFailure:
        steps.append(
            ResolutionStep(1.0, None, "failure", reason="resolution_1_failed")
        )
        result = StartingResolution(0.1, res0, 2, tuple(steps))
        if trace is not None:
            for step in steps:
                trace.append(
                    "resolution_transition",
                    "hierarchy",
                    parameters={
                        "resolution": step.resolution,
                        "cluster_count": step.cluster_count,
                        "gap": step.gap,
                    },
                    seed=random_state,
                    input_membership_hashes=(membership_sha256(selected),),
                    decision=step.action,
                    reason=step.reason,
                )
            trace.append(
                "resolution_search",
                "hierarchy",
                parameters={"starting_resolution": 0.1, "decimal_places": 2},
                seed=random_state,
                input_membership_hashes=(membership_sha256(selected),),
                reason="resolution_1_failed",
            )
        return result
    count1 = int(np.unique(res1).size)
    steps.append(ResolutionStep(1.0, count1, "probe"))
    decimal_places = 1

    if count0 == count1:
        resolution = 1.5
        current_count = count1
        for _ in range(max_steps):
            if current_count > count1:
                break
            try:
                current = _partition(
                    partitioner,
                    resolution,
                    selected,
                    random_state,
                    group_singletons=group_singletons,
                )
            except ClusteringFailure:
                steps.append(
                    ResolutionStep(
                        resolution,
                        None,
                        "failure_backtrack",
                        reason="clustering_failed",
                    )
                )
                resolution -= _ascending_step(resolution)
                break
            current_count = int(np.unique(current).size)
            steps.append(ResolutionStep(resolution, current_count, "probe_up"))
            if current_count <= count1:
                resolution += _ascending_step(resolution)
        else:
            raise ClusteringFailure("starting-resolution upward search exceeded limit")
        starting = resolution
    elif count1 > count0:
        resolution = 0.7
        current_count = count1
        for _ in range(max_steps):
            if current_count <= count0:
                break
            current = _partition(
                partitioner,
                resolution,
                selected,
                random_state,
                group_singletons=group_singletons,
            )
            current_count = int(np.unique(current).size)
            steps.append(ResolutionStep(resolution, current_count, "probe_down"))
            if current_count > count0:
                if resolution > 0.1:
                    resolution -= 0.3
                else:
                    resolution /= 10
                    decimal_places += 1
            resolution = round(resolution, decimal_places)
        else:
            raise ClusteringFailure(
                "starting-resolution downward search exceeded limit"
            )
        if resolution >= 0.1:
            starting = resolution + 0.3
        else:
            starting = resolution * 10
            decimal_places -= 1
    else:
        raise ClusteringFailure(
            "resolution 1 produced fewer clusters than resolution 0"
        )

    result = StartingResolution(
        round(float(starting), decimal_places),
        res0,
        decimal_places + 1,
        tuple(steps),
    )
    if trace is not None:
        for step in steps:
            trace.append(
                "resolution_transition",
                "hierarchy",
                parameters={
                    "resolution": step.resolution,
                    "cluster_count": step.cluster_count,
                    "gap": step.gap,
                },
                seed=random_state,
                input_membership_hashes=(membership_sha256(selected),),
                decision=step.action,
                reason=step.reason,
            )
        trace.append(
            "resolution_search",
            "hierarchy",
            parameters={
                "starting_resolution": result.resolution,
                "decimal_places": result.decimal_places,
            },
            seed=random_state,
            input_membership_hashes=(membership_sha256(selected),),
            decision="selected",
        )
    return result


def _append_subtree_refinement(
    levels: list[np.ndarray[Any, Any]],
    current_local: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    serial: int,
) -> None:
    refined = levels[-1].copy()
    mapping: dict[int, str] = {}
    for value in labels.tolist():
        integer = int(value)
        mapping.setdefault(integer, f"subtree-{serial}-{len(mapping) + 1}")
    refined[current_local] = np.asarray(
        [mapping[int(value)] for value in labels.tolist()], dtype=object
    )
    levels.append(refined)


def _subtree_clusters(
    values: np.ndarray[Any, Any],
    current_local: np.ndarray[Any, Any],
    global_members: np.ndarray[Any, Any],
    *,
    tree_id: str,
    level: int,
) -> tuple[list[PruningCluster], dict[str, np.ndarray[Any, Any]]]:
    order = list(dict.fromkeys(str(value) for value in values.tolist()))
    clusters: list[PruningCluster] = []
    local_by_label: dict[str, np.ndarray[Any, Any]] = {}
    for ordinal, value in enumerate(order, start=1):
        local = current_local[np.flatnonzero(np.asarray(values, dtype=str) == value)]
        label = f"{tree_id}_L{level}_{ordinal}"
        clusters.append(PruningCluster.create(label, global_members[local]))
        local_by_label[label] = local
    return clusters, local_by_label


def _centroid_distance(
    left: PruningCluster,
    right: PruningCluster,
    representation: np.ndarray[Any, Any],
) -> float:
    left_center = representation[np.asarray(left.members, dtype=np.int64)].mean(axis=0)
    right_center = representation[np.asarray(right.members, dtype=np.int64)].mean(
        axis=0
    )
    return float(np.linalg.norm(left_center - right_center))


def _nearest_cluster_pairs(
    clusters: list[PruningCluster],
    representation: np.ndarray[Any, Any],
) -> list[tuple[PruningCluster, PruningCluster, float]]:
    if len(clusters) < 2:
        return []
    distances = np.zeros((len(clusters), len(clusters)), dtype=np.float64)
    for left_index, left in enumerate(clusters[:-1]):
        for right_index in range(left_index + 1, len(clusters)):
            value = _centroid_distance(left, clusters[right_index], representation)
            distances[left_index, right_index] = value
            distances[right_index, left_index] = value
    pairs: list[tuple[PruningCluster, PruningCluster, float]] = []
    seen: set[tuple[str, str]] = set()
    for left_index, left in enumerate(clusters):
        positive = np.flatnonzero(distances[left_index] > 0)
        candidates = positive if positive.size else np.flatnonzero(
            np.arange(len(clusters)) != left_index
        )
        if candidates.size == 0:
            continue
        minimum = float(distances[left_index, candidates].min())
        right_index = int(
            candidates[np.flatnonzero(distances[left_index, candidates] == minimum)[0]]
        )
        right = clusters[right_index]
        first, second = sorted(
            (left.membership_sha256, right.membership_sha256)
        )
        key = (first, second)
        if key not in seen:
            seen.add(key)
            pairs.append((left, right, minimum))
    return pairs


def build_overclustered_subtree(
    partitioner: Partitioner,
    *,
    members: Any,
    cell_ids: Any,
    representation: Any,
    compare: CompareClusters,
    min_cluster_depth: int,
    alpha: float,
    random_state: int,
    group_singletons: bool = True,
    rules: RuleSet | str = RuleSet.CORRECTED,
    max_steps: int = 512,
    tree_id: str = "P1_1",
    trace: TraceRecorder | None = None,
) -> SubtreeBuild:
    """Reproduce the recursive comparator-stopped ``.getTree.subtree`` search."""

    selected = np.asarray(members, dtype=np.int64)
    cells = tuple(str(value) for value in cell_ids)
    coordinates = np.asarray(representation, dtype=np.float64)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("subtree members must be a non-empty vector")
    if len(cells) != selected.size:
        raise ValueError("subtree cell IDs must align to members")
    if coordinates.ndim != 2 or selected.max(initial=-1) >= coordinates.shape[0]:
        raise ValueError("subtree representation is misaligned")
    if min_cluster_depth < 1:
        raise ValueError("min_cluster_depth must be positive")
    levels: list[np.ndarray[Any, Any]] = [
        np.zeros(selected.size, dtype=np.int64).astype(object)
    ]
    resolutions: list[float | None] = [0.0]
    unresolved: set[int] = set(range(selected.size))
    steps: list[ResolutionStep] = []
    comparisons: list[ComparisonResult] = []
    stop_reasons: list[str] = []
    serial = 0
    operation_count = 0

    def resolve_neighbor_pairs(
        values: np.ndarray[Any, Any],
        current_local: np.ndarray[Any, Any],
        *,
        level: int,
        skip: tuple[str, str] | None = None,
    ) -> None:
        clusters, local_by_label = _subtree_clusters(
            values,
            current_local,
            selected,
            tree_id=tree_id,
            level=level,
        )
        small = {
            cluster.label for cluster in clusters if len(cluster.members) < 4
        }
        for label in small:
            unresolved.difference_update(local_by_label[label].tolist())
        pairs = _nearest_cluster_pairs(clusters, coordinates)
        if skip is not None:
            pairs = [
                pair
                for pair in pairs
                if {pair[0].membership_sha256, pair[1].membership_sha256}
                != set(skip)
            ]
        while pairs:
            left, right, pair_distance = pairs.pop(0)
            result = compare(
                left,
                right,
                pair_kind="subtree_neighbor",
                alpha=alpha,
            )
            recorded = _record_comparison(
                result,
                left,
                right,
                "subtree_neighbor",
                trace,
                parent_label=tree_id,
                level=level,
                root_distance=pair_distance,
                subtree_distance=pair_distance,
            )
            comparisons.append(recorded)
            if result.decision is Decision.MERGE:
                unresolved.difference_update(local_by_label[left.label].tolist())
                unresolved.difference_update(local_by_label[right.label].tolist())
                # Compatibility behavior removes later pairs whose first cluster
                # is the second member of the accepted merge.
                pairs = [pair for pair in pairs if pair[0].label != right.label]

    while unresolved:
        operation_count += 1
        if operation_count > max_steps:
            raise ClusteringFailure("subtree outer search exceeded limit")
        latest = levels[-1]
        first_unresolved = min(unresolved)
        current_label = latest[first_unresolved]
        current_local = np.flatnonzero(latest == current_label).astype(
            np.int64, copy=False
        )
        current_global = selected[current_local]

        def record_step(
            step: ResolutionStep,
            step_members: np.ndarray[Any, Any] = current_global,
        ) -> None:
            steps.append(step)
            _trace_resolution_transition(
                step,
                step_members,
                random_state=random_state,
                trace=trace,
            )

        starting = find_starting_resolution(
            partitioner,
            current_global,
            random_state=random_state,
            group_singletons=group_singletons,
            max_steps=max_steps,
            trace=trace,
        )
        steps.extend(starting.steps)
        resolution = starting.resolution
        old_resolution = 0.0
        gap = starting.resolution / 2
        gap_counter = 0
        reset_counter = 0
        level_counter = 0
        count = int(np.unique(starting.resolution_zero_partition).size)
        stopped = False
        if count > 1:
            serial += 1
            _append_subtree_refinement(
                levels,
                current_local,
                starting.resolution_zero_partition,
                serial=serial,
            )
            resolutions.append(0.0)
            _, sizes = np.unique(
                starting.resolution_zero_partition, return_counts=True
            )
            if int(sizes.min()) <= min_cluster_depth:
                stopped = True
                stop_reasons.append("minimum_cluster_depth_reached")

        while not stopped:
            operation_count += 1
            if operation_count > max_steps:
                raise ClusteringFailure("subtree resolution search exceeded limit")
            try:
                new_partition = _partition(
                    partitioner,
                    resolution,
                    current_global,
                    random_state,
                    group_singletons=group_singletons,
                )
            except ClusteringFailure:
                reason = "clustering_failed"
                record_step(
                    ResolutionStep(resolution, None, "stop", gap=gap, reason=reason)
                )
                stop_reasons.append(reason)
                unresolved.difference_update(current_local.tolist())
                break
            new_count = int(np.unique(new_partition).size)
            _, sizes = np.unique(new_partition, return_counts=True)
            if int(sizes.max()) <= 2:
                reason = "cannot_subdivide_further"
                record_step(
                    ResolutionStep(
                        resolution, new_count, "stop", gap=gap, reason=reason
                    )
                )
                stop_reasons.append(reason)
                unresolved.difference_update(current_local.tolist())
                break
            if new_count < count:
                reason = "cluster_count_decreased"
                record_step(
                    ResolutionStep(
                        resolution, new_count, "stop", gap=gap, reason=reason
                    )
                )
                stop_reasons.append(reason)
                previous_values = levels[-1][current_local]
                previous_clusters, _ = _subtree_clusters(
                    previous_values,
                    current_local,
                    selected,
                    tree_id=tree_id,
                    level=len(levels),
                )
                if sum(len(cluster.members) > 1 for cluster in previous_clusters) > 1:
                    resolve_neighbor_pairs(
                        previous_values, current_local, level=len(levels)
                    )
                else:
                    _, local_by_label = _subtree_clusters(
                        previous_values,
                        current_local,
                        selected,
                        tree_id=tree_id,
                        level=len(levels),
                    )
                    for cluster in previous_clusters:
                        if len(cluster.members) < 4:
                            unresolved.difference_update(
                                local_by_label[cluster.label].tolist()
                            )
                break
            if (
                new_count > count * 1.25
                and new_count - count > 3
                and reset_counter < 3
            ):
                resolution -= gap
                gap /= 2
                gap_counter = 0
                reset_counter += 1
                record_step(
                    ResolutionStep(
                        resolution,
                        new_count,
                        "halve_gap",
                        gap=gap,
                        reason="large_jump",
                    )
                )
            elif new_count > count:
                serial += 1
                _append_subtree_refinement(
                    levels, current_local, new_partition, serial=serial
                )
                resolutions.append(resolution)
                gap = resolution - old_resolution
                gap_counter = 0
                reset_counter = 0
                old_resolution = resolution
                count = new_count
                record_step(
                    ResolutionStep(resolution, new_count, "accept", gap=gap)
                )
                clusters, local_by_label = _subtree_clusters(
                    new_partition,
                    current_local,
                    selected,
                    tree_id=tree_id,
                    level=len(levels),
                )
                non_singletons = [
                    cluster for cluster in clusters if len(cluster.members) > 1
                ]
                if len(non_singletons) > 1:
                    nearest = _nearest_cluster_pairs(clusters, coordinates)
                    if not nearest:
                        unresolved.difference_update(current_local.tolist())
                        stop_reasons.append("no_positive_neighbor_distance")
                        break
                    furthest = max(
                        enumerate(nearest), key=lambda item: (item[1][2], -item[0])
                    )[1]
                    left, right, pair_distance = furthest
                    result = compare(
                        left,
                        right,
                        pair_kind="subtree_furthest",
                        alpha=alpha,
                    )
                    recorded = _record_comparison(
                        result,
                        left,
                        right,
                        "subtree_furthest",
                        trace,
                        parent_label=tree_id,
                        level=len(levels),
                        root_distance=pair_distance,
                        subtree_distance=pair_distance,
                    )
                    comparisons.append(recorded)
                    if result.decision is Decision.MERGE:
                        unresolved.difference_update(current_local.tolist())
                        stop_reasons.append("furthest_neighbors_merge")
                        break
                    level_counter += 1
                    if level_counter >= 5 and current_local.size > min_cluster_depth:
                        stop_reasons.append("all_neighbors_after_five_levels")
                        resolve_neighbor_pairs(
                            new_partition,
                            current_local,
                            level=len(levels),
                            skip=(left.membership_sha256, right.membership_sha256),
                        )
                        break
                else:
                    for cluster in clusters:
                        if len(cluster.members) < 4:
                            unresolved.difference_update(
                                local_by_label[cluster.label].tolist()
                            )
                    stop_reasons.append("all_but_one_cluster_are_singletons")
                    break
            resolution += gap
            gap_counter += 1
            if gap_counter >= 5:
                gap *= 2
                gap_counter = 0
            resolution = round(resolution, starting.decimal_places)

    tree = hierarchy_from_partitions(
        levels,
        cell_ids=cells,
        resolutions=resolutions,
        root=tree_id,
    )
    tree = optimize_hierarchy(
        tree,
        coordinates[selected],
        rules=rules,
        trace=trace,
    )
    return SubtreeBuild(
        tree=tree,
        steps=tuple(steps),
        stop_reasons=tuple(stop_reasons),
        accepted_resolutions=tuple(
            float(value) for value in resolutions if value is not None
        ),
        comparisons=tuple(comparisons),
    )


def build_resolution_hierarchy(
    partitioner: Partitioner,
    *,
    cell_ids: Any,
    members: Any | None = None,
    random_state: int,
    silhouette: Silhouette | None = None,
    max_clusters: int | None = None,
    group_singletons: bool = True,
    max_steps: int = 512,
    representation: Any | None = None,
    rules: RuleSet | str = RuleSet.CORRECTED,
    trace: TraceRecorder | None = None,
    emit_trace: bool = True,
) -> HierarchyBuild:
    """Construct the root SNN-Louvain hierarchy with R-compatible transitions."""

    cells = tuple(str(value) for value in cell_ids)
    selected_members = (
        np.arange(len(cells), dtype=np.int64)
        if members is None
        else np.asarray(members, dtype=np.int64)
    )
    if selected_members.ndim != 1 or selected_members.size != len(cells):
        raise ValueError("hierarchy members must align to cell_ids")
    if len(set(selected_members.tolist())) != selected_members.size:
        raise ValueError("hierarchy members must be unique")
    starting = find_starting_resolution(
        partitioner,
        selected_members,
        random_state=random_state,
        group_singletons=group_singletons,
        max_steps=max_steps,
        trace=trace,
    )
    partitions: list[np.ndarray[Any, np.dtype[np.int64]]] = [
        starting.resolution_zero_partition
    ]
    resolutions: list[float] = [0.0]
    steps = list(starting.steps)
    old_resolution = 0.0
    resolution = starting.resolution
    gap = starting.resolution / 2
    count = int(np.unique(partitions[0]).size)
    if count > 1 and silhouette is not None:
        current_silhouette = float(silhouette(partitions[0], selected_members))
        maximum_silhouette = current_silhouette
    else:
        current_silhouette = -1.0
        maximum_silhouette = -2.0
    maximum_silhouette_index = 0
    stop_counter = 0
    gap_counter = 0
    reset_counter = 0
    stop_reason = "further_subclustering_impossible"

    for _ in range(max_steps):
        try:
            new_partition = _partition(
                partitioner,
                resolution,
                selected_members,
                random_state,
                group_singletons=group_singletons,
            )
        except ClusteringFailure:
            stop_reason = "clustering_failed"
            steps.append(
                ResolutionStep(
                    resolution, None, "stop", gap=gap, reason=stop_reason
                )
            )
            break
        new_count = int(np.unique(new_partition).size)
        _, sizes = np.unique(new_partition, return_counts=True)
        if int(sizes.max()) <= 2:
            stop_reason = "cannot_subdivide_further"
            steps.append(
                ResolutionStep(
                    resolution, new_count, "stop", gap=gap, reason=stop_reason
                )
            )
            break
        if new_count < count:
            stop_reason = "cluster_count_decreased"
            steps.append(
                ResolutionStep(
                    resolution, new_count, "stop", gap=gap, reason=stop_reason
                )
            )
            break
        if new_count > count * 1.25 and new_count - count > 3 and reset_counter < 3:
            resolution -= gap
            gap /= 2
            gap_counter = 0
            reset_counter += 1
            steps.append(
                ResolutionStep(
                    resolution, new_count, "halve_gap", gap=gap, reason="large_jump"
                )
            )
        elif new_count > count:
            partitions.append(new_partition)
            resolutions.append(resolution)
            gap = resolution - old_resolution
            gap_counter = 0
            reset_counter = 0
            old_resolution = resolution
            count = new_count
            steps.append(ResolutionStep(resolution, new_count, "accept", gap=gap))
            if max_clusters is not None and new_count > max_clusters:
                stop_reason = "maximum_clusters_reached"
                break
            if silhouette is not None:
                current_silhouette = float(
                    silhouette(new_partition, selected_members)
                )
                if current_silhouette >= maximum_silhouette:
                    maximum_silhouette = current_silhouette
                    maximum_silhouette_index = len(partitions) - 1
                    stop_counter = 0
                else:
                    stop_counter += 1
                    if stop_counter >= 5:
                        stop_reason = "maximum_silhouette_identified"
                        break
        resolution += gap
        gap_counter += 1
        if gap_counter >= 5:
            gap *= 2
            gap_counter = 0
        resolution = round(resolution, starting.decimal_places)
    else:
        raise ClusteringFailure("hierarchy resolution search exceeded limit")

    if max_clusters is None and silhouette is not None:
        partitions = partitions[: maximum_silhouette_index + 1]
        resolutions = resolutions[: maximum_silhouette_index + 1]
    for step in steps[len(starting.steps) :]:
        _trace_resolution_transition(
            step,
            selected_members,
            random_state=random_state,
            trace=trace,
        )
    tree = hierarchy_from_partitions(
        partitions,
        cell_ids=cells,
        resolutions=resolutions,
    )
    if representation is not None:
        tree = optimize_hierarchy(
            tree,
            representation,
            rules=rules,
            trace=trace,
        )
    if emit_trace:
        _emit_hierarchy_trace(
            tree,
            random_state=random_state,
            stop_reason=stop_reason,
            accepted_resolutions=resolutions,
            trace=trace,
        )
    return HierarchyBuild(
        tree=tree,
        steps=tuple(steps),
        stop_reason=stop_reason,
        accepted_resolutions=tuple(resolutions),
    )


def _combine_refinement_sequences(
    base: np.ndarray[Any, Any],
    branches: list[
        tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]], str]
    ],
) -> list[np.ndarray[Any, Any]]:
    depth = max((len(sequence) for _, sequence, _ in branches), default=0)
    combined: list[np.ndarray[Any, Any]] = []
    previous = np.asarray(base, dtype=object).copy()
    for level_index in range(depth):
        current = previous.copy()
        for positions, sequence, prefix in branches:
            if not sequence:
                continue
            values = np.asarray(sequence[min(level_index, len(sequence) - 1)])
            mapping: dict[Any, str] = {}
            translated: list[str] = []
            for value in values.tolist():
                key = value.item() if isinstance(value, np.generic) else value
                mapping.setdefault(key, f"{prefix}-{len(mapping) + 1}")
                translated.append(mapping[key])
            current[positions] = np.asarray(translated, dtype=object)
        combined.append(current)
        previous = current
    return combined


def build_automatic_hierarchy(
    partitioner: Partitioner,
    *,
    cell_ids: Any,
    representation: Any,
    silhouette: Silhouette | None,
    compare: CompareClusters,
    max_clusters: int | None,
    min_cluster_depth: int,
    multiplicity: Multiplicity | str,
    alpha: float,
    random_state: int,
    group_singletons: bool = True,
    rules: RuleSet | str = RuleSet.CORRECTED,
    max_steps: int = 512,
    trace: TraceRecorder | None = None,
) -> HierarchyBuild:
    """Build the root tree and all automatic CHOIR subtree refinements."""

    cells = tuple(str(value) for value in cell_ids)
    coordinates = np.asarray(representation, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[0] != len(cells):
        raise ValueError("automatic hierarchy representation is misaligned")
    if max_clusters is not None:
        return build_resolution_hierarchy(
            partitioner,
            cell_ids=cells,
            random_state=random_state,
            silhouette=None,
            max_clusters=max_clusters,
            group_singletons=group_singletons,
            max_steps=max_steps,
            representation=coordinates,
            rules=rules,
            trace=trace,
        )

    root = build_resolution_hierarchy(
        partitioner,
        cell_ids=cells,
        random_state=random_state,
        silhouette=silhouette,
        max_clusters=None,
        group_singletons=group_singletons,
        max_steps=max_steps,
        representation=coordinates,
        rules=rules,
        trace=trace,
        emit_trace=False,
    )
    all_steps = list(root.steps)
    all_comparisons = list(root.comparisons)
    all_resolutions = list(root.accepted_resolutions)
    method = Multiplicity(multiplicity)
    subtree_alpha = (
        alpha / (len(root.tree.leaves) * 4)
        if method is Multiplicity.BONFERRONI
        else alpha
    )
    root_branches: list[
        tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]], str]
    ] = []

    for root_ordinal, root_leaf in enumerate(root.tree.leaves, start=1):
        selected = root.tree.members(root_leaf.node_id)
        if selected.size <= 3:
            root_branches.append((selected, [], f"root-{root_ordinal}"))
            continue
        if selected.size > min_cluster_depth:
            local = build_resolution_hierarchy(
                partitioner,
                cell_ids=[cells[index] for index in selected],
                members=selected,
                random_state=random_state,
                silhouette=silhouette,
                max_clusters=None,
                group_singletons=group_singletons,
                max_steps=max_steps,
                representation=coordinates[selected],
                rules=rules,
                trace=trace,
                emit_trace=False,
            )
            all_steps.extend(local.steps)
            all_resolutions.extend(local.accepted_resolutions)
        else:
            local_tree = hierarchy_from_partitions(
                [np.zeros(selected.size, dtype=np.int64)],
                cell_ids=[cells[index] for index in selected],
                resolutions=[None],
                root=f"P{root_ordinal}",
            )
            local = HierarchyBuild(
                local_tree,
                (),
                "minimum_cluster_depth_reached",
                (),
            )

        local_sequence = [
            np.asarray(level, dtype=object)
            for level in local.tree.assignments[1:]
        ]
        local_base = np.asarray(local.tree.assignments[-1], dtype=object)
        overclustered_branches: list[
            tuple[np.ndarray[Any, Any], list[np.ndarray[Any, Any]], str]
        ] = []
        for leaf_ordinal, local_leaf in enumerate(local.tree.leaves, start=1):
            leaf_local = local.tree.members(local_leaf.node_id)
            leaf_global = selected[leaf_local]
            if leaf_global.size <= 3:
                overclustered_branches.append(
                    (leaf_local, [], f"over-{root_ordinal}-{leaf_ordinal}")
                )
                continue
            overclustered = build_overclustered_subtree(
                partitioner,
                members=leaf_global,
                cell_ids=[cells[index] for index in leaf_global],
                representation=coordinates,
                compare=compare,
                min_cluster_depth=min_cluster_depth,
                alpha=subtree_alpha,
                random_state=random_state,
                group_singletons=group_singletons,
                rules=rules,
                max_steps=max_steps,
                tree_id=f"P{root_ordinal}_{leaf_ordinal}",
                trace=trace,
            )
            all_steps.extend(overclustered.steps)
            all_resolutions.extend(overclustered.accepted_resolutions)
            all_comparisons.extend(overclustered.comparisons)
            overclustered_branches.append(
                (
                    leaf_local,
                    [
                        np.asarray(level, dtype=object)
                        for level in overclustered.tree.assignments[1:]
                    ],
                    f"over-{root_ordinal}-{leaf_ordinal}",
                )
            )
        local_sequence.extend(
            _combine_refinement_sequences(local_base, overclustered_branches)
        )
        root_branches.append(
            (selected, local_sequence, f"root-{root_ordinal}")
        )

    root_sequence = [
        np.asarray(level, dtype=object) for level in root.tree.assignments[1:]
    ]
    subtree_sequence = _combine_refinement_sequences(
        np.asarray(root.tree.assignments[-1], dtype=object), root_branches
    )
    complete_sequence = [*root_sequence, *subtree_sequence]
    if not complete_sequence:
        complete_sequence = [np.zeros(len(cells), dtype=np.int64)]
    complete_resolutions: list[float | None] = [
        *root.tree.resolutions[1:],
        *([None] * len(subtree_sequence)),
    ]
    if len(complete_resolutions) < len(complete_sequence):
        complete_resolutions.extend(
            [None] * (len(complete_sequence) - len(complete_resolutions))
        )
    tree = hierarchy_from_partitions(
        complete_sequence,
        cell_ids=cells,
        resolutions=complete_resolutions,
    )
    stop_reason = "automatic_subtrees_complete"
    _emit_hierarchy_trace(
        tree,
        random_state=random_state,
        stop_reason=stop_reason,
        accepted_resolutions=all_resolutions,
        trace=trace,
    )
    return HierarchyBuild(
        tree=tree,
        steps=tuple(all_steps),
        stop_reason=stop_reason,
        accepted_resolutions=tuple(all_resolutions),
        comparisons=tuple(all_comparisons),
    )
