"""Immutable, strictly hierarchical clustering topology."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from ._hashing import membership_sha256


@dataclass(frozen=True, slots=True)
class HierarchyNode:
    """One stable membership node in a hierarchy.

    Membership vectors live once in :class:`HierarchyTree.assignments`; nodes
    retain only their content identity and topology.
    """

    node_id: int
    label: str
    level: int
    ordinal: int
    size: int
    membership_sha256: str
    parent_id: int | None
    child_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class HierarchyTree:
    """A validated immutable hierarchy aligned to ``cell_ids``."""

    cell_ids: tuple[str, ...]
    assignments: tuple[tuple[int, ...], ...]
    nodes: tuple[HierarchyNode, ...]
    resolutions: tuple[float | None, ...]
    root_id: int

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_levels(self) -> int:
        return len(self.assignments)

    @property
    def node_map(self) -> MappingProxyType[int, HierarchyNode]:
        return MappingProxyType({node.node_id: node for node in self.nodes})

    @property
    def leaves(self) -> tuple[HierarchyNode, ...]:
        return tuple(node for node in self.nodes if not node.child_ids)

    def members(self, node_id: int) -> np.ndarray[Any, np.dtype[np.int64]]:
        node = self.node_map[node_id]
        assignment = np.asarray(self.assignments[node.level - 1], dtype=object)
        return np.flatnonzero(assignment == node_id).astype(np.int64, copy=False)

    def labels(self, level: int = -1) -> tuple[str, ...]:
        assignment = self.assignments[level]
        nodes = self.node_map
        return tuple(nodes[node_id].label for node_id in assignment)

    def validate(self) -> None:
        if not self.cell_ids:
            raise ValueError("hierarchy must contain at least one cell")
        if len(set(self.cell_ids)) != len(self.cell_ids):
            raise ValueError("hierarchy cell IDs must be unique")
        if not self.assignments:
            raise ValueError("hierarchy must contain at least one level")
        if len(self.resolutions) != len(self.assignments):
            raise ValueError("hierarchy resolutions must align to levels")
        node_map = {node.node_id: node for node in self.nodes}
        if len(node_map) != len(self.nodes):
            raise ValueError("hierarchy node IDs must be unique")
        if self.root_id not in node_map:
            raise ValueError("hierarchy root does not exist")
        if node_map[self.root_id].parent_id is not None:
            raise ValueError("hierarchy root cannot have a parent")
        for level_index, assignment in enumerate(self.assignments, start=1):
            if len(assignment) != len(self.cell_ids):
                raise ValueError("every hierarchy level must align to all cells")
            for node_id in set(assignment):
                if node_id not in node_map:
                    raise ValueError("hierarchy assignment references an unknown node")
                if node_map[node_id].level > level_index:
                    raise ValueError("node appears before its creation level")
        for node in self.nodes:
            members = self.members(node.node_id)
            if members.size != node.size:
                raise ValueError(f"node {node.label!r} has an invalid size")
            if membership_sha256(members) != node.membership_sha256:
                raise ValueError(f"node {node.label!r} has a stale membership hash")
            if node.parent_id is not None:
                parent = node_map.get(node.parent_id)
                if parent is None or node.node_id not in parent.child_ids:
                    raise ValueError(f"node {node.label!r} has an invalid parent link")
                if not set(members.tolist()) <= set(
                    self.members(parent.node_id).tolist()
                ):
                    raise ValueError("child membership must be contained in its parent")
            child_members: set[int] = set()
            for child_id in node.child_ids:
                child = node_map.get(child_id)
                if child is None or child.parent_id != node.node_id:
                    raise ValueError(f"node {node.label!r} has an invalid child link")
                current = set(self.members(child_id).tolist())
                if child_members & current:
                    raise ValueError("sibling memberships must be disjoint")
                child_members.update(current)
            if node.child_ids and child_members != set(members.tolist()):
                raise ValueError("children must partition their parent membership")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "rapids_choir.hierarchy/v1",
            "cell_ids": list(self.cell_ids),
            # Node identities are full SHA-256 integers.  JSON numbers cannot
            # carry them losslessly through R's IEEE-754 numeric type, so the
            # cross-language wire representation is always decimal text.
            "assignments": [
                [str(node_id) for node_id in level] for level in self.assignments
            ],
            "nodes": [
                {
                    "node_id": str(node.node_id),
                    "label": node.label,
                    "level": node.level,
                    "ordinal": node.ordinal,
                    "size": node.size,
                    "membership_sha256": node.membership_sha256,
                    "parent_id": (
                        None if node.parent_id is None else str(node.parent_id)
                    ),
                    "child_ids": [str(value) for value in node.child_ids],
                }
                for node in self.nodes
            ],
            "resolutions": list(self.resolutions),
            "root_id": str(self.root_id),
        }

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> HierarchyTree:
        if value.get("schema") != "rapids_choir.hierarchy/v1":
            raise ValueError("unsupported hierarchy schema")
        nodes = tuple(
            HierarchyNode(
                node_id=int(item["node_id"]),
                label=str(item["label"]),
                level=int(item["level"]),
                ordinal=int(item["ordinal"]),
                size=int(item["size"]),
                membership_sha256=str(item["membership_sha256"]),
                parent_id=(
                    None if item["parent_id"] is None else int(item["parent_id"])
                ),
                child_ids=tuple(int(child) for child in item["child_ids"]),
            )
            for item in value["nodes"]
        )
        return cls(
            cell_ids=tuple(str(item) for item in value["cell_ids"]),
            assignments=tuple(
                tuple(int(item) for item in level) for level in value["assignments"]
            ),
            nodes=nodes,
            resolutions=tuple(
                None if item is None else float(item) for item in value["resolutions"]
            ),
            root_id=int(value["root_id"]),
        )


def _node_id(membership_hash: str) -> int:
    return int(membership_hash, 16)


def _canonical_partition(values: Any) -> np.ndarray[Any, np.dtype[np.int64]]:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError("partitions must be one-dimensional")
    mapping: dict[Any, int] = {}
    output = np.empty(labels.size, dtype=np.int64)
    for index, value in enumerate(labels.tolist()):
        key = value.item() if isinstance(value, np.generic) else value
        if key not in mapping:
            mapping[key] = len(mapping)
        output[index] = mapping[key]
    return output


def hierarchy_from_partitions(
    partitions: Sequence[Any],
    *,
    cell_ids: Any,
    resolutions: Sequence[float | None] | None = None,
    root: str = "P0",
) -> HierarchyTree:
    """Reconcile resolution partitions into a strict, stable hierarchy.

    Each new raw cluster is intersected with its preceding parent. This is the
    topology-preserving Python counterpart to the R MRtree reconciliation.
    Levels that do not refine any parent are removed.
    """

    cells = tuple(str(item) for item in cell_ids)
    if not partitions:
        raise ValueError("at least one partition is required")
    if len(set(cells)) != len(cells):
        raise ValueError("cell_ids must be unique")
    n_cells = len(cells)
    raw = [_canonical_partition(partition) for partition in partitions]
    if any(partition.size != n_cells for partition in raw):
        raise ValueError("every partition must align to cell_ids")
    source_resolutions = resolutions or [None] * len(raw)
    if len(source_resolutions) != len(raw):
        raise ValueError("resolutions must align to partitions")

    all_cells = np.arange(n_cells, dtype=np.int64)
    root_hash = membership_sha256(all_cells)
    root_id = _node_id(root_hash)
    mutable_nodes: dict[int, HierarchyNode] = {
        root_id: HierarchyNode(
            node_id=root_id,
            label=f"{root}_L1_1",
            level=1,
            ordinal=1,
            size=n_cells,
            membership_sha256=root_hash,
            parent_id=None,
        )
    }
    assignments: list[tuple[int, ...]] = [(root_id,) * n_cells]
    kept_resolutions: list[float | None] = [None]
    previous = np.full(n_cells, root_id, dtype=object)

    for source_index, partition in enumerate(raw):
        groups: dict[tuple[int, int], list[int]] = {}
        order: list[tuple[int, int]] = []
        for cell_index, raw_label in enumerate(partition.tolist()):
            key = (int(previous[cell_index]), int(raw_label))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(cell_index)

        proposed = previous.copy()
        refinements: dict[int, list[tuple[tuple[int, int], list[int]]]] = {}
        for key in order:
            refinements.setdefault(key[0], []).append((key, groups[key]))
        changed = False
        pending_children: dict[int, list[int]] = {}
        level_number = len(assignments) + 1
        ordinal = 0
        for parent_id in dict.fromkeys(int(value) for value in previous.tolist()):
            parent_groups = refinements[parent_id]
            if len(parent_groups) == 1:
                continue
            changed = True
            for _, members_list in parent_groups:
                ordinal += 1
                members = np.asarray(members_list, dtype=np.int64)
                digest = membership_sha256(members)
                node_id = _node_id(digest)
                existing = mutable_nodes.get(node_id)
                if existing is not None and existing.membership_sha256 != digest:
                    raise ValueError("hierarchy node identity collision")
                if existing is None:
                    mutable_nodes[node_id] = HierarchyNode(
                        node_id=node_id,
                        label=f"{root}_L{level_number}_{ordinal}",
                        level=level_number,
                        ordinal=ordinal,
                        size=int(members.size),
                        membership_sha256=digest,
                        parent_id=parent_id,
                    )
                proposed[members] = node_id
                pending_children.setdefault(parent_id, []).append(node_id)
        if not changed:
            continue
        for parent_id, child_ids in pending_children.items():
            parent = mutable_nodes[parent_id]
            mutable_nodes[parent_id] = replace(
                parent, child_ids=tuple(dict.fromkeys(child_ids))
            )
        previous = proposed
        assignments.append(tuple(int(value) for value in proposed.tolist()))
        kept_resolutions.append(source_resolutions[source_index])

    tree = HierarchyTree(
        cell_ids=cells,
        assignments=tuple(assignments),
        nodes=tuple(
            sorted(mutable_nodes.values(), key=lambda node: (node.level, node.ordinal))
        ),
        resolutions=tuple(kept_resolutions),
        root_id=root_id,
    )
    return tree
