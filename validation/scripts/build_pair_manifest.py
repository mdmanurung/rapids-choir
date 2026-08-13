#!/usr/bin/env python3
"""Freeze cross-label comparisons and deterministic within-label controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
from gate2_common import atomic_json, file_hash, identifier_hash


def stable_halves(cells: list[str], seed: int) -> tuple[list[str], list[str]]:
    key = seed.to_bytes(8, "little", signed=False)

    def order(cell: str) -> tuple[bytes, str]:
        digest = hashlib.blake2b(
            cell.encode("utf-8"),
            digest_size=16,
            key=key,
            person=b"rch-pair-v1",
        ).digest()
        return digest, cell

    ordered = sorted(cells, key=order)
    midpoint = len(ordered) // 2
    if midpoint == 0:
        raise RuntimeError("merge controls require at least two cells per population")
    return ordered[:midpoint], ordered[midpoint:]


def cross_label_edges(
    graph: Any, labels: np.ndarray[Any, Any]
) -> dict[tuple[str, str], int]:
    if not sp.issparse(graph):
        graph = sp.csr_matrix(graph)
    rows, columns = graph.nonzero()
    counts: dict[tuple[str, str], int] = {}
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        left = str(labels[row])
        right = str(labels[column])
        if left == right:
            continue
        pair = tuple(sorted((left, right)))
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--data-manifest", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    data_manifest = json.loads(args.data_manifest.read_text(encoding="utf-8"))
    if data_manifest.get("status") != "PASS":
        raise RuntimeError("H5AD contract evidence is not PASS")
    adata = ad.read_h5ad(args.h5ad)
    labels = adata.obs["truth_label"].astype(str).to_numpy()
    cell_names = adata.obs_names.astype(str).to_numpy()
    groups = sorted(set(labels.tolist()))
    edge_counts = cross_label_edges(adata.obsp["choir_nn"], labels)

    pairs: list[dict[str, Any]] = []
    for left, right in combinations(groups, 2):
        key = tuple(sorted((left, right)))
        if key not in edge_counts:
            continue
        cells1 = cell_names[labels == left].tolist()
        cells2 = cell_names[labels == right].tolist()
        pairs.append(
            {
                "pair_id": f"cross::{left}::{right}",
                "kind": "cross_label",
                "expected_decision": "split",
                "cluster1_label": left,
                "cluster2_label": right,
                "cluster1_cells": cells1,
                "cluster2_cells": cells2,
                "directed_nn_edges": edge_counts[key],
            }
        )
    for group in groups:
        cells = cell_names[labels == group].tolist()
        first, second = stable_halves(cells, args.seed)
        pairs.append(
            {
                "pair_id": f"merge_control::{group}",
                "kind": "merge_control",
                "expected_decision": "merge",
                "cluster1_label": f"{group}::half1",
                "cluster2_label": f"{group}::half2",
                "cluster1_cells": first,
                "cluster2_cells": second,
                "directed_nn_edges": None,
            }
        )
    pairs.sort(key=lambda pair: (pair["kind"], pair["pair_id"]))
    if not pairs:
        raise RuntimeError("pair manifest would be empty")
    flattened_ids = [
        cell
        for pair in pairs
        for key in ("cluster1_cells", "cluster2_cells")
        for cell in pair[key]
    ]
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "dataset_id": args.dataset_id,
            "h5ad_path": str(args.h5ad.resolve()),
            "h5ad_sha256": file_hash(args.h5ad),
            "data_manifest_sha256": file_hash(args.data_manifest),
            "truth_column": "truth_label",
            "ground_truth_groups": len(groups),
            "seed": args.seed,
            "pair_count": len(pairs),
            "cross_label_pair_count": sum(
                pair["kind"] == "cross_label" for pair in pairs
            ),
            "merge_control_pair_count": sum(
                pair["kind"] == "merge_control" for pair in pairs
            ),
            "referenced_cell_ids_sha256": identifier_hash(flattened_ids),
            "pairs": pairs,
        },
    )


if __name__ == "__main__":
    main()
