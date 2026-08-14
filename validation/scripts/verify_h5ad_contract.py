#!/usr/bin/env python3
"""Verify and fingerprint the explicit Gate 2 H5AD mapping."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import scipy.sparse as sp
from gate2_common import (
    file_hash,
    identifier_hash,
    immutable_json,
    validate_candidate_manifest,
)


def nonzero_count(value: Any) -> int:
    if sp.issparse(value):
        return int(value.nnz)
    if hasattr(value, "nnz"):
        return int(value.nnz)
    return int(np.count_nonzero(np.asarray(value)))


def matrix_record(value: Any) -> dict[str, Any]:
    rows, columns = value.shape
    nnz = nonzero_count(value)
    record = {
        "shape": [int(rows), int(columns)],
        "dtype": str(value.dtype),
        "nnz": nnz,
        "sparsity": 1.0 - nnz / (rows * columns),
    }
    if sp.issparse(value):
        record["sparse_format"] = value.format
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--expected-cells", required=True, type=int)
    parser.add_argument("--expected-groups", required=True, type=int)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate = validate_candidate_manifest(
        args.root.resolve(), args.candidate_manifest
    )
    adata = ad.read_h5ad(args.input)
    if adata.n_obs != args.expected_cells:
        raise RuntimeError("H5AD cell count does not match the frozen dataset contract")
    required = {
        "layers[counts]": "counts" in adata.layers,
        "obs[truth_label]": "truth_label" in adata.obs,
        "var[highly_variable]": "highly_variable" in adata.var,
        "obsm[X_pca]": "X_pca" in adata.obsm,
        "obsp[choir_nn]": "choir_nn" in adata.obsp,
    }
    missing = sorted(name for name, present in required.items() if not present)
    if missing:
        raise RuntimeError(f"H5AD mapping is incomplete: {missing}")
    truth_values = adata.obs["truth_label"]
    if truth_values.isna().any():
        raise RuntimeError("H5AD truth labels contain missing values")
    truth = truth_values.astype(str)
    if truth.nunique() != args.expected_groups:
        raise RuntimeError("H5AD truth-label cardinality contract failed")
    hvg = adata.var["highly_variable"].to_numpy(dtype=bool, copy=False)
    if int(hvg.sum()) != 2000:
        raise RuntimeError("H5AD must contain exactly 2,000 variable features")
    if adata.obsm["X_pca"].shape != (adata.n_obs, 50):
        raise RuntimeError("H5AD PCA mapping must be n_obs by 50")
    graph = adata.obsp["choir_nn"]
    if graph.shape != (adata.n_obs, adata.n_obs) or not sp.issparse(graph):
        raise RuntimeError("choir_nn must be an aligned sparse graph")
    if adata.layers["counts"].shape != adata.shape:
        raise RuntimeError("counts layer must align to X")
    counts = adata.layers["counts"]
    count_values = counts.data if sp.issparse(counts) else np.asarray(counts)
    if not np.isfinite(count_values).all() or np.any(count_values < 0):
        raise RuntimeError("counts layer must be finite and non-negative")
    if not np.equal(count_values, np.floor(count_values)).all():
        raise RuntimeError("counts layer must contain integer-valued raw counts")
    if str(adata.uns.get("candidate_commit")) != str(
        candidate["candidate_commit"]
    ):
        raise RuntimeError("H5AD candidate commit does not match current candidate")
    if str(adata.uns.get("runtime_tree_sha256")) != str(
        candidate["runtime_tree_sha256"]
    ):
        raise RuntimeError("H5AD runtime digest does not match current candidate")

    obs_names = adata.obs_names.astype(str).tolist()
    var_names = adata.var_names.astype(str).tolist()
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "candidate": {
            "commit": candidate["candidate_commit"],
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": file_hash(args.candidate_manifest),
            "dirty_state": candidate["dirty_state"],
        },
        "dataset_id": args.dataset_id,
        "path": str(args.input.resolve()),
        "bytes": args.input.stat().st_size,
        "sha256": file_hash(args.input),
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "ground_truth_groups": int(truth.nunique()),
        "mappings": {
            "X": matrix_record(adata.X),
            "layers[counts]": matrix_record(adata.layers["counts"]),
            "obsm[X_pca]": matrix_record(adata.obsm["X_pca"]),
            "obsp[choir_nn]": matrix_record(graph),
            "obs_truth": "truth_label",
            "var_hvg": "highly_variable",
        },
        "identifiers": {
            "obs_names_sha256": identifier_hash(obs_names),
            "var_names_sha256": identifier_hash(var_names),
            "truth_labels_sha256": identifier_hash(truth.tolist()),
            "hvg_names_sha256": identifier_hash(
                adata.var_names[hvg].astype(str).tolist()
            ),
        },
        "preprocessing": {
            "normalization": "LogNormalize",
            "scale_factor": 10000,
            "variable_feature_method": "vst",
            "variable_features": 2000,
            "principal_components": 50,
            "nearest_neighbors": 15,
        },
    }
    immutable_json(args.output, payload)


if __name__ == "__main__":
    main()
