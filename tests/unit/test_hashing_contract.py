from __future__ import annotations

import hashlib

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rapids_choir._hashing import (
    anndata_input_hashes,
    hash_matrix,
    membership_sha256,
)


def _prepared_adata() -> ad.AnnData:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    adata = ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(
            {"batch": ["a", "a", "b", "b", "a", "b"]},
            index=[f"cell-{index}" for index in range(6)],
        ),
        var=pd.DataFrame(
            {"highly_variable": [True, True, False, True]},
            index=[f"gene-{index}" for index in range(4)],
        ),
    )
    adata.layers["log1p"] = sparse.csr_matrix(values / 10)
    adata.obsm["X_pca"] = values[:, :2].copy()
    adata.obsp["choir_nn"] = sparse.eye(6, dtype=np.float32, format="csr")
    adata.obsp["choir_snn"] = sparse.eye(6, dtype=np.float32, format="csr")
    return adata


def _identity(adata: ad.AnnData) -> tuple[str, dict[str, str]]:
    return anndata_input_hashes(
        adata,
        layer="log1p",
        use_rep="X_pca",
        nn_graph_key="choir_nn",
        snn_graph_key="choir_snn",
        feature_mask_key="highly_variable",
        batch_key="batch",
    )


def test_dense_and_sparse_matrix_hashes_are_equivalent() -> None:
    values = np.asarray([[0, 2, 0], [3, 0, 4]], dtype=np.float32)
    assert hash_matrix(values) == hash_matrix(sparse.csr_matrix(values))
    assert hash_matrix(values) == hash_matrix(sparse.csc_matrix(values))


def test_membership_hash_has_a_cross_language_ascii_contract() -> None:
    expected = hashlib.sha256(
        b"rapids-choir/membership/v1\0" + b"1,3,7"
    ).hexdigest()
    assert membership_sha256([7, 1, 3, 3]) == expected


@pytest.mark.parametrize(
    "component",
    [
        "features",
        "representation",
        "nn_graph",
        "snn_graph",
        "feature_mask",
        "batches",
        "obs_names",
        "var_names",
    ],
)
def test_every_content_bearing_input_invalidates_identity(component: str) -> None:
    baseline = _prepared_adata()
    baseline_sha, baseline_components = _identity(baseline)
    changed = baseline.copy()
    if component == "features":
        matrix = changed.layers["log1p"].tolil()
        matrix[0, 0] += 1
        changed.layers["log1p"] = matrix.tocsr()
    elif component == "representation":
        changed.obsm["X_pca"][0, 0] += 1
    elif component in {"nn_graph", "snn_graph"}:
        key = "choir_nn" if component == "nn_graph" else "choir_snn"
        matrix = changed.obsp[key].tolil()
        matrix[0, 1] = 0.5
        changed.obsp[key] = matrix.tocsr()
    elif component == "feature_mask":
        changed.var.loc["gene-2", "highly_variable"] = True
    elif component == "batches":
        changed.obs.loc["cell-0", "batch"] = "b"
    elif component == "obs_names":
        changed.obs_names = ["renamed", *changed.obs_names[1:]]
    elif component == "var_names":
        changed.var_names = ["renamed", *changed.var_names[1:]]
    changed_sha, changed_components = _identity(changed)
    assert changed_sha != baseline_sha
    assert changed_components[component] != baseline_components[component]
