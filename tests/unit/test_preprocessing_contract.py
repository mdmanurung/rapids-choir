from __future__ import annotations

from collections.abc import Callable

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rapids_choir._hashing import hash_matrix
from rapids_choir.exceptions import ArtifactCollisionError, StaleArtifactError
from rapids_choir.pp import prepare_rna
from rapids_choir.pp._rna import PREPARED_UNS_KEY, validate_prepared_rna


def _adata(*, dense: bool = False) -> ad.AnnData:
    counts = np.asarray(
        [
            [8, 0, 1, 0, 3],
            [7, 1, 0, 0, 4],
            [0, 8, 2, 3, 0],
            [0, 7, 1, 4, 0],
            [3, 0, 7, 0, 2],
            [2, 0, 8, 1, 1],
        ],
        dtype=np.float32,
    )
    matrix = counts if dense else sparse.csr_matrix(counts)
    return ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"batch": ["a", "a", "b", "b", "a", "b"]},
            index=[f"cell-{index}" for index in range(6)],
        ),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(5)]),
    )


def _prepare(adata: ad.AnnData, *, copy: bool = True) -> ad.AnnData | None:
    return prepare_rna(
        adata,
        n_top_genes=4,
        n_pcs=3,
        n_neighbors=2,
        batch_key="batch",
        copy=copy,
    )


def _validate(adata: ad.AnnData) -> None:
    validate_prepared_rna(
        adata,
        layer="log1p",
        use_rep="X_pca",
        nn_graph_key="choir_nn",
        snn_graph_key="choir_snn",
        feature_mask_key="highly_variable",
        batch_key="batch",
        require_manifest=True,
    )


def test_prepare_rna_copy_contract_and_dense_csr_equivalence() -> None:
    dense_input = _adata(dense=True)
    sparse_input = _adata(dense=False)
    dense_result = _prepare(dense_input)
    sparse_result = _prepare(sparse_input)
    assert dense_result is not None and sparse_result is not None
    assert PREPARED_UNS_KEY not in dense_input.uns
    assert dense_result.uns[PREPARED_UNS_KEY] == sparse_result.uns[PREPARED_UNS_KEY]
    _validate(dense_result)
    _validate(sparse_result)


def test_prepare_rna_mutates_and_returns_none_by_default() -> None:
    adata = _adata()
    result = _prepare(adata, copy=False)
    assert result is None
    assert PREPARED_UNS_KEY in adata.uns
    assert "counts" in adata.layers


def test_prepare_rna_without_batch_is_deterministic() -> None:
    first = prepare_rna(
        _adata(), n_top_genes=4, n_pcs=3, n_neighbors=2, copy=True
    )
    second = prepare_rna(
        _adata(), n_top_genes=4, n_pcs=3, n_neighbors=2, copy=True
    )
    assert first is not None and second is not None
    assert first.uns[PREPARED_UNS_KEY] == second.uns[PREPARED_UNS_KEY]


def test_prepare_rna_honors_an_explicit_counts_layer() -> None:
    adata = _adata()
    counts = adata.X.copy()
    adata.layers["raw_counts"] = counts
    adata.X = sparse.csr_matrix(np.full(adata.shape, 5, dtype=np.float32))
    prepared = prepare_rna(
        adata,
        counts_layer="raw_counts",
        n_top_genes=4,
        n_pcs=3,
        n_neighbors=2,
        copy=True,
    )
    assert prepared is not None
    assert prepared.uns[PREPARED_UNS_KEY]["input_hashes"]["counts"] == hash_matrix(
        counts
    )


def test_prepare_rna_rejects_collisions_unless_overwrite_is_explicit() -> None:
    adata = _adata()
    _prepare(adata, copy=False)
    with pytest.raises(ArtifactCollisionError, match="overwrite"):
        _prepare(adata, copy=False)
    prepare_rna(
        adata,
        n_top_genes=4,
        n_pcs=3,
        n_neighbors=2,
        batch_key="batch",
        overwrite=True,
    )


def _mutate_sparse(container: object, key: str) -> None:
    matrix = container[key].tolil()  # type: ignore[index]
    matrix[0, 0] += 0.25
    container[key] = matrix.tocsr()  # type: ignore[index]


def _set_batch(adata: ad.AnnData) -> None:
    adata.obs.loc[:, "batch"] = ["b", "a", "b", "b", "a", "b"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: _mutate_sparse(value.layers, "counts"),
        lambda value: _mutate_sparse(value.layers, "log1p"),
        lambda value: value.obsm["X_pca"].__setitem__((0, 0), 99),
        lambda value: _mutate_sparse(value.obsp, "choir_nn"),
        lambda value: _mutate_sparse(value.obsp, "choir_snn"),
        lambda value: value.var.__setitem__(
            "highly_variable", [False, True, True, True, True]
        ),
        _set_batch,
        lambda value: setattr(
            value, "obs_names", ["changed", *value.obs_names[1:]]
        ),
        lambda value: setattr(
            value, "var_names", ["changed", *value.var_names[1:]]
        ),
    ],
    ids=[
        "counts",
        "normalized",
        "representation",
        "nn",
        "snn",
        "feature-mask",
        "batch",
        "obs-names",
        "var-names",
    ],
)
def test_prepared_manifest_rejects_every_stale_input(
    mutation: Callable[[ad.AnnData], None],
) -> None:
    prepared = _prepare(_adata())
    assert prepared is not None
    mutation(prepared)
    with pytest.raises(StaleArtifactError, match="changed"):
        _validate(prepared)
