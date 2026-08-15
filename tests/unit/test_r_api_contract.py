from __future__ import annotations

import inspect
import math
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import rapids_choir as rc
import rapids_choir._api as api
from rapids_choir.pp import prepare_rna
from rapids_choir.pp._rna import validate_prepared_rna

R_SHARED_DEFAULTS: dict[str, Any] = {
    "key": "CHOIR",
    "alpha": 0.05,
    "p_adjust": "bonferroni",
    "feature_set": "var",
    "n_iterations": 100,
    "n_trees": 50,
    "use_variance": True,
    "min_accuracy": 0.5,
    "min_connections": 1,
    "max_repeat_errors": 20,
    "sample_max": math.inf,
    "downsampling_rate": "auto",
    "min_reads": None,
    "max_clusters": "auto",
    "min_cluster_depth": 2_000,
    "normalization_method": "none",
    "batch_labels": None,
    "random_seed": 1,
    "verbose": True,
}


def _adata(n_obs: int = 10_000) -> ad.AnnData:
    return ad.AnnData(
        X=sparse.csr_matrix((n_obs, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"cluster": ["A", "B"] * (n_obs // 2)},
            index=[f"cell-{index}" for index in range(n_obs)],
        ),
        var=pd.DataFrame(index=["gene-1", "gene-2"]),
    )


def test_top_level_api_uses_r_choir_names_and_shared_defaults() -> None:
    assert rc.CHOIR is api.CHOIR
    assert rc.compareClusters is api.compareClusters
    assert rc.getRecords is api.getRecords
    parameters = inspect.signature(rc.CHOIR).parameters
    assert parameters["object"].default is inspect.Parameter.empty
    for name, expected in R_SHARED_DEFAULTS.items():
        actual = parameters[name].default
        if isinstance(expected, float) and math.isinf(expected):
            assert math.isinf(actual)
        else:
            assert actual == expected


def test_choir_maps_r_names_and_gpu_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    marker = object()

    def fake_choir(value: Any, **kwargs: Any) -> Any:
        captured["object"] = value
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(api, "_choir", fake_choir)
    adata = _adata()
    result = rc.CHOIR(
        adata,
        feature_set="all",
        p_adjust="fdr",
        random_seed=7,
        device="cuda:2",
        vram_headroom=0.1,
        rf_profile=rc.RFProfile(),
    )
    assert result is marker
    assert captured["object"] is adata
    assert captured["key_added"] == "CHOIR"
    assert captured["feature_mask_key"] is None
    assert captured["multiplicity"] == "fdr"
    assert captured["random_state"] == 7
    assert captured["device"] == "cuda:2"
    comparison = captured["comparison"]
    assert comparison.downsampling_rate == pytest.approx(
        min(1.0, 0.5 ** math.log10(adata.n_obs / 5_000)) * 0.5
    )


def test_compare_clusters_accepts_r_identity_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    marker = object()

    def fake_compare(value: Any, **kwargs: Any) -> Any:
        captured["object"] = value
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(api, "_compare_clusters", fake_compare)
    adata = _adata(4)
    result = rc.compareClusters(
        adata,
        ident1="A",
        ident2="B",
        group_by="cluster",
        feature_set="all",
        rf_profile=rc.RFProfile(),
    )
    assert result is marker
    assert captured["cluster1_cells"].tolist() == [True, False, True, False]
    assert captured["cluster2_cells"].tolist() == [False, True, False, True]
    assert captured["feature_mask_key"] is None
    assert captured["downsampling_rate"] == 1.0


def test_r_api_rejects_unsupported_normalization() -> None:
    with pytest.raises(NotImplementedError, match="prepare AnnData"):
        rc.CHOIR(
            _adata(4),
            normalization_method="SCTransform",
            rf_profile=rc.RFProfile(),
        )


def test_preparation_manifest_allows_all_feature_analysis() -> None:
    adata = ad.AnnData(
        X=sparse.csr_matrix(
            np.asarray(
                [
                    [8, 0, 1],
                    [7, 1, 0],
                    [0, 8, 2],
                    [0, 7, 1],
                ],
                dtype=np.float32,
            )
        ),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(4)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(3)]),
    )
    prepared = prepare_rna(
        adata,
        n_top_genes=2,
        n_pcs=2,
        n_neighbors=2,
        copy=True,
    )
    assert prepared is not None
    validate_prepared_rna(
        prepared,
        layer="log1p",
        use_rep="X_pca",
        nn_graph_key="choir_nn",
        snn_graph_key="choir_snn",
        feature_mask_key=None,
        batch_key=None,
    )
