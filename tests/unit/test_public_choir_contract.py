from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rapids_choir import ChoirResult, ComparisonConfig, RFProfile, RNAConfig
from rapids_choir.exceptions import ArtifactCollisionError
from rapids_choir.tl import _choir
from rapids_choir.tl._hierarchy import HierarchyBuild
from rapids_choir.tl._pruning import PruningResult
from rapids_choir.tree import hierarchy_from_partitions


def _adata(*, prepared: bool = True) -> ad.AnnData:
    values = np.asarray(
        [
            [8, 0, 1, 0],
            [7, 1, 0, 0],
            [0, 8, 2, 3],
            [0, 7, 1, 4],
        ],
        dtype=np.float32,
    )
    adata = ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=pd.DataFrame(index=[f"cell-{index}" for index in range(4)]),
        var=pd.DataFrame(index=[f"gene-{index}" for index in range(4)]),
    )
    if prepared:
        adata.layers["log1p"] = sparse.csr_matrix(values / 10)
        adata.var["highly_variable"] = [True, True, True, False]
        adata.obsm["X_pca"] = values[:, :2]
        adata.obsp["choir_nn"] = sparse.eye(
            4, dtype=np.float32, format="csr"
        )
        adata.obsp["choir_snn"] = sparse.eye(
            4, dtype=np.float32, format="csr"
        )
    return adata


def _patch_cpu_state_machines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_choir, "initialize_cuda", lambda _device: None)

    def ingest(adata: Any, config: Any) -> _choir._DeviceInputs:
        del adata, config
        return _choir._DeviceInputs(None, None, None, None, None, 123)

    def hierarchy(*args: Any, **kwargs: Any) -> HierarchyBuild:
        del args
        cells = list(kwargs["cell_ids"])
        tree = hierarchy_from_partitions(
            [[0, 0, 1, 1]], cell_ids=cells, resolutions=[0.5]
        )
        return HierarchyBuild(tree, (), "fixture_stop", (0.5,))

    def pruning(*args: Any, **kwargs: Any) -> PruningResult:
        del args, kwargs
        return PruningResult(
            labels=np.asarray([1, 1, 2, 2], dtype=np.int64),
            record_labels=("left", "left", "right", "right"),
            comparisons=(),
            adjusted_alpha=0.05,
            stepwise_labels=(("left", "left", "right", "right"),),
        )

    monkeypatch.setattr(_choir, "_ingest_inputs", ingest)
    monkeypatch.setattr(_choir, "build_automatic_hierarchy", hierarchy)
    monkeypatch.setattr(_choir, "prune_hierarchy", pruning)


def _comparison() -> ComparisonConfig:
    return ComparisonConfig(rf_profile=RFProfile())


def test_choir_always_returns_result_and_mutates_when_inplace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cpu_state_machines(monkeypatch)
    adata = _adata()
    result = _choir.choir(adata, comparison=_comparison())
    assert isinstance(result, ChoirResult)
    assert result.labels.tolist() == [1, 1, 2, 2]
    assert result.tree.n_cells == adata.n_obs
    assert adata.obs["choir"].tolist() == [1, 1, 2, 2]
    assert adata.uns["choir"]["schema"] == "rapids_choir.result/v1"


def test_non_inplace_analysis_leaves_input_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cpu_state_machines(monkeypatch)
    adata = _adata()
    result = _choir.choir(
        adata, comparison=_comparison(), inplace=False
    )
    assert isinstance(result, ChoirResult)
    assert "choir" not in adata.obs
    assert "choir" not in adata.uns


def test_builtin_preprocessing_and_prepared_input_share_public_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cpu_state_machines(monkeypatch)
    adata = _adata(prepared=False)
    preprocessing = RNAConfig(n_top_genes=3, n_pcs=2, n_neighbors=2)
    result = _choir.choir(
        adata,
        preprocessing=preprocessing,
        comparison=_comparison(),
    )
    assert isinstance(result, ChoirResult)
    assert "rapids_choir_prepare_rna" in adata.uns
    assert "choir_snn" in adata.obsp


def test_output_and_record_collisions_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cpu_state_machines(monkeypatch)
    adata = _adata()
    adata.obs["choir"] = "existing"
    with pytest.raises(ArtifactCollisionError, match="output collision"):
        _choir.choir(adata, comparison=_comparison())

    clean = _adata()
    destination = tmp_path / "records.parquet"
    destination.write_text("existing", encoding="utf-8")
    with pytest.raises(ArtifactCollisionError, match="records artifact"):
        _choir.choir(
            clean,
            comparison=_comparison(),
            records_path=str(destination),
        )


def test_record_artifact_is_linked_to_result_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_cpu_state_machines(monkeypatch)
    destination = tmp_path / "records.parquet"
    result = _choir.choir(
        _adata(),
        comparison=_comparison(),
        records_path=str(destination),
    )
    assert result.records_manifest is not None
    assert result.records_manifest.run_id == result.provenance["run_id"]
    assert result.records_manifest.input_sha256 == result.provenance["input_sha256"]
