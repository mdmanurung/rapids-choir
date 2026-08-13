from __future__ import annotations

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from rapids_choir import RFProfile
from rapids_choir.exceptions import GraphSemanticError
from rapids_choir.tl import _compare


def fixture_adata() -> ad.AnnData:
    matrix = sparse.csr_matrix(np.arange(24, dtype=np.float32).reshape(6, 4))
    adata = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(
            {"batch": ["b", "a", "a", "b", "b", "a"]},
            index=[f"cell{i}" for i in range(6)],
        ),
        var=pd.DataFrame(
            {"highly_variable": [True, False, True, False]},
            index=[f"gene{i}" for i in range(4)],
        ),
    )
    adata.layers["log1p"] = matrix.copy()
    adata.obsp["choir_nn"] = sparse.eye(6, format="csr", dtype=np.float32)
    return adata


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        (["cell4", "cell1"], [4, 1]),
        ([False, True, False, True, False, False], [1, 3]),
        (np.asarray([5, 0]), [5, 0]),
        (slice(1, 5, 2), [1, 3]),
        ("cell2", [2]),
    ],
)
def test_cell_selectors_resolve_to_stable_positions(
    selector: object, expected: list[int]
) -> None:
    adata = fixture_adata()
    result = _compare._resolve_cell_selector(
        selector, adata.obs_names, name="cells"
    )
    assert result.tolist() == expected


def test_cell_selectors_fail_closed() -> None:
    adata = fixture_adata()
    with pytest.raises(KeyError, match="unknown"):
        _compare._resolve_cell_selector(
            ["cell0", "missing"], adata.obs_names, name="cells"
        )
    with pytest.raises(ValueError, match="unique"):
        _compare._resolve_cell_selector(
            ["cell0", "cell0"], adata.obs_names, name="cells"
        )
    with pytest.raises(ValueError, match="length n_obs"):
        _compare._resolve_cell_selector(
            [True, False], adata.obs_names, name="cells"
        )


def test_missing_graph_fails_before_cuda_initialization() -> None:
    adata = fixture_adata()
    del adata.obsp["choir_nn"]
    with pytest.raises(GraphSemanticError, match="does not exist"):
        _compare.compare_clusters(
            adata,
            cluster1_cells=["cell0"],
            cluster2_cells=["cell1"],
        )


def test_adapter_resolves_layer_mask_batches_and_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adata = fixture_adata()
    captured: dict[str, object] = {}

    class FakeComparator:
        def __init__(self, config: object) -> None:
            captured["config"] = config

        def compare(self, *args: object, **kwargs: object) -> object:
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(ok=True)

    monkeypatch.setattr(_compare, "ChoirComparator", FakeComparator)
    result = _compare.compare_clusters(
        adata,
        cluster1_cells=["cell0", "cell2"],
        cluster2_cells=["cell3", "cell5"],
        rf_profile={"n_bins": 512, "max_depth": None, "n_streams": 4},
        batch_key="batch",
    )
    assert result.ok
    config = captured["config"]
    assert config.rf_profile == RFProfile(n_bins=512, max_depth=None, n_streams=4)
    assert captured["args"][0] is adata.layers["log1p"]
    kwargs = captured["kwargs"]
    assert kwargs["feature_mask"].tolist() == [True, False, True, False]
    assert kwargs["batch_codes"].tolist() == [1, 0, 0, 1, 1, 0]
