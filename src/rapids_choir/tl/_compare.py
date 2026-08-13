"""AnnData adapter and public comparator API."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from .._comparator import ChoirComparator
from ..config import ComparisonConfig, RFProfile
from ..exceptions import GraphSemanticError, RapidsChoirError
from ..results import ComparisonResult

__all__ = ["ChoirComparator", "ComparatorNotCalibratedError", "compare_clusters"]


class ComparatorNotCalibratedError(RapidsChoirError):
    """Deprecated compatibility exception; comparator execution is implemented."""


def _resolve_cell_selector(selector: Any, obs_names: Any, *, name: str) -> Any:
    import numpy as np

    pd = importlib.import_module("pandas")
    names = pd.Index(obs_names)
    if not names.is_unique:
        raise ValueError("AnnData obs_names must be unique")
    n_obs = len(names)
    if isinstance(selector, slice):
        positions = np.arange(n_obs, dtype=np.int64)[selector]
    elif isinstance(selector, str):
        positions = names.get_indexer([selector]).astype(np.int64, copy=False)
    elif isinstance(selector, (int, np.integer)) and not isinstance(selector, bool):
        positions = np.asarray([selector], dtype=np.int64)
    else:
        values = np.asarray(selector)
        if values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional selector")
        if values.dtype.kind == "b":
            if values.size != n_obs:
                raise ValueError(f"boolean {name} must have length n_obs")
            positions = np.flatnonzero(values).astype(np.int64, copy=False)
        elif values.dtype.kind in {"i", "u"}:
            positions = values.astype(np.int64, copy=False)
        else:
            positions = names.get_indexer(values.astype(str)).astype(
                np.int64, copy=False
            )
    if positions.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if np.any(positions < 0) or np.any(positions >= n_obs):
        raise KeyError(f"{name} contains unknown names or out-of-range positions")
    if np.unique(positions).size != positions.size:
        raise ValueError(f"{name} must resolve to unique cells")
    return positions


def _resolve_layer(adata: Any, layer: str | None) -> Any:
    if layer is None or layer == "X":
        if getattr(adata, "X", None) is None:
            raise ValueError("AnnData X is empty")
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"AnnData layer {layer!r} does not exist")
    return adata.layers[layer]


def _resolve_feature_mask(adata: Any, key: str | None) -> Any | None:
    if key is None:
        return None
    if key not in adata.var:
        raise KeyError(f"AnnData var field {key!r} does not exist")
    values = adata.var[key]
    if bool(values.isna().any()):
        raise ValueError(f"feature mask {key!r} contains missing values")
    return values.to_numpy(dtype=bool, copy=False)


def _resolve_batches(adata: Any, key: str | None) -> Any | None:
    if key is None:
        return None
    if key not in adata.obs:
        raise KeyError(f"AnnData obs field {key!r} does not exist")
    values = adata.obs[key]
    if bool(values.isna().any()):
        raise ValueError(f"batch field {key!r} contains missing values")
    pd = importlib.import_module("pandas")
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype("int32", copy=False)


def compare_clusters(
    adata: Any,
    *,
    cluster1_cells: Any,
    cluster2_cells: Any,
    layer: str | None = "log1p",
    nn_graph_key: str = "choir_nn",
    feature_mask_key: str | None = "highly_variable",
    batch_key: str | None = None,
    n_iterations: int = 100,
    n_trees: int = 50,
    alpha: float = 0.05,
    use_variance: bool = True,
    min_accuracy: float = 0.5,
    min_connections: float = 1.0,
    max_repeat_errors: int = 20,
    sample_max: int | None = None,
    downsampling_rate: float = 1.0,
    min_reads: float | None = None,
    rules: str = "corrected",
    input_policy: str = "transfer_once",
    device: str = "cuda:0",
    vram_headroom: float = 0.20,
    random_state: int = 1,
    rf_profile: RFProfile | Mapping[str, Any] | None = None,
) -> ComparisonResult:
    """Resolve AnnData fields once, ingest once, and run the GPU comparator."""

    if adata is None or not hasattr(adata, "obs_names"):
        raise TypeError("adata must be an AnnData-like object")
    if nn_graph_key not in adata.obsp:
        raise GraphSemanticError(
            f"AnnData obsp graph {nn_graph_key!r} does not exist"
        )
    cluster1 = _resolve_cell_selector(
        cluster1_cells, adata.obs_names, name="cluster1_cells"
    )
    cluster2 = _resolve_cell_selector(
        cluster2_cells, adata.obs_names, name="cluster2_cells"
    )
    if set(cluster1.tolist()) & set(cluster2.tolist()):
        raise ValueError("cluster selectors must not overlap")
    profile = (
        RFProfile.from_mapping(rf_profile)
        if isinstance(rf_profile, Mapping)
        else rf_profile
    )
    config_kwargs: dict[str, Any] = {
        "alpha": alpha,
        "n_iterations": n_iterations,
        "n_trees": n_trees,
        "use_variance": use_variance,
        "min_accuracy": min_accuracy,
        "min_connections": min_connections,
        "max_repeat_errors": max_repeat_errors,
        "sample_max": sample_max,
        "downsampling_rate": downsampling_rate,
        "min_reads": min_reads,
        "rules": rules,
        "input_policy": input_policy,
        "device": device,
        "vram_headroom": vram_headroom,
        "random_state": random_state,
    }
    if profile is not None:
        config_kwargs["rf_profile"] = profile
    config = ComparisonConfig(**config_kwargs)
    return ChoirComparator(config).compare(
        _resolve_layer(adata, layer),
        cluster1,
        cluster2,
        adata.obsp[nn_graph_key],
        feature_mask=_resolve_feature_mask(adata, feature_mask_key),
        batch_codes=_resolve_batches(adata, batch_key),
    )
