"""R CHOIR-shaped public API for AnnData and one CUDA device."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .config import ComparisonConfig, RFProfile, RNAConfig
from .get import records as _records
from .results import ChoirResult, ComparisonResult
from .tl import choir as _choir
from .tl import compare_clusters as _compare_clusters


def _profile(value: RFProfile | Mapping[str, Any] | None) -> RFProfile | None:
    return RFProfile.from_mapping(value) if isinstance(value, Mapping) else value


def _sample_max(value: int | float) -> int | None:
    if isinstance(value, bool):
        raise TypeError("sample_max must be a positive integer or infinity")
    if isinstance(value, float) and math.isinf(value) and value > 0:
        return None
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError("sample_max must be a positive integer or infinity")


def _downsampling_rate(value: str | float, n_cells: int, *, tree: bool) -> float:
    if value == "auto":
        rate = float(min(1.0, 0.5 ** math.log10(n_cells / 5_000)))
        return rate * 0.5 if tree else rate
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("downsampling_rate must be 'auto' or a positive number")
    return float(value)


def _feature_mask(feature_set: str, feature_mask_key: str) -> str | None:
    if feature_set == "var":
        return feature_mask_key
    if feature_set == "all":
        return None
    raise ValueError("feature_set must be 'var' or 'all'")


def _group_singletons(cluster_params: Mapping[str, Any] | None) -> bool:
    if cluster_params is None:
        return True
    unknown = set(cluster_params) - {"algorithm", "group.singletons"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise NotImplementedError(f"unsupported cluster_params: {names}")
    if cluster_params.get("algorithm", 1) != 1:
        raise NotImplementedError("only CHOIR cluster algorithm 1 is supported")
    value = cluster_params.get("group.singletons", True)
    if not isinstance(value, bool):
        raise TypeError("cluster_params['group.singletons'] must be boolean")
    return value


def CHOIR(
    object: Any,
    *,
    key: str = "CHOIR",
    alpha: float = 0.05,
    p_adjust: str = "bonferroni",
    feature_set: str = "var",
    n_iterations: int = 100,
    n_trees: int = 50,
    use_variance: bool = True,
    min_accuracy: float = 0.5,
    min_connections: float = 1,
    max_repeat_errors: int = 20,
    sample_max: int | float = math.inf,
    downsampling_rate: str | float = "auto",
    min_reads: float | None = None,
    max_clusters: int | str | None = "auto",
    min_cluster_depth: int = 2_000,
    normalization_method: str = "none",
    batch_labels: str | None = None,
    cluster_params: Mapping[str, Any] | None = None,
    random_seed: int = 1,
    verbose: bool = True,
    layer: str = "log1p",
    reduction: str = "X_pca",
    nn_graph_key: str = "choir_nn",
    snn_graph_key: str = "choir_snn",
    feature_mask_key: str = "highly_variable",
    preprocessing: RNAConfig | None = None,
    rules: str = "corrected",
    max_resolution_steps: int = 512,
    records_path: str | None = None,
    inplace: bool = True,
    overwrite: bool = False,
    device: str = "cuda:0",
    vram_headroom: float = 0.20,
    rf_profile: RFProfile | Mapping[str, Any] | None = None,
) -> ChoirResult:
    """Run the AnnData CHOIR workflow with R CHOIR parameter names.

    The scientific arguments retain the names and defaults that have direct
    equivalents in CHOIR v0.3.0. AnnData field names and GPU controls are
    explicit additions. The function always returns a :class:`ChoirResult`.
    """

    if object is None or not hasattr(object, "n_obs"):
        raise TypeError("object must be an AnnData-like object")
    if normalization_method != "none":
        raise NotImplementedError(
            "normalization_method must be 'none'; prepare AnnData before CHOIR"
        )
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be boolean")
    comparison = ComparisonConfig(
        alpha=alpha,
        n_iterations=n_iterations,
        n_trees=n_trees,
        use_variance=use_variance,
        min_accuracy=min_accuracy,
        min_connections=min_connections,
        max_repeat_errors=max_repeat_errors,
        sample_max=_sample_max(sample_max),
        downsampling_rate=_downsampling_rate(
            downsampling_rate, int(object.n_obs), tree=True
        ),
        min_reads=min_reads,
        rules=rules,
        device=device,
        vram_headroom=vram_headroom,
        random_state=random_seed,
        rf_profile=_profile(rf_profile),
    )
    return _choir(
        object,
        key_added=key,
        layer=layer,
        use_rep=reduction,
        nn_graph_key=nn_graph_key,
        snn_graph_key=snn_graph_key,
        feature_mask_key=_feature_mask(feature_set, feature_mask_key),
        batch_key=batch_labels,
        preprocessing=preprocessing,
        rules=rules,
        multiplicity=p_adjust,
        max_clusters=max_clusters,
        min_cluster_depth=min_cluster_depth,
        group_singletons=_group_singletons(cluster_params),
        max_resolution_steps=max_resolution_steps,
        comparison=comparison,
        records_path=records_path,
        inplace=inplace,
        overwrite=overwrite,
        device=device,
        vram_headroom=vram_headroom,
        random_state=random_seed,
    )


def compareClusters(
    object: Any,
    *,
    cluster1_cells: Any | None = None,
    cluster2_cells: Any | None = None,
    ident1: Any | None = None,
    ident2: Any | None = None,
    group_by: str | None = None,
    alpha: float = 0.05,
    feature_set: str = "var",
    n_iterations: int = 100,
    n_trees: int = 50,
    use_variance: bool = True,
    min_accuracy: float = 0.5,
    min_connections: float = 1,
    max_repeat_errors: int = 20,
    sample_max: int | float = math.inf,
    downsampling_rate: str | float = "auto",
    min_reads: float | None = None,
    batch_labels: str | None = None,
    random_seed: int = 1,
    layer: str = "log1p",
    nn_graph_key: str = "choir_nn",
    feature_mask_key: str = "highly_variable",
    rules: str = "corrected",
    device: str = "cuda:0",
    input_policy: str = "transfer_once",
    vram_headroom: float = 0.20,
    rf_profile: RFProfile | Mapping[str, Any] | None = None,
) -> ComparisonResult:
    """Compare two AnnData clusters with R CHOIR parameter names."""

    if object is None or not hasattr(object, "obs"):
        raise TypeError("object must be an AnnData-like object")
    direct = cluster1_cells is not None or cluster2_cells is not None
    identified = ident1 is not None or ident2 is not None or group_by is not None
    if direct and identified:
        raise ValueError("use cell selectors or ident1/ident2/group_by, not both")
    if identified:
        if ident1 is None or ident2 is None or group_by is None:
            raise ValueError("ident1, ident2, and group_by must be supplied together")
        if group_by not in object.obs:
            raise KeyError(f"AnnData obs field {group_by!r} does not exist")
        labels = object.obs[group_by]
        cluster1_cells = labels == ident1
        cluster2_cells = labels == ident2
    if cluster1_cells is None or cluster2_cells is None:
        raise ValueError("two cluster selectors are required")
    return _compare_clusters(
        object,
        cluster1_cells=cluster1_cells,
        cluster2_cells=cluster2_cells,
        layer=layer,
        nn_graph_key=nn_graph_key,
        feature_mask_key=_feature_mask(feature_set, feature_mask_key),
        batch_key=batch_labels,
        n_iterations=n_iterations,
        n_trees=n_trees,
        alpha=alpha,
        use_variance=use_variance,
        min_accuracy=min_accuracy,
        min_connections=min_connections,
        max_repeat_errors=max_repeat_errors,
        sample_max=_sample_max(sample_max),
        downsampling_rate=_downsampling_rate(
            downsampling_rate, int(object.n_obs), tree=False
        ),
        min_reads=min_reads,
        rules=rules,
        input_policy=input_policy,
        device=device,
        vram_headroom=vram_headroom,
        random_state=random_seed,
        rf_profile=_profile(rf_profile),
    )


def getRecords(object: Any, key: str = "CHOIR") -> Any:
    """Return CHOIR comparison records, matching R's ``getRecords`` name."""

    return _records(object, key=key)


__all__ = ["CHOIR", "compareClusters", "getRecords"]
