"""Comparator public API; implementation is completed at Gate 2."""

from __future__ import annotations

from typing import Any

from ..config import ComparisonConfig
from ..exceptions import RapidsChoirError
from ..results import ComparisonResult


class ComparatorNotCalibratedError(RapidsChoirError):
    """The frozen cuML RF profile has not passed Gate 2 calibration."""


class ChoirComparator:
    """Representation-independent GPU comparator.

    The class is import-safe on CPU hosts. Construction validates only the
    CUDA-independent configuration; GPU modules load on ``compare``.
    """

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config = config if config is not None else ComparisonConfig()

    def compare(
        self,
        features: Any,
        cluster1_indices: Any,
        cluster2_indices: Any,
        nn_graph: Any,
        *,
        feature_mask: Any | None = None,
        batch_codes: Any | None = None,
        cluster1_id: str = "cluster1",
        cluster2_id: str = "cluster2",
    ) -> ComparisonResult:
        raise ComparatorNotCalibratedError(
            "Gate 2 comparator execution is not yet calibrated; "
            "no CPU fallback is allowed"
        )


def compare_clusters(
    adata: Any,
    *,
    cluster1_cells: Any,
    cluster2_cells: Any,
    layer: str = "log1p",
    nn_graph_key: str = "choir_nn",
    feature_mask_key: str = "highly_variable",
    batch_key: str | None = None,
    n_iterations: int = 100,
    n_trees: int = 50,
    alpha: float = 0.05,
    rules: str = "corrected",
    input_policy: str = "transfer_once",
    device: str = "cuda:0",
    vram_headroom: float = 0.20,
    random_state: int = 1,
) -> ComparisonResult:
    config = ComparisonConfig(
        alpha=alpha,
        n_iterations=n_iterations,
        n_trees=n_trees,
        rules=rules,
        input_policy=input_policy,
        device=device,
        vram_headroom=vram_headroom,
        random_state=random_state,
    )
    del adata, cluster1_cells, cluster2_cells, layer, nn_graph_key
    del feature_mask_key, batch_key
    return ChoirComparator(config).compare(None, None, None, None)
