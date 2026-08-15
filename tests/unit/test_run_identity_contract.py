from __future__ import annotations

from dataclasses import replace

import anndata as ad
import numpy as np
import pytest

from rapids_choir import ChoirConfig, ComparisonConfig, RFProfile, RNAConfig
from rapids_choir.exceptions import GPUUnavailableError, MissingCalibrationError
from rapids_choir.tl import _choir
from rapids_choir.tl._choir import _choir_impl, _run_identity


def _config() -> ChoirConfig:
    return ChoirConfig(comparison=ComparisonConfig(rf_profile=RFProfile()))


def test_run_identity_binds_effective_comparator_and_hierarchy_config() -> None:
    input_sha256 = "a" * 64
    baseline = _config()
    comparator_changed = replace(
        baseline,
        comparison=replace(baseline.comparison, alpha=0.04),
    )
    hierarchy_changed = replace(baseline, max_resolution_steps=64)
    preprocessing_changed = replace(baseline, preprocessing=RNAConfig())
    profile_changed = replace(
        baseline,
        comparison=replace(
            baseline.comparison,
            rf_profile=RFProfile(n_bins=256),
        ),
    )
    identities = {
        _run_identity(input_sha256, config)
        for config in (
            baseline,
            comparator_changed,
            hierarchy_changed,
            preprocessing_changed,
            profile_changed,
        )
    }
    assert len(identities) == 5
    assert _run_identity("b" * 64, baseline) not in identities


def test_missing_default_profile_fails_before_preprocessing_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAPIDS_CHOIR_RF_PROFILE", raising=False)
    adata = ad.AnnData(np.asarray([[1, 2], [3, 4]], dtype=np.float32))
    config = ChoirConfig(
        preprocessing=RNAConfig(n_top_genes=2, n_pcs=1, n_neighbors=1)
    )
    with pytest.raises(MissingCalibrationError):
        _choir_impl(adata, config)
    assert not adata.layers
    assert "rapids_choir_prepare_rna" not in adata.uns


def test_gpu_preflight_fails_before_preprocessing_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_device: str) -> None:
        raise GPUUnavailableError("fixture unavailable")

    monkeypatch.setattr(_choir, "initialize_cuda", unavailable)
    adata = ad.AnnData(np.asarray([[1, 2], [3, 4]], dtype=np.float32))
    config = ChoirConfig(
        preprocessing=RNAConfig(n_top_genes=2, n_pcs=1, n_neighbors=1),
        comparison=ComparisonConfig(rf_profile=RFProfile()),
    )
    with pytest.raises(GPUUnavailableError):
        _choir_impl(adata, config)
    assert not adata.layers
    assert "rapids_choir_prepare_rna" not in adata.uns
