from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

cp = pytest.importorskip("cupy")
cps = pytest.importorskip("cupyx.scipy.sparse")

import rapids_choir._comparator as engine  # noqa: E402
from rapids_choir._comparator import ChoirComparator  # noqa: E402
from rapids_choir._enums import RuleSet  # noqa: E402
from rapids_choir.config import ComparisonConfig, RFProfile  # noqa: E402
from rapids_choir.exceptions import (  # noqa: E402
    GraphSemanticError,
    InsufficientDeviceMemoryError,
)
from rapids_choir.tl import compare_clusters  # noqa: E402

ROOT = Path(__file__).parents[2]


def separable_inputs() -> tuple[object, object, object, object, object]:
    generator = cp.random.RandomState(91)
    first = generator.normal(1, 0.2, size=(20, 5)).astype(cp.float32)
    second = generator.normal(5, 0.2, size=(20, 5)).astype(cp.float32)
    features = cp.ascontiguousarray(cp.vstack((first, second)))
    graph = cps.csr_matrix(cp.ones((40, 40), dtype=cp.float32))
    cluster1 = cp.arange(0, 20, dtype=cp.int64)
    cluster2 = cp.arange(20, 40, dtype=cp.int64)
    mask = cp.asarray([True, True, True, False, True], dtype=cp.bool_)
    return features, graph, cluster1, cluster2, mask


def fast_config(**overrides: object) -> ComparisonConfig:
    values: dict[str, object] = {
        "n_iterations": 4,
        "n_trees": 5,
        "max_repeat_errors": 0,
        "input_policy": "device_only",
        "random_state": 1,
        "rf_profile": RFProfile(),
    }
    values.update(overrides)
    return ComparisonConfig(**values)


@pytest.mark.gpu
def test_dense_and_csr_comparators_match_and_record_four_fits_per_round() -> None:
    features, graph, cluster1, cluster2, mask = separable_inputs()
    dense = ChoirComparator(fast_config()).compare(
        features,
        cluster1,
        cluster2,
        graph,
        feature_mask=mask,
    )
    csr = ChoirComparator(fast_config()).compare(
        cps.csr_matrix(features),
        cluster1,
        cluster2,
        graph,
        feature_mask=mask,
    )
    assert dense.decision == csr.decision
    assert dense.metrics is not None and csr.metrics is not None
    assert dense.metrics.mean_observed_accuracy == pytest.approx(
        csr.metrics.mean_observed_accuracy, abs=1e-7
    )
    assert dense.samples.split_rounds == 2
    assert dense.samples.forest_fits == 8
    assert len(dense.seeds.fit_seeds) == 8
    assert dense.features.selected == 4


@pytest.mark.gpu
def test_short_circuit_order_is_connectivity_then_batch_then_features() -> None:
    features = cp.zeros((8, 3), dtype=cp.float32)
    cluster1 = cp.arange(0, 4, dtype=cp.int64)
    cluster2 = cp.arange(4, 8, dtype=cp.int64)
    disconnected = cps.csr_matrix((8, 8), dtype=cp.float32)
    no_common_batch = cp.asarray([0] * 4 + [1] * 4, dtype=cp.int32)
    result = ChoirComparator(fast_config()).compare(
        features,
        cluster1,
        cluster2,
        disconnected,
        batch_codes=no_common_batch,
    )
    assert result.reason_code.value == "split_min_connections"
    assert result.samples.forest_fits == 0

    connected = cps.csr_matrix(cp.ones((8, 8), dtype=cp.float32))
    result = ChoirComparator(fast_config()).compare(
        features,
        cluster1,
        cluster2,
        connected,
        batch_codes=no_common_batch,
    )
    assert result.reason_code.value == "merge_batch_dependent"

    result = ChoirComparator(fast_config()).compare(
        features,
        cluster1,
        cluster2,
        connected,
        batch_codes=cp.zeros(8, dtype=cp.int32),
    )
    assert result.reason_code.value == "merge_minimum_reads"


@pytest.mark.gpu
def test_batch_metrics_and_complete_records_are_present() -> None:
    features, graph, cluster1, cluster2, mask = separable_inputs()
    batches = cp.zeros(40, dtype=cp.int32)
    result = ChoirComparator(fast_config(rules="corrected")).compare(
        features,
        cluster1,
        cluster2,
        graph,
        feature_mask=mask,
        batch_codes=batches,
        cluster1_id="left",
        cluster2_id="right",
    )
    assert result.metrics is not None
    assert "0" in result.metrics.batch_mean_accuracies
    assert result.cluster1_id == "left"
    assert result.samples.observed_accuracies == 4
    assert result.samples.null_accuracies == 4
    assert result.memory.estimated_bytes > 0
    assert result.timings.total_seconds > 0
    assert result.compatibility_aliases["decision"]


@pytest.mark.gpu
def test_batch_summaries_preserve_reference_position_quirks() -> None:
    fixture = json.loads(
        (
            ROOT / "validation/fixtures/generated/batch_positional_metrics.json"
        ).read_text()
    )
    observed = cp.asarray(
        fixture["flattened_observed_accuracies"], dtype=cp.float64
    )
    eligible = cp.asarray([0, 1], dtype=cp.int32)
    batches = cp.asarray([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10, dtype=cp.int32)
    cluster1 = cp.arange(0, 20, dtype=cp.int64)
    cluster2 = cp.arange(20, 40, dtype=cp.int64)

    compatibility = fast_config(rules="choir_v0_3_0")
    _, compatibility_rounds, _ = engine._batch_sampling_plan(
        batches, cluster1, cluster2, compatibility
    )
    cp.testing.assert_array_equal(
        compatibility_rounds, cp.asarray([0, 1, 0], dtype=cp.int32)
    )
    means, _ = engine._batch_accuracy_summaries(
        observed, eligible, compatibility_rounds, RuleSet.CHOIR_V0_3_0
    )
    assert means == pytest.approx({"0": 0.7, "1": 0.62})

    corrected = fast_config(rules="corrected")
    _, corrected_rounds, _ = engine._batch_sampling_plan(
        batches, cluster1, cluster2, corrected
    )
    cp.testing.assert_array_equal(
        corrected_rounds, cp.asarray([0, 1], dtype=cp.int32)
    )
    means, _ = engine._batch_accuracy_summaries(
        observed, eligible, corrected_rounds, RuleSet.CORRECTED
    )
    assert means == pytest.approx({"0": 0.61, "1": 0.81})


@pytest.mark.gpu
def test_repeated_errors_distinguish_stable_ids_from_sampled_positions() -> None:
    compatibility_count, compatibility_mean = engine._repeat_summary(
        [cp.asarray([0], dtype=cp.int64), cp.asarray([1], dtype=cp.int64)],
        cluster_size=20,
        n_iterations=4,
        n_per_half=5,
    )
    corrected_count, corrected_mean = engine._repeat_summary(
        [cp.asarray([7], dtype=cp.int64), cp.asarray([7], dtype=cp.int64)],
        cluster_size=20,
        n_iterations=4,
        n_per_half=5,
    )
    assert int(compatibility_count.item()) == 0
    assert float(compatibility_mean.item()) == 0.0
    assert int(corrected_count.item()) == 1
    assert float(corrected_mean.item()) == pytest.approx(0.125)


@pytest.mark.gpu
def test_host_ann_data_is_ingested_once_and_post_ingest_transfers_are_small() -> None:
    host_features = sparse.csr_matrix(
        np.vstack(
            (
                np.full((20, 4), -2, dtype=np.float32),
                np.full((20, 4), 2, dtype=np.float32),
            )
        )
    )
    adata = ad.AnnData(
        X=host_features,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(40)]),
        var=pd.DataFrame(
            {"highly_variable": [True] * 4},
            index=[f"gene{i}" for i in range(4)],
        ),
    )
    adata.obsp["choir_nn"] = sparse.csr_matrix(
        np.ones((40, 40), dtype=np.float32)
    )
    result = compare_clusters(
        adata,
        cluster1_cells=adata.obs_names[:20],
        cluster2_cells=adata.obs_names[20:],
        layer="X",
        n_iterations=4,
        n_trees=5,
        max_repeat_errors=0,
        rf_profile=RFProfile(),
    )
    ingests = [
        transfer
        for transfer in result.memory.transfers
        if transfer.direction == "host_to_device"
    ]
    assert len(ingests) == 1
    post_ingest = sum(
        transfer.bytes
        for transfer in result.memory.transfers
        if transfer.direction == "device_to_host" and not transfer.approved
    )
    assert post_ingest <= 1024 * 1024


@pytest.mark.gpu
def test_vram_preflight_fails_without_adapting_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, graph, cluster1, cluster2, mask = separable_inputs()
    fake = SimpleNamespace(
        cupy=cp,
        memory_info=lambda: (1024, 1024),
    )
    monkeypatch.setattr(engine, "initialize_cuda", lambda _device: fake)
    with pytest.raises(InsufficientDeviceMemoryError) as captured:
        ChoirComparator(fast_config(min_connections=0)).compare(
            features,
            cluster1,
            cluster2,
            graph,
            feature_mask=mask,
        )
    assert captured.value.free_bytes == 1024


@pytest.mark.gpu
def test_repeated_comparison_is_fully_recomputed() -> None:
    features, graph, cluster1, cluster2, mask = separable_inputs()
    comparator = ChoirComparator(fast_config())
    comparator.compare(features, cluster1, cluster2, graph, feature_mask=mask)
    repeated = comparator.compare(
        features, cluster1, cluster2, graph, feature_mask=mask
    )
    assert not hasattr(comparator, "_cache")
    assert "cache_lookup" not in repeated.timings.components_seconds
    assert repeated.timings.components_seconds["observed_fits"] > 0
    assert repeated.timings.components_seconds["null_fits"] > 0


@pytest.mark.gpu
def test_graph_shape_is_validated() -> None:
    features, _, cluster1, cluster2, mask = separable_inputs()
    with pytest.raises(GraphSemanticError, match="square and aligned"):
        ChoirComparator(fast_config()).compare(
            features,
            cluster1,
            cluster2,
            cps.csr_matrix((39, 39), dtype=cp.float32),
            feature_mask=mask,
        )


@pytest.mark.gpu
@pytest.mark.reference
def test_synthetic_ranger_and_cuml_fixture_agree() -> None:
    fixture = json.loads(
        (ROOT / "validation/fixtures/comparator_synthetic.json").read_text()
    )
    rows, columns = fixture["feature_shape"]
    features = cp.asarray(
        fixture["features_row_major"], dtype=cp.float32
    ).reshape(rows, columns)
    graph = cps.csr_matrix(cp.ones((rows, rows), dtype=cp.float32))
    config = ComparisonConfig(
        rules="choir_v0_3_0",
        input_policy="device_only",
        n_iterations=fixture["n_iterations"],
        n_trees=fixture["n_trees"],
        max_repeat_errors=0,
        random_state=fixture["seed"],
        rf_profile=RFProfile(),
    )
    for reference in fixture["results"]:
        result = ChoirComparator(config).compare(
            features,
            cp.asarray(reference["cluster1_indices_1_based"], dtype=cp.int64) - 1,
            cp.asarray(reference["cluster2_indices_1_based"], dtype=cp.int64) - 1,
            graph,
        )
        assert result.decision.value == reference["decision"]
        assert result.metrics is not None
        assert result.metrics.mean_observed_accuracy == pytest.approx(
            reference["mean_observed_accuracy"], abs=0.10
        )
