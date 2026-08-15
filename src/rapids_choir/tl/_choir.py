"""Public AnnData CHOIR orchestration over the pure hierarchy state machines."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .._comparator import ChoirComparator, _as_device_matrix, _as_device_vector
from .._cuda import initialize_cuda
from .._enums import InputPolicy, Multiplicity, RuleSet
from .._hashing import (
    anndata_input_hashes,
    canonical_json_bytes,
    membership_sha256,
    sha256_bytes,
)
from .._persistence import write_comparison_records
from .._semantic_trace import TraceRecorder
from ..config import ChoirConfig, ComparisonConfig, RNAConfig
from ..exceptions import ArtifactCollisionError, PreparedInputError
from ..pp import prepare_rna, validate_prepared_rna
from ..results import ChoirResult
from ._hierarchy import ClusteringFailure, build_automatic_hierarchy
from ._pruning import PruningCluster, prune_hierarchy

RESULT_SCHEMA = "rapids_choir.result/v1"


@dataclass(frozen=True, slots=True)
class _DeviceInputs:
    features: Any
    nn_graph: Any
    snn_graph: Any
    feature_mask: Any
    batch_codes: Any | None
    transferred_bytes: int


def _batch_codes(adata: Any, key: str | None) -> Any | None:
    if key is None:
        return None
    import pandas as pd  # type: ignore[import-untyped]

    codes, _ = pd.factorize(adata.obs[key], sort=True)
    return codes.astype(np.int32, copy=False)


def _ingest_inputs(adata: Any, config: ChoirConfig) -> _DeviceInputs:
    context = initialize_cuda(config.device)
    cp = context.cupy
    input_policy = InputPolicy(config.input_policy)
    matrix = adata.X if config.layer == "X" else adata.layers[config.layer]
    features, feature_bytes = _as_device_matrix(
        matrix,
        name="features",
        dtype=cp.float32,
        csr_required=False,
        policy=input_policy,
    )
    nn_graph, nn_bytes = _as_device_matrix(
        adata.obsp[config.nn_graph_key],
        name="nn_graph",
        dtype=cp.float32,
        csr_required=True,
        policy=input_policy,
    )
    snn_graph, snn_bytes = _as_device_matrix(
        adata.obsp[config.snn_graph_key],
        name="snn_graph",
        dtype=cp.float32,
        csr_required=True,
        policy=input_policy,
    )
    feature_mask, mask_bytes = _as_device_vector(
        adata.var[config.feature_mask_key].to_numpy(dtype=bool, copy=False),
        name="feature_mask",
        dtype=cp.bool_,
        policy=input_policy,
    )
    batch_codes, batch_bytes = _as_device_vector(
        _batch_codes(adata, config.batch_key),
        name="batch_codes",
        dtype=cp.int32,
        policy=input_policy,
        optional=True,
    )
    assert feature_mask is not None
    return _DeviceInputs(
        features=features,
        nn_graph=nn_graph,
        snn_graph=snn_graph,
        feature_mask=feature_mask,
        batch_codes=batch_codes,
        transferred_bytes=sum(
            (feature_bytes, nn_bytes, snn_bytes, mask_bytes, batch_bytes)
        ),
    )


class _CugraphLouvainPartitioner:
    """Cache induced SNN graphs and run the locked cuGraph Louvain backend."""

    def __init__(self, snn_graph: Any) -> None:
        self._snn_graph = snn_graph
        self._graphs: dict[str, Any] = {}

    def _graph(self, members: np.ndarray[Any, Any]) -> Any:
        from .._hashing import membership_sha256

        key = membership_sha256(members)
        if key in self._graphs:
            return self._graphs[key]
        cp = importlib.import_module("cupy")
        cps = importlib.import_module("cupyx.scipy.sparse")
        cudf = importlib.import_module("cudf")
        cugraph = importlib.import_module("cugraph")
        device_members = cp.asarray(members, dtype=cp.int64)
        matrix = self._snn_graph[device_members][:, device_members]
        if not cps.isspmatrix_csr(matrix):
            matrix = cps.csr_matrix(matrix)
        coo = matrix.tocoo()
        keep = (coo.row < coo.col) & (coo.data > 0)
        count = int(members.size)
        vertices = cp.arange(count, dtype=cp.int32)
        frame = cudf.DataFrame(
            {
                "source": cp.concatenate((coo.row[keep].astype(cp.int32), vertices)),
                "destination": cp.concatenate(
                    (coo.col[keep].astype(cp.int32), vertices)
                ),
                "weight": cp.concatenate(
                    (
                        coo.data[keep].astype(cp.float32),
                        cp.full(count, cp.finfo(cp.float32).eps, dtype=cp.float32),
                    )
                ),
            }
        )
        graph = cugraph.Graph(directed=False)
        graph.from_cudf_edgelist(
            frame,
            source="source",
            destination="destination",
            edge_attr="weight",
            renumber=False,
        )
        self._graphs[key] = graph
        return graph

    def __call__(
        self,
        resolution: float,
        members: np.ndarray[Any, np.dtype[np.int64]],
        seed: int,
    ) -> Any:
        del seed  # cuGraph 26.06 Louvain exposes no seed parameter.
        cugraph = importlib.import_module("cugraph")
        graph = self._graph(members)
        try:
            partitions, _ = cugraph.louvain(graph, resolution=float(resolution))
        except Exception as error:
            raise ClusteringFailure(str(error)) from error
        ordered = partitions.sort_values("vertex")
        labels = ordered["partition"].to_cupy()
        if int(labels.size) != int(members.size):
            raise ClusteringFailure("cuGraph omitted vertices from Louvain output")
        return labels.get()


def _approximate_silhouette(representation: Any) -> Any:
    coordinates = np.asarray(representation, dtype=np.float64)

    def calculate(labels: np.ndarray[Any, Any], members: np.ndarray[Any, Any]) -> float:
        current = coordinates[members]
        unique = list(dict.fromkeys(labels.tolist()))
        if len(unique) < 2:
            return -1.0
        centroids = {
            label: current[labels == label].mean(axis=0) for label in unique
        }
        widths = np.empty(labels.size, dtype=np.float64)
        for index, label in enumerate(labels.tolist()):
            own = np.linalg.norm(current[index] - centroids[label])
            other = min(
                np.linalg.norm(current[index] - centroid)
                for other_label, centroid in centroids.items()
                if other_label != label
            )
            denominator = max(own, other)
            widths[index] = 0.0 if denominator == 0 else (other - own) / denominator
        return float(widths.mean())

    return calculate


def _resolve_choir_config(config: ChoirConfig) -> ChoirConfig:
    if config.comparison is not None:
        if config.comparison.input_policy is InputPolicy.DEVICE_ONLY:
            return config
        return replace(
            config,
            comparison=replace(
                config.comparison,
                input_policy=InputPolicy.DEVICE_ONLY,
            ),
        )
    return replace(
        config,
        comparison=ComparisonConfig(
            rules=config.rules,
            input_policy=InputPolicy.DEVICE_ONLY,
            device=config.device,
            vram_headroom=config.vram_headroom,
            random_state=config.random_state,
        ),
    )


def _comparison_record(comparison: ComparisonConfig) -> dict[str, Any]:
    return {
        "alpha": comparison.alpha,
        "n_iterations": comparison.n_iterations,
        "n_trees": comparison.n_trees,
        "use_variance": comparison.use_variance,
        "min_accuracy": comparison.min_accuracy,
        "min_connections": comparison.min_connections,
        "max_repeat_errors": comparison.max_repeat_errors,
        "sample_max": comparison.sample_max,
        "downsampling_rate": comparison.downsampling_rate,
        "min_reads": comparison.min_reads,
        "rules": RuleSet(comparison.rules).value,
        "input_policy": InputPolicy(comparison.input_policy).value,
        "device": comparison.device,
        "vram_headroom": comparison.vram_headroom,
        "random_state": comparison.random_state,
        "rf_profile": comparison.resolved_rf_profile.to_record(),
        "rf_profile_source": comparison.rf_profile_source,
        "rf_profile_evidence_eligible": comparison.rf_profile_evidence_eligible,
        "rf_profile_manifest_sha256": comparison.rf_profile_manifest_sha256,
    }


def _scientific_config(config: ChoirConfig) -> dict[str, Any]:
    comparison = config.comparison
    if comparison is None:
        raise ValueError("run identity requires a resolved comparison config")
    return {
        "key_added": config.key_added,
        "layer": config.layer,
        "use_rep": config.use_rep,
        "nn_graph_key": config.nn_graph_key,
        "snn_graph_key": config.snn_graph_key,
        "feature_mask_key": config.feature_mask_key,
        "batch_key": config.batch_key,
        "input_policy": InputPolicy(config.input_policy).value,
        "preprocessing": (
            None if config.preprocessing is None else asdict(config.preprocessing)
        ),
        "rules": RuleSet(config.rules).value,
        "multiplicity": Multiplicity(config.multiplicity).value,
        "max_clusters": config.max_clusters,
        "min_cluster_depth": config.min_cluster_depth,
        "group_singletons": config.group_singletons,
        "max_resolution_steps": config.max_resolution_steps,
        "device": config.device,
        "vram_headroom": config.vram_headroom,
        "random_state": config.random_state,
        "comparison": _comparison_record(comparison),
    }


def _run_identity(input_sha256: str, config: ChoirConfig) -> str:
    config = _resolve_choir_config(config)
    payload = {
        "schema": RESULT_SCHEMA,
        "input_sha256": input_sha256,
        "configuration": _scientific_config(config),
    }
    return f"run-{sha256_bytes(canonical_json_bytes(payload))[:32]}"


def _prepare_if_requested(adata: Any, config: ChoirConfig) -> None:
    preprocessing = config.preprocessing
    if preprocessing is None:
        return
    prepare_rna(
        adata,
        counts_layer=preprocessing.counts_layer,
        normalized_layer=preprocessing.normalized_layer,
        target_sum=preprocessing.target_sum,
        hvg_flavor=preprocessing.hvg_flavor,
        n_top_genes=preprocessing.n_top_genes,
        n_pcs=preprocessing.n_pcs,
        n_neighbors=preprocessing.n_neighbors,
        batch_key=config.batch_key,
        use_rep=config.use_rep,
        nn_graph_key=config.nn_graph_key,
        snn_graph_key=config.snn_graph_key,
        feature_mask_key=config.feature_mask_key,
        random_state=config.random_state,
        overwrite=config.overwrite,
    )


def _choir_impl(
    adata: Any,
    config: ChoirConfig,
    *,
    trace: TraceRecorder | None = None,
    preprocessed: bool = False,
) -> ChoirResult:
    if adata is None or not hasattr(adata, "obs"):
        raise TypeError("adata must be an AnnData-like object")
    config = _resolve_choir_config(config)
    comparison_config = config.comparison
    if comparison_config is None:  # pragma: no cover - resolved above
        raise ValueError("comparison configuration resolution failed")
    target = adata if config.inplace else adata.copy()
    collision_names = [
        name
        for name, container in (
            (f"obs[{config.key_added!r}]", target.obs),
            (f"uns[{config.key_added!r}]", target.uns),
        )
        if config.key_added in container
    ]
    if collision_names and not config.overwrite:
        raise ArtifactCollisionError(
            "CHOIR output collision: " + ", ".join(collision_names)
        )
    if (
        config.records_path is not None
        and Path(config.records_path).expanduser().exists()
        and not config.overwrite
    ):
        raise ArtifactCollisionError(
            f"records artifact already exists: {config.records_path}"
        )
    # Fail closed before optional preprocessing can mutate the AnnData object.
    initialize_cuda(config.device)
    if preprocessed:
        if config.preprocessing is None:
            raise ValueError("preprocessed execution requires preprocessing config")
    else:
        _prepare_if_requested(target, config)
    prepared_manifest = validate_prepared_rna(
        target,
        layer=config.layer,
        use_rep=config.use_rep,
        nn_graph_key=config.nn_graph_key,
        snn_graph_key=config.snn_graph_key,
        feature_mask_key=config.feature_mask_key,
        batch_key=config.batch_key,
        require_manifest=config.preprocessing is not None,
    )
    input_sha256, input_components = anndata_input_hashes(
        target,
        layer=config.layer,
        use_rep=config.use_rep,
        nn_graph_key=config.nn_graph_key,
        snn_graph_key=config.snn_graph_key,
        feature_mask_key=config.feature_mask_key,
        batch_key=config.batch_key,
    )
    run_id = _run_identity(input_sha256, config)
    if trace is not None and trace.run_id != run_id:
        raise ValueError("semantic trace run ID does not match the analysis run")
    if trace is not None:
        trace.append(
            "provenance",
            "initialization",
            parameters={
                "schema": RESULT_SCHEMA,
                "input_sha256": input_sha256,
                "configuration": _scientific_config(config),
            },
            seed=config.random_state,
            decision="validated",
        )
        if config.preprocessing is not None:
            if prepared_manifest is None:  # pragma: no cover - fail-closed guard
                raise PreparedInputError("built-in preprocessing manifest is absent")
            all_members = membership_sha256(np.arange(target.n_obs))
            trace.append(
                "preprocessing",
                "preprocessing",
                parameters=dict(prepared_manifest["parameters"]),
                seed=config.random_state,
                input_membership_hashes=(all_members,),
                output_membership_hashes=(all_members,),
                decision="prepared",
                reason="built_in_rna",
                payload={
                    "input_hashes": dict(prepared_manifest["input_hashes"]),
                    "output_hashes": dict(prepared_manifest["output_hashes"]),
                    "input_sha256": prepared_manifest["input_sha256"],
                    "output_sha256": prepared_manifest["output_sha256"],
                },
            )
        trace.append(
            "prepared_input_validation",
            "preprocessing",
            parameters={"components": input_components},
            seed=config.random_state,
            decision="pass",
        )

    if comparison_config.rules is not config.rules:
        raise PreparedInputError("CHOIR and comparator rule sets must agree")
    device_inputs = _ingest_inputs(target, config)
    partitioner = _CugraphLouvainPartitioner(device_inputs.snn_graph)
    representation = np.asarray(target.obsm[config.use_rep], dtype=np.float64)

    def compare(
        left: PruningCluster,
        right: PruningCluster,
        *,
        pair_kind: str,
        alpha: float,
    ) -> Any:
        del pair_kind
        current_config = replace(
            comparison_config,
            alpha=alpha,
            input_policy=InputPolicy.DEVICE_ONLY,
        )
        return ChoirComparator(current_config).compare(
            device_inputs.features,
            importlib.import_module("cupy").asarray(left.members, dtype=np.int64),
            importlib.import_module("cupy").asarray(right.members, dtype=np.int64),
            device_inputs.nn_graph,
            feature_mask=device_inputs.feature_mask,
            batch_codes=device_inputs.batch_codes,
            cluster1_id=left.label,
            cluster2_id=right.label,
        )

    hierarchy = build_automatic_hierarchy(
        partitioner,
        cell_ids=target.obs_names,
        representation=representation,
        silhouette=(
            _approximate_silhouette(target.obsm[config.use_rep])
            if config.max_clusters is None
            else None
        ),
        compare=compare,
        max_clusters=config.max_clusters,
        min_cluster_depth=config.min_cluster_depth,
        multiplicity=config.multiplicity,
        alpha=comparison_config.alpha,
        random_state=config.random_state,
        group_singletons=config.group_singletons,
        rules=config.rules,
        max_steps=config.max_resolution_steps,
        trace=trace,
    )

    def centroid_distance(left: PruningCluster, right: PruningCluster) -> float:
        left_center = representation[np.asarray(left.members)].mean(axis=0)
        right_center = representation[np.asarray(right.members)].mean(axis=0)
        return float(np.linalg.norm(left_center - right_center))

    pruning = prune_hierarchy(
        hierarchy.tree,
        compare,
        alpha=comparison_config.alpha,
        multiplicity=config.multiplicity,
        rules=config.rules,
        use_variance=comparison_config.use_variance,
        distance=centroid_distance,
        trace=trace,
    )
    complete_records = (*hierarchy.comparisons, *pruning.comparisons)
    import pandas as pd

    labels = pd.Series(
        pd.Categorical(
            pruning.labels,
            categories=list(range(1, int(pruning.labels.max()) + 1)),
            ordered=True,
        ),
        index=target.obs_names.copy(),
        name=config.key_added,
    )
    provenance = {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "input_sha256": input_sha256,
        "input_components": input_components,
        "input_contract": {
            "layer": config.layer,
            "use_rep": config.use_rep,
            "nn_graph_key": config.nn_graph_key,
            "snn_graph_key": config.snn_graph_key,
            "feature_mask_key": config.feature_mask_key,
            "batch_key": config.batch_key,
        },
        "configuration": _scientific_config(config),
        "rules": RuleSet(config.rules).value,
        "multiplicity": Multiplicity(config.multiplicity).value,
        "adjusted_alpha": pruning.adjusted_alpha,
        "random_state": config.random_state,
        "device": config.device,
        "initial_host_to_device_bytes": device_inputs.transferred_bytes,
        "hierarchy_stop_reason": hierarchy.stop_reason,
        "accepted_resolutions": list(hierarchy.accepted_resolutions),
        "hierarchy_comparison_count": len(hierarchy.comparisons),
        "pruning_comparison_count": len(pruning.comparisons),
        "record_cluster_labels": list(pruning.record_labels),
    }
    manifest = None
    if config.records_path is not None:
        manifest = write_comparison_records(
            config.records_path,
            complete_records,
            run_id=run_id,
            input_sha256=input_sha256,
            overwrite=config.overwrite,
        )
    result = ChoirResult(
        key_added=config.key_added,
        labels=labels,
        records=complete_records,
        provenance=provenance,
        tree=hierarchy.tree,
        records_manifest=manifest,
    )
    if config.inplace:
        target.obs[config.key_added] = labels
        target.uns[config.key_added] = {
            **provenance,
            "records_manifest": None if manifest is None else asdict(manifest),
            "tree": hierarchy.tree.to_record(),
        }
    return result


def choir(
    adata: Any,
    *,
    key_added: str = "choir",
    layer: str = "log1p",
    use_rep: str = "X_pca",
    nn_graph_key: str = "choir_nn",
    snn_graph_key: str = "choir_snn",
    feature_mask_key: str = "highly_variable",
    batch_key: str | None = None,
    preprocessing: RNAConfig | None = None,
    rules: str = "corrected",
    multiplicity: str = "bonferroni",
    max_clusters: int | str | None = "auto",
    min_cluster_depth: int = 2_000,
    group_singletons: bool = True,
    max_resolution_steps: int = 512,
    comparison: ComparisonConfig | None = None,
    records_path: str | None = None,
    inplace: bool = True,
    overwrite: bool = False,
    device: str = "cuda:0",
    vram_headroom: float = 0.20,
    random_state: int = 1,
) -> ChoirResult:
    """Run CHOIR and always return a :class:`~rapids_choir.ChoirResult`."""

    if max_clusters == "auto":
        resolved_max_clusters = None
    elif isinstance(max_clusters, int) and not isinstance(max_clusters, bool):
        resolved_max_clusters = max_clusters
    elif max_clusters is None:
        resolved_max_clusters = None
    else:
        raise ValueError("max_clusters must be 'auto', None, or a positive integer")
    config = ChoirConfig(
        key_added=key_added,
        layer=layer,
        use_rep=use_rep,
        nn_graph_key=nn_graph_key,
        snn_graph_key=snn_graph_key,
        feature_mask_key=feature_mask_key,
        batch_key=batch_key,
        preprocessing=preprocessing,
        rules=rules,
        multiplicity=multiplicity,
        max_clusters=resolved_max_clusters,
        min_cluster_depth=min_cluster_depth,
        group_singletons=group_singletons,
        max_resolution_steps=max_resolution_steps,
        records_path=records_path,
        inplace=inplace,
        overwrite=overwrite,
        device=device,
        vram_headroom=vram_headroom,
        random_state=random_state,
        comparison=comparison,
    )
    return _choir_impl(adata, config)
