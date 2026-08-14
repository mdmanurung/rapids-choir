"""Device-resident cluster comparator implementation."""

from __future__ import annotations

import importlib
from time import perf_counter
from typing import Any

from ._cuda import initialize_cuda
from ._decision import decide_comparison
from ._enums import REFERENCE_REASON_ALIASES, Decision, InputPolicy, ReasonCode, RuleSet
from ._memory import (
    TransferLedger,
    estimate_comparison_bytes,
    preflight_device_memory,
)
from ._sampling import balanced_sample_size, balanced_split_round, eligible_batch_codes
from ._seeding import derive_seed, device_membership_fingerprint
from ._stats import overall_accuracy, tail_probabilities
from .config import ComparisonConfig, RFProfile
from .exceptions import GraphSemanticError, InputResidencyError
from .results import (
    ComparisonMetrics,
    ComparisonResult,
    FeatureCounts,
    MemoryRecord,
    SampleCounts,
    SeedRecord,
    TimingRecord,
)


def _module(name: str) -> Any:
    return importlib.import_module(name)


def _host_nbytes(value: Any) -> int:
    if value is None:
        return 0
    try:
        sparse = _module("scipy.sparse")
        if sparse.issparse(value):
            matrix = value.tocsr(copy=False)
            return int(
                matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            )
    except ImportError:
        pass
    return int(getattr(value, "nbytes", 0))


def _device_csr_from_host(value: Any, *, dtype: Any) -> Any:
    cp = _module("cupy")
    cps = _module("cupyx.scipy.sparse")
    scipy_sparse = _module("scipy.sparse")
    matrix = scipy_sparse.csr_matrix(value, dtype=dtype)
    return cps.csr_matrix(
        (
            cp.asarray(matrix.data, dtype=dtype),
            cp.asarray(matrix.indices, dtype=cp.int32),
            cp.asarray(matrix.indptr, dtype=cp.int32),
        ),
        shape=matrix.shape,
    )


def _as_device_matrix(
    value: Any,
    *,
    name: str,
    dtype: Any,
    csr_required: bool,
    policy: InputPolicy,
) -> tuple[Any, int]:
    cp = _module("cupy")
    cps = _module("cupyx.scipy.sparse")
    if cps.isspmatrix_csr(value):
        if value.dtype != dtype:
            if policy is InputPolicy.DEVICE_ONLY:
                raise InputResidencyError(f"{name} must have dtype {dtype}")
            value = value.astype(dtype)
        return value, 0
    if isinstance(value, cp.ndarray):
        if csr_required:
            raise InputResidencyError(f"{name} must be a CuPy CSR matrix")
        if value.ndim != 2:
            raise ValueError(f"{name} must be two-dimensional")
        if value.dtype != dtype:
            if policy is InputPolicy.DEVICE_ONLY:
                raise InputResidencyError(f"{name} must have dtype {dtype}")
            value = value.astype(dtype)
        return value, 0
    if policy is InputPolicy.DEVICE_ONLY:
        expected = "CuPy CSR" if csr_required else "CuPy dense or CSR"
        raise InputResidencyError(f"{name} must be a {expected} matrix")
    transferred = _host_nbytes(value)
    try:
        scipy_sparse = _module("scipy.sparse")
        is_sparse = scipy_sparse.issparse(value)
    except ImportError:
        is_sparse = False
    if is_sparse or csr_required:
        return _device_csr_from_host(value, dtype=dtype), transferred
    matrix = cp.asarray(value, dtype=dtype)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    return matrix, transferred


def _as_device_vector(
    value: Any,
    *,
    name: str,
    dtype: Any,
    policy: InputPolicy,
    optional: bool = False,
) -> tuple[Any | None, int]:
    if value is None and optional:
        return None, 0
    cp = _module("cupy")
    if isinstance(value, cp.ndarray):
        if value.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if value.dtype != dtype:
            if policy is InputPolicy.DEVICE_ONLY:
                raise InputResidencyError(f"{name} must have dtype {dtype}")
            value = value.astype(dtype)
        return value, 0
    if policy is InputPolicy.DEVICE_ONLY:
        raise InputResidencyError(f"{name} must be a CuPy device array")
    transferred = _host_nbytes(value)
    vector = cp.asarray(value, dtype=dtype)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return vector, transferred


def _validate_memberships(cluster1: Any, cluster2: Any, n_rows: int) -> None:
    cp = _module("cupy")
    memberships = (
        ("cluster1_indices", cluster1),
        ("cluster2_indices", cluster2),
    )
    for name, values in memberships:
        if values.size == 0:
            raise ValueError(f"{name} cannot be empty")
        if bool(cp.any(values < 0).item()):
            raise ValueError(f"{name} cannot contain negative indices")
        if int(cp.max(values).item()) >= n_rows:
            raise ValueError(f"{name} exceeds feature/graph alignment")
        if int(cp.unique(values).size) != int(values.size):
            raise ValueError(f"{name} must contain unique stable indices")
    if bool(cp.any(cp.isin(cluster1, cluster2)).item()):
        raise ValueError("cluster memberships must not overlap")


def _column_sums(matrix: Any) -> Any:
    cp = _module("cupy")
    result = matrix.sum(axis=0, dtype=cp.float64)
    return cp.asarray(result).ravel()


def _connectivity(graph: Any, cluster1: Any, cluster2: Any) -> float:
    cp = _module("cupy")
    forward = graph[cluster1][:, cluster2].sum(dtype=cp.float64)
    reverse = graph[cluster2][:, cluster1].sum(dtype=cp.float64)
    return float((forward + reverse).item())


def _selected_columns(
    features: Any,
    cluster1: Any,
    cluster2: Any,
    feature_mask: Any | None,
    min_reads: float | None,
) -> Any:
    cp = _module("cupy")
    columns = cp.arange(features.shape[1], dtype=cp.int64)
    if feature_mask is not None:
        columns = columns[feature_mask]
    if columns.size == 0:
        return columns
    sums1 = _column_sums(features[cluster1][:, columns])
    sums2 = _column_sums(features[cluster2][:, columns])
    if min_reads is None:
        keep = (sums1 + sums2) > cp.float64(0)
    else:
        keep = (sums1 > cluster1.size / min_reads) | (
            sums2 > cluster2.size / min_reads
        )
    return columns[keep]


def _batch_sampling_plan(
    batch_codes: Any,
    cluster1: Any,
    cluster2: Any,
    config: ComparisonConfig,
) -> tuple[Any, Any, int]:
    cp = _module("cupy")
    eligible = eligible_batch_codes(batch_codes, cluster1, cluster2)
    if eligible.size == 0:
        return eligible, cp.empty(0, dtype=cp.int32), 0
    repetitions, remainder = divmod(config.split_rounds, int(eligible.size))
    round_batches = cp.tile(eligible, repetitions)
    if remainder:
        round_batches = cp.concatenate((round_batches, eligible[:remainder]))
    elif config.rules is RuleSet.CHOIR_V0_3_0:
        # Frozen R expression ``batches[1:(rounds %% length(batches))]`` uses
        # ``1:0`` when the remainder is zero. Positive/zero R indexing ignores
        # zero but retains index one, appending the first batch. The extra code
        # is never sampled because the RF loop has only ``split_rounds`` rounds;
        # it does, however, affect the positional per-batch summary below.
        round_batches = cp.concatenate((round_batches, eligible[:1]))
    counts = []
    for code in eligible:
        counts.append(cp.sum(batch_codes[cluster1] == code))
        counts.append(cp.sum(batch_codes[cluster2] == code))
    minimum_batch_count = int(cp.min(cp.stack(counts)).item())
    rate_sample = round(
        config.downsampling_rate * min(cluster1.size, cluster2.size) / 2
    )
    per_half = max(1, min(rate_sample, minimum_batch_count // 2))
    if config.sample_max is not None:
        per_half = min(per_half, config.sample_max)
    return eligible, round_batches, int(per_half)


def _dense_sample(features: Any, rows: Any, columns: Any) -> Any:
    cp = _module("cupy")
    cps = _module("cupyx.scipy.sparse")
    sampled = features[rows][:, columns]
    if cps.issparse(sampled):
        sampled = sampled.toarray()
    return cp.ascontiguousarray(sampled, dtype=cp.float32)


def _rf_kwargs(profile: RFProfile, *, n_trees: int, seed: int) -> dict[str, Any]:
    values = profile.to_record()
    values.update(n_estimators=n_trees, random_state=seed % (2**31 - 1))
    return values


def _fit_predict(
    matrix: Any,
    labels: Any,
    train_mask: Any,
    *,
    profile: RFProfile,
    n_trees: int,
    seed: int,
) -> tuple[Any, Any, float]:
    cp = _module("cupy")
    classifier = _module("cuml.ensemble").RandomForestClassifier(
        **_rf_kwargs(profile, n_trees=n_trees, seed=seed)
    )
    test_mask = ~train_mask
    train_x = cp.ascontiguousarray(matrix[train_mask], dtype=cp.float32)
    train_y = cp.ascontiguousarray(labels[train_mask], dtype=cp.int32)
    test_x = cp.ascontiguousarray(matrix[test_mask], dtype=cp.float32)
    test_y = cp.ascontiguousarray(labels[test_mask], dtype=cp.int32)
    started = perf_counter()
    classifier.fit(train_x, train_y, convert_dtype=False)
    predicted = classifier.predict(test_x, convert_dtype=False).astype(
        cp.int32, copy=False
    )
    cp.cuda.runtime.deviceSynchronize()
    elapsed = perf_counter() - started
    accuracy = overall_accuracy(predicted, test_y)
    test_positions = cp.where(test_mask)[0]
    errors = test_positions[predicted != test_y]
    del classifier
    return accuracy, errors, elapsed


def _repeat_summary(
    error_ids: list[Any],
    *,
    cluster_size: int,
    n_iterations: int,
    n_per_half: int,
) -> tuple[Any, Any]:
    cp = _module("cupy")
    if not error_ids:
        return cp.int64(0), cp.float64(0)
    combined = cp.concatenate(error_ids)
    if combined.size == 0:
        return cp.int64(0), cp.float64(0)
    _, counts = cp.unique(combined, return_counts=True)
    recurrent = counts > max(n_iterations * 0.3, 1)
    number = cp.sum(recurrent, dtype=cp.int64)
    mean = (
        cp.sum(counts[recurrent], dtype=cp.float64)
        / cp.float64(n_iterations)
        * cp.float64(n_per_half / cluster_size)
    )
    return number, mean


def _batch_accuracy_summaries(
    observed_values: Any,
    eligible_batches: Any,
    round_batches: Any,
    rules: RuleSet,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return corrected or frozen positional per-batch accuracy summaries."""

    cp = _module("cupy")
    batch_means: dict[str, float] = {}
    batch_variances: dict[str, float] = {}
    for code in eligible_batches:
        round_positions = cp.where(round_batches == code)[0]
        if rules is RuleSet.CORRECTED:
            positions = cp.concatenate((2 * round_positions, 2 * round_positions + 1))
        else:
            # Frozen R bug: positions in ``use_batch`` index the flattened
            # accuracy vector directly rather than both complementary fits.
            positions = round_positions
        values = observed_values[positions]
        name = str(int(code.item()))
        batch_means[name] = float(cp.mean(values, dtype=cp.float64).item())
        batch_variances[name] = (
            float(cp.var(values, ddof=1, dtype=cp.float64).item())
            if values.size > 1
            else float("nan")
        )
    return batch_means, batch_variances


def _empty_samples(cluster1_size: int, cluster2_size: int) -> SampleCounts:
    return SampleCounts(
        cluster1_cells=cluster1_size,
        cluster2_cells=cluster2_size,
        per_class_per_half=0,
        split_rounds=0,
        observed_accuracies=0,
        null_accuracies=0,
        forest_fits=0,
    )


def _short_result(
    *,
    config: ComparisonConfig,
    decision: Decision,
    reason: ReasonCode,
    cluster1_id: str,
    cluster2_id: str,
    cluster1_size: int,
    cluster2_size: int,
    available_features: int,
    selected_features: int,
    connectivity: float | None,
    operation_seed: int,
    started: float,
    components: dict[str, float],
    ledger: TransferLedger,
    free_bytes: int,
    total_bytes: int,
) -> ComparisonResult:
    elapsed = perf_counter() - started
    return ComparisonResult(
        decision=decision,
        reason_code=reason,
        rules=RuleSet(config.rules),
        cluster1_id=cluster1_id,
        cluster2_id=cluster2_id,
        connectivity=connectivity,
        metrics=None,
        samples=_empty_samples(cluster1_size, cluster2_size),
        features=FeatureCounts(
            available=available_features,
            selected=selected_features,
        ),
        seeds=SeedRecord(root=config.random_state, operation=operation_seed),
        timings=TimingRecord(total_seconds=elapsed, components_seconds=components),
        memory=MemoryRecord(
            estimated_bytes=0,
            free_bytes_at_preflight=free_bytes,
            total_device_bytes=total_bytes,
            required_headroom_bytes=0,
            peak_device_bytes=0,
            transfers=ledger.freeze(),
        ),
        compatibility_aliases={"decision": REFERENCE_REASON_ALIASES[reason]},
    )


class ChoirComparator:
    """Representation-independent single-GPU comparator."""

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
        started = perf_counter()
        config = self.config
        context = initialize_cuda(config.device)
        cp = context.cupy
        policy = InputPolicy(config.input_policy)
        ledger = TransferLedger()
        ingest_started = perf_counter()
        features, feature_bytes = _as_device_matrix(
            features,
            name="features",
            dtype=cp.float32,
            csr_required=False,
            policy=policy,
        )
        graph, graph_bytes = _as_device_matrix(
            nn_graph,
            name="nn_graph",
            dtype=cp.float32,
            csr_required=True,
            policy=policy,
        )
        cluster1, cluster1_bytes = _as_device_vector(
            cluster1_indices,
            name="cluster1_indices",
            dtype=cp.int64,
            policy=policy,
        )
        cluster2, cluster2_bytes = _as_device_vector(
            cluster2_indices,
            name="cluster2_indices",
            dtype=cp.int64,
            policy=policy,
        )
        mask, mask_bytes = _as_device_vector(
            feature_mask,
            name="feature_mask",
            dtype=cp.bool_,
            policy=policy,
            optional=True,
        )
        batches, batch_bytes = _as_device_vector(
            batch_codes,
            name="batch_codes",
            dtype=cp.int32,
            policy=policy,
            optional=True,
        )
        assert cluster1 is not None and cluster2 is not None
        transferred = sum(
            (
                feature_bytes,
                graph_bytes,
                cluster1_bytes,
                cluster2_bytes,
                mask_bytes,
                batch_bytes,
            )
        )
        if transferred:
            ledger.record(
                direction="host_to_device",
                bytes=transferred,
                purpose="initial comparison ingest",
            )
        ledger.finish_ingest()
        components: dict[str, float] = {"ingest": perf_counter() - ingest_started}

        if features.shape[0] != graph.shape[0] or graph.shape[0] != graph.shape[1]:
            raise GraphSemanticError("NN graph must be square and aligned to features")
        if mask is not None and mask.size != features.shape[1]:
            raise ValueError("feature_mask must align to feature columns")
        if batches is not None and batches.size != features.shape[0]:
            raise ValueError("batch_codes must align to feature rows")
        _validate_memberships(cluster1, cluster2, features.shape[0])

        fingerprint_started = perf_counter()
        fingerprint1 = device_membership_fingerprint(cluster1)
        fingerprint2 = device_membership_fingerprint(cluster2)
        canonical_fingerprints = tuple(sorted((fingerprint1, fingerprint2)))
        operation_seed = derive_seed(
            config.random_state,
            f"{config.rules}/comparison",
            *canonical_fingerprints,
        )
        components["fingerprint"] = perf_counter() - fingerprint_started
        ledger.record(
            direction="device_to_host",
            bytes=32,
            purpose="membership fingerprints and validation scalars",
        )

        free_initial, total_bytes = context.memory_info()
        available_features = int(features.shape[1])
        cluster1_size = int(cluster1.size)
        cluster2_size = int(cluster2.size)

        connectivity: float | None = None
        connectivity_started = perf_counter()
        if config.min_connections > 0:
            connectivity = _connectivity(graph, cluster1, cluster2)
            ledger.record(
                direction="device_to_host",
                bytes=8,
                purpose="NN connectivity scalar",
            )
            if connectivity < config.min_connections:
                components["connectivity"] = perf_counter() - connectivity_started
                result = _short_result(
                    config=config,
                    decision=Decision.SPLIT,
                    reason=ReasonCode.SPLIT_MIN_CONNECTIONS,
                    cluster1_id=cluster1_id,
                    cluster2_id=cluster2_id,
                    cluster1_size=cluster1_size,
                    cluster2_size=cluster2_size,
                    available_features=available_features,
                    selected_features=0,
                    connectivity=connectivity,
                    operation_seed=operation_seed,
                    started=started,
                    components=components,
                    ledger=ledger,
                    free_bytes=free_initial,
                    total_bytes=total_bytes,
                )
                return result
        components["connectivity"] = perf_counter() - connectivity_started

        batch_started = perf_counter()
        eligible_batches = None
        round_batches = None
        if batches is not None:
            eligible_batches, round_batches, n_per_half = _batch_sampling_plan(
                batches, cluster1, cluster2, config
            )
            ledger.record(
                direction="device_to_host",
                bytes=16,
                purpose="batch eligibility scalars",
            )
            if eligible_batches.size == 0:
                components["batch_filter"] = perf_counter() - batch_started
                result = _short_result(
                    config=config,
                    decision=Decision.MERGE,
                    reason=ReasonCode.MERGE_BATCH_DEPENDENT,
                    cluster1_id=cluster1_id,
                    cluster2_id=cluster2_id,
                    cluster1_size=cluster1_size,
                    cluster2_size=cluster2_size,
                    available_features=available_features,
                    selected_features=0,
                    connectivity=connectivity,
                    operation_seed=operation_seed,
                    started=started,
                    components=components,
                    ledger=ledger,
                    free_bytes=free_initial,
                    total_bytes=total_bytes,
                )
                return result
        else:
            n_per_half = balanced_sample_size(
                cluster1_size,
                cluster2_size,
                downsampling_rate=config.downsampling_rate,
                sample_max=config.sample_max,
            )
        components["batch_filter"] = perf_counter() - batch_started

        feature_started = perf_counter()
        columns = _selected_columns(
            features, cluster1, cluster2, mask, config.min_reads
        )
        selected_features = int(columns.size)
        ledger.record(
            direction="device_to_host",
            bytes=8,
            purpose="selected feature count",
        )
        if selected_features == 0:
            components["feature_filter"] = perf_counter() - feature_started
            result = _short_result(
                config=config,
                decision=Decision.MERGE,
                reason=ReasonCode.MERGE_MINIMUM_READS,
                cluster1_id=cluster1_id,
                cluster2_id=cluster2_id,
                cluster1_size=cluster1_size,
                cluster2_size=cluster2_size,
                available_features=available_features,
                selected_features=0,
                connectivity=connectivity,
                operation_seed=operation_seed,
                started=started,
                components=components,
                ledger=ledger,
                free_bytes=free_initial,
                total_bytes=total_bytes,
            )
            return result
        components["feature_filter"] = perf_counter() - feature_started

        if config.split_rounds < 1:
            raise ValueError("n_iterations produces no RF split rounds")
        preflight_started = perf_counter()
        estimated_bytes = estimate_comparison_bytes(
            sampled_rows=4 * n_per_half,
            selected_features=selected_features,
            n_trees=config.n_trees,
            max_depth=config.resolved_rf_profile.max_depth,
            n_iterations=config.n_iterations,
        )
        free_preflight, total_bytes = context.memory_info()
        required_bytes = preflight_device_memory(
            estimated_bytes=estimated_bytes,
            free_bytes=free_preflight,
            headroom_fraction=config.vram_headroom,
        )
        components["memory_preflight"] = perf_counter() - preflight_started
        minimum_free = free_preflight

        execution_swapped = fingerprint2 < fingerprint1
        if execution_swapped:
            execution1, execution2 = cluster2, cluster1
        else:
            execution1, execution2 = cluster1, cluster2

        observed: list[Any] = []
        null: list[Any] = []
        errors1: list[Any] = []
        errors2: list[Any] = []
        round_seeds: list[int] = []
        fit_seeds: list[int] = []
        sampling_seconds = 0.0
        densify_seconds = 0.0
        observed_fit_seconds = 0.0
        null_fit_seconds = 0.0

        for round_index in range(config.split_rounds):
            round_started = perf_counter()
            if batches is not None:
                assert round_batches is not None
                code = round_batches[round_index]
                current1 = execution1[batches[execution1] == code]
                current2 = execution2[batches[execution2] == code]
            else:
                current1, current2 = execution1, execution2
            round_seed = derive_seed(
                config.random_state,
                f"{config.rules}/comparison/round",
                *canonical_fingerprints,
                round_index,
            )
            split = balanced_split_round(
                current1,
                current2,
                n_per_half=n_per_half,
                seed=round_seed,
                rules=config.rules,
            )
            round_seeds.append(round_seed)
            sampling_seconds += perf_counter() - round_started

            dense_started = perf_counter()
            sampled = _dense_sample(features, split.row_indices, columns)
            densify_seconds += perf_counter() - dense_started

            roles = (
                (split.true_labels, split.first_train_mask, "observed-a", observed),
                (split.true_labels, split.second_train_mask, "observed-b", observed),
                (split.permuted_labels, split.first_train_mask, "null-a", null),
                (split.permuted_labels, split.second_train_mask, "null-b", null),
            )
            sampled_per_class = 2 * n_per_half
            for labels, train_mask, role, destination in roles:
                seed = derive_seed(round_seed, f"{config.rules}/fit", role)
                fit_seeds.append(seed)
                accuracy, error_positions, elapsed = _fit_predict(
                    sampled,
                    labels,
                    train_mask,
                    profile=config.resolved_rf_profile,
                    n_trees=config.n_trees,
                    seed=seed,
                )
                destination.append(accuracy)
                if role.startswith("observed") and config.max_repeat_errors > 0:
                    first_errors = error_positions[error_positions < sampled_per_class]
                    second_errors = error_positions[
                        error_positions >= sampled_per_class
                    ]
                    if config.rules is RuleSet.CORRECTED:
                        errors1.append(split.row_indices[first_errors])
                        errors2.append(split.row_indices[second_errors])
                    else:
                        errors1.append(first_errors)
                        errors2.append(second_errors - sampled_per_class)
                if role.startswith("observed"):
                    observed_fit_seconds += elapsed
                else:
                    null_fit_seconds += elapsed
                free_now, _ = context.memory_info()
                minimum_free = min(minimum_free, free_now)
            del sampled

        observed_values = cp.stack(observed).astype(cp.float64, copy=False)
        null_values = cp.stack(null).astype(cp.float64, copy=False)
        statistics_started = perf_counter()
        bootstrap_seed = derive_seed(
            operation_seed, f"{config.rules}/bootstrap", "null-variance"
        )
        bootstrap_count = (
            config.n_iterations
            if config.rules is RuleSet.CHOIR_V0_3_0
            else int(null_values.size)
        )
        accuracy_tail, variance_tail, _ = tail_probabilities(
            observed_values,
            null_values,
            bootstrap_seed=bootstrap_seed,
            n_bootstrap=bootstrap_count,
            bootstrap_sample_size=bootstrap_count,
        )
        observed_mean = cp.mean(observed_values, dtype=cp.float64)
        observed_variance = cp.var(observed_values, ddof=1, dtype=cp.float64)
        null_mean = cp.mean(null_values, dtype=cp.float64)
        null_variance = cp.var(null_values, ddof=1, dtype=cp.float64)

        repeated1 = cp.int64(0)
        repeated2 = cp.int64(0)
        modified_mean = None
        modified_variance = None
        modified_accuracy_tail = None
        modified_variance_tail = None
        if config.max_repeat_errors > 0:
            repeated1, mean_repeated1 = _repeat_summary(
                errors1,
                cluster_size=int(execution1.size),
                n_iterations=config.n_iterations,
                n_per_half=n_per_half,
            )
            repeated2, mean_repeated2 = _repeat_summary(
                errors2,
                cluster_size=int(execution2.size),
                n_iterations=config.n_iterations,
                n_per_half=n_per_half,
            )
            mean_error = (cp.float64(1) - observed_mean) * n_per_half
            modified_mean = cp.minimum(
                cp.float64(1),
                (
                    n_per_half
                    - (mean_error - (mean_repeated1 + mean_repeated2) / 2)
                )
                / n_per_half,
            )
            modified_values = observed_values + (modified_mean - observed_mean)
            modified_variance = cp.var(
                modified_values, ddof=1, dtype=cp.float64
            )
            modified_accuracy_tail = cp.float64(1) - _module(
                "rapids_choir._stats"
            ).r_density_cdf(null_values, modified_mean)
            modified_variance_tail = _module("rapids_choir._stats").r_density_cdf(
                _module("rapids_choir._stats").bootstrap_variances(
                    null_values,
                    n_bootstrap=bootstrap_count,
                    sample_size=bootstrap_count,
                    seed=bootstrap_seed,
                ),
                modified_variance,
            )

        batch_means: dict[str, float] = {}
        batch_variances: dict[str, float] = {}
        if eligible_batches is not None and round_batches is not None:
            batch_means, batch_variances = _batch_accuracy_summaries(
                observed_values,
                eligible_batches,
                round_batches,
                RuleSet(config.rules),
            )

        scalar_values = [
            observed_mean,
            observed_variance,
            null_mean,
            null_variance,
            accuracy_tail,
            variance_tail,
        ]
        host_values = [float(value.item()) for value in scalar_values]
        optional_values = [
            None if modified_mean is None else float(modified_mean.item()),
            None if modified_variance is None else float(modified_variance.item()),
            None
            if modified_accuracy_tail is None
            else float(modified_accuracy_tail.item()),
            None
            if modified_variance_tail is None
            else float(modified_variance_tail.item()),
        ]
        repeated1_host = int(repeated1.item())
        repeated2_host = int(repeated2.item())
        if execution_swapped:
            repeated1_host, repeated2_host = repeated2_host, repeated1_host
        ledger.record(
            direction="device_to_host",
            bytes=8 * (12 + 2 * len(batch_means)),
            purpose="comparison result scalars",
        )
        components["statistics"] = perf_counter() - statistics_started
        components["sampling"] = sampling_seconds
        components["sample_densification"] = densify_seconds
        components["observed_fits"] = observed_fit_seconds
        components["null_fits"] = null_fit_seconds

        metrics = ComparisonMetrics(
            mean_observed_accuracy=host_values[0],
            variance_observed_accuracy=host_values[1],
            mean_null_accuracy=host_values[2],
            variance_null_accuracy=host_values[3],
            accuracy_reference_tail_probability=host_values[4],
            variance_reference_tail_probability=host_values[5],
            mean_modified_accuracy=optional_values[0],
            variance_modified_accuracy=optional_values[1],
            modified_accuracy_reference_tail_probability=optional_values[2],
            modified_variance_reference_tail_probability=optional_values[3],
            repeated_error_cells_cluster1=repeated1_host,
            repeated_error_cells_cluster2=repeated2_host,
            batch_mean_accuracies=batch_means,
            batch_variance_accuracies=batch_variances,
        )
        outcome = decide_comparison(
            metrics,
            config,
            cluster1_size=cluster1_size,
            cluster2_size=cluster2_size,
        )
        samples = SampleCounts(
            cluster1_cells=cluster1_size,
            cluster2_cells=cluster2_size,
            per_class_per_half=n_per_half,
            split_rounds=config.split_rounds,
            observed_accuracies=int(observed_values.size),
            null_accuracies=int(null_values.size),
            forest_fits=4 * config.split_rounds,
        )
        aliases = {
            "decision": REFERENCE_REASON_ALIASES[outcome.reason_code],
            "mean_accuracy": metrics.mean_observed_accuracy,
            "var_accuracy": metrics.variance_observed_accuracy,
            "mean_permuted_accuracy": metrics.mean_null_accuracy,
            "var_permuted_accuracy": metrics.variance_null_accuracy,
            "percentile_accuracy": metrics.accuracy_reference_tail_probability,
            "percentile_variance": metrics.variance_reference_tail_probability,
            "max_p": outcome.maximum_tail_score,
        }
        result = ComparisonResult(
            decision=outcome.decision,
            reason_code=outcome.reason_code,
            rules=RuleSet(config.rules),
            cluster1_id=cluster1_id,
            cluster2_id=cluster2_id,
            connectivity=connectivity,
            metrics=metrics,
            samples=samples,
            features=FeatureCounts(
                available=available_features,
                selected=selected_features,
            ),
            seeds=SeedRecord(
                root=config.random_state,
                operation=operation_seed,
                round_seeds=tuple(round_seeds),
                fit_seeds=tuple(fit_seeds),
            ),
            timings=TimingRecord(
                total_seconds=perf_counter() - started,
                components_seconds=components,
            ),
            memory=MemoryRecord(
                estimated_bytes=estimated_bytes,
                free_bytes_at_preflight=free_preflight,
                total_device_bytes=total_bytes,
                required_headroom_bytes=required_bytes - estimated_bytes,
                peak_device_bytes=free_preflight - minimum_free,
                transfers=ledger.freeze(),
            ),
            compatibility_aliases=aliases,
        )
        return result
