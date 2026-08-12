"""Device-side balanced sampling and label permutation."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from ._enums import RuleSet
from .exceptions import GPUUnavailableError, InputResidencyError


@dataclass(frozen=True, slots=True)
class SplitRound:
    """One balanced sample supporting two complementary observed/null fits."""

    row_indices: Any
    true_labels: Any
    first_train_mask: Any
    permuted_labels: Any
    seed: int

    @property
    def second_train_mask(self) -> Any:
        return ~self.first_train_mask


def _cupy() -> Any:
    try:
        return importlib.import_module("cupy")
    except Exception as error:
        raise GPUUnavailableError(f"could not import CuPy: {error}") from error


def _index_vector(values: Any, *, name: str) -> Any:
    cp = _cupy()
    if not isinstance(values, cp.ndarray):
        raise InputResidencyError(f"{name} must be a CuPy device array")
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must have integer dtype")
    if bool(cp.any(values < 0).item()):
        raise ValueError(f"{name} cannot contain negative indices")
    if int(cp.unique(values).size) != int(values.size):
        raise ValueError(f"{name} must contain unique stable cell indices")
    return values


def _batch_vector(values: Any) -> Any:
    cp = _cupy()
    if not isinstance(values, cp.ndarray):
        raise InputResidencyError("batch_codes must be a CuPy device array")
    if values.ndim != 1:
        raise ValueError("batch_codes must be one-dimensional")
    if values.dtype.kind not in {"i", "u"}:
        raise TypeError("batch_codes must have integer dtype")
    if bool(cp.any(values < 0).item()):
        raise ValueError("batch_codes cannot contain negative values")
    return values


def balanced_sample_size(
    cluster1_size: int,
    cluster2_size: int,
    *,
    downsampling_rate: float = 1.0,
    sample_max: int | None = None,
) -> int:
    """Return the per-class size of each complementary train/test half."""

    if cluster1_size < 1 or cluster2_size < 1:
        raise ValueError("cluster sizes must be positive")
    if downsampling_rate <= 0:
        raise ValueError("downsampling_rate must be positive")
    if sample_max is not None and sample_max < 1:
        raise ValueError("sample_max must be positive when supplied")
    sampled = max(1, int(downsampling_rate * min(cluster1_size, cluster2_size) // 2))
    return min(sampled, sample_max) if sample_max is not None else sampled


def eligible_batch_codes(
    batch_codes: Any,
    cluster1_indices: Any,
    cluster2_indices: Any,
) -> Any:
    """Return sorted batch identities eligible in both clusters on-device."""

    cp = _cupy()
    batches = _batch_vector(batch_codes)
    cluster1 = _index_vector(cluster1_indices, name="cluster1_indices")
    cluster2 = _index_vector(cluster2_indices, name="cluster2_indices")
    if batches.size == 0:
        raise ValueError("batch_codes cannot be empty")
    if int(cp.max(cp.concatenate((cluster1, cluster2))).item()) >= int(batches.size):
        raise ValueError("cluster index exceeds batch_codes alignment")

    def eligible(indices: Any) -> Any:
        identities, counts = cp.unique(batches[indices], return_counts=True)
        threshold = cp.maximum(cp.int64(2), cp.float64(0.20) * cp.max(counts))
        return identities[counts >= threshold]

    return cp.intersect1d(eligible(cluster1), eligible(cluster2), assume_unique=True)


def balanced_split_round(
    cluster1_indices: Any,
    cluster2_indices: Any,
    *,
    n_per_half: int,
    seed: int,
    rules: RuleSet | str = RuleSet.CORRECTED,
) -> SplitRound:
    """Sample cells and construct complementary masks plus permuted labels.

    Compatibility mode preserves CHOIR's cluster-2-before-cluster-1 random call
    ordering. The random generator is still CuPy rather than R and is therefore
    recorded as compatibility behavior, not numerical RNG identity.
    """

    cp = _cupy()
    cluster1 = _index_vector(cluster1_indices, name="cluster1_indices")
    cluster2 = _index_vector(cluster2_indices, name="cluster2_indices")
    if bool(cp.any(cp.isin(cluster1, cluster2)).item()):
        raise ValueError("cluster memberships must not overlap")
    if isinstance(n_per_half, bool) or not isinstance(n_per_half, int):
        raise TypeError("n_per_half must be an integer")
    if n_per_half < 1:
        raise ValueError("n_per_half must be positive")
    sampled_per_class = 2 * n_per_half
    if cluster1.size < sampled_per_class or cluster2.size < sampled_per_class:
        raise ValueError("each cluster needs at least 2 * n_per_half cells")
    try:
        rule_set = RuleSet(rules)
    except ValueError as error:
        raise ValueError(f"unsupported rules: {rules}") from error

    generator = cp.random.RandomState(seed)
    if rule_set is RuleSet.CHOIR_V0_3_0:
        sampled2 = generator.permutation(cluster2)[:sampled_per_class]
        sampled1 = generator.permutation(cluster1)[:sampled_per_class]
    else:
        sampled1 = generator.permutation(cluster1)[:sampled_per_class]
        sampled2 = generator.permutation(cluster2)[:sampled_per_class]

    base_mask = cp.concatenate(
        (
            cp.ones(n_per_half, dtype=cp.bool_),
            cp.zeros(n_per_half, dtype=cp.bool_),
        )
    )
    first_train_mask = cp.concatenate(
        (generator.permutation(base_mask), generator.permutation(base_mask))
    )
    labels = cp.concatenate(
        (
            cp.zeros(sampled_per_class, dtype=cp.int32),
            cp.ones(sampled_per_class, dtype=cp.int32),
        )
    )
    permuted_labels = generator.permutation(labels).astype(cp.int32, copy=False)
    return SplitRound(
        row_indices=cp.concatenate((sampled1, sampled2)),
        true_labels=labels,
        first_train_mask=first_train_mask,
        permuted_labels=permuted_labels,
        seed=seed,
    )
