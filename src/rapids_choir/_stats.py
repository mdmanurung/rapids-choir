"""GPU float64 statistics used by the comparator."""

from __future__ import annotations

import importlib
from math import pi
from typing import Any

from .exceptions import GPUUnavailableError, InputResidencyError


def _cupy() -> Any:
    try:
        return importlib.import_module("cupy")
    except Exception as error:
        raise GPUUnavailableError(f"could not import CuPy: {error}") from error


def _float64_vector(values: Any, *, name: str) -> Any:
    cp = _cupy()
    if not isinstance(values, cp.ndarray):
        raise InputResidencyError(f"{name} must be a CuPy device array")
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array with >=2 values")
    if not bool(cp.all(cp.isfinite(values)).item()):
        raise ValueError(f"{name} must contain only finite values")
    return values.astype(cp.float64, copy=False)


def overall_accuracy(predicted: Any, expected: Any) -> Any:
    cp = _cupy()
    if not isinstance(predicted, cp.ndarray) or not isinstance(expected, cp.ndarray):
        raise InputResidencyError("predicted and expected must be CuPy arrays")
    if predicted.shape != expected.shape or predicted.ndim != 1:
        raise ValueError("predicted and expected must be aligned vectors")
    return cp.mean(predicted == expected, dtype=cp.float64)


def r_bw_nrd0(values: Any) -> Any:
    """GPU implementation of R's ``stats::bw.nrd0``."""

    cp = _cupy()
    values = _float64_vector(values, name="values")
    standard_deviation = cp.std(values, ddof=1, dtype=cp.float64)
    quartiles = cp.quantile(values, cp.asarray([0.25, 0.75], dtype=cp.float64))
    robust_scale = (quartiles[1] - quartiles[0]) / cp.float64(1.34)
    scale = cp.minimum(standard_deviation, robust_scale)
    if float(scale.item()) == 0.0:
        scale = standard_deviation
    if float(scale.item()) == 0.0:
        scale = cp.abs(values[0])
    if float(scale.item()) == 0.0:
        scale = cp.float64(1.0)
    return cp.float64(0.9) * scale * cp.float64(values.size) ** cp.float64(-0.2)


def r_density_cdf(values: Any, query: Any, *, n: int = 512) -> Any:
    """Match ``CDF(stats::density(...))`` with GPU float64 arithmetic.

    R's density estimate uses an FFT-linear-binned Gaussian KDE. Evaluating the
    same Gaussian mixture directly on the output grid avoids binning error; the
    spatstat CDF normalization, left-rectangle integration, and interpolation
    are reproduced exactly enough for the frozen 1e-3 tolerance.
    """

    cp = _cupy()
    values = _float64_vector(values, name="values")
    if n < 512:
        raise ValueError("R density uses at least 512 grid points")
    bandwidth = r_bw_nrd0(values)
    start = cp.min(values) - cp.float64(3.0) * bandwidth
    stop = cp.max(values) + cp.float64(3.0) * bandwidth
    grid = cp.linspace(start, stop, n, dtype=cp.float64)
    standardized = (grid[:, None] - values[None, :]) / bandwidth
    density = cp.mean(cp.exp(-cp.float64(0.5) * standardized**2), axis=1)
    density /= bandwidth * cp.sqrt(cp.float64(2.0 * pi))
    increments = cp.concatenate((cp.zeros(1, dtype=cp.float64), cp.diff(grid, axis=0)))
    cumulative = cp.cumsum(density * increments, dtype=cp.float64)
    cumulative /= cumulative[-1]
    query_value = cp.asarray(query, dtype=cp.float64)
    return cp.interp(query_value, grid, cumulative)


def bootstrap_variances(
    values: Any,
    *,
    n_bootstrap: int | None = None,
    sample_size: int | None = None,
    seed: int | None = None,
    indices: Any | None = None,
) -> Any:
    """Bootstrap sample variance on-device, optionally from frozen indices."""

    cp = _cupy()
    values = _float64_vector(values, name="values")
    if indices is None:
        if n_bootstrap is None or n_bootstrap < 1:
            raise ValueError("n_bootstrap must be positive when indices are absent")
        if seed is None:
            raise ValueError("seed must be explicit when indices are absent")
        if sample_size is None:
            sample_size = int(values.size)
        if sample_size < 2:
            raise ValueError("bootstrap sample_size must be >=2")
        generator = cp.random.RandomState(seed)
        indices = generator.randint(
            0, values.size, size=(n_bootstrap, sample_size), dtype=cp.int64
        )
    else:
        if not isinstance(indices, cp.ndarray):
            raise InputResidencyError("indices must be a CuPy device array")
        if indices.ndim != 2:
            raise ValueError("indices must be a two-dimensional array")
        if indices.dtype.kind not in {"i", "u"}:
            raise TypeError("bootstrap indices must have integer dtype")
        if bool(cp.any(indices < 0).item()) or bool(
            cp.any(indices >= values.size).item()
        ):
            raise ValueError("bootstrap indices are out of bounds")
    return cp.var(values[indices], axis=1, ddof=1, dtype=cp.float64)


def tail_probabilities(
    observed_accuracies: Any,
    null_accuracies: Any,
    *,
    bootstrap_seed: int,
    bootstrap_indices: Any | None = None,
    n_bootstrap: int | None = None,
    bootstrap_sample_size: int | None = None,
) -> tuple[Any, Any, Any]:
    cp = _cupy()
    observed = _float64_vector(observed_accuracies, name="observed_accuracies")
    null = _float64_vector(null_accuracies, name="null_accuracies")
    observed_mean = cp.mean(observed, dtype=cp.float64)
    observed_variance = cp.var(observed, ddof=1, dtype=cp.float64)
    bootstrapped = bootstrap_variances(
        null,
        n_bootstrap=int(null.size) if n_bootstrap is None else n_bootstrap,
        sample_size=bootstrap_sample_size,
        seed=bootstrap_seed,
        indices=bootstrap_indices,
    )
    accuracy_tail = cp.float64(1.0) - r_density_cdf(null, observed_mean)
    variance_tail = r_density_cdf(bootstrapped, observed_variance)
    return accuracy_tail, variance_tail, bootstrapped


def bonferroni_threshold(alpha: float, number_of_scores: int) -> float:
    if number_of_scores < 1:
        raise ValueError("number_of_scores must be positive")
    return alpha / number_of_scores


def benjamini_hochberg(values: Any) -> Any:
    cp = _cupy()
    values = _float64_vector(values, name="values")
    if bool(cp.any((values < 0) | (values > 1)).item()):
        raise ValueError("tail scores must lie in [0, 1]")
    order = cp.argsort(values, kind="stable")
    ranked = values[order]
    ranks = cp.arange(1, values.size + 1, dtype=cp.float64)
    adjusted_ranked = ranked * cp.float64(values.size) / ranks
    # CuPy 14.1 exposes ``minimum.accumulate`` but does not implement it.  A
    # logarithmic suffix scan keeps the BH adjustment entirely on-device.
    step = 1
    while step < values.size:
        previous = adjusted_ranked
        adjusted_ranked = previous.copy()
        adjusted_ranked[:-step] = cp.minimum(
            previous[:-step], previous[step:]
        )
        step *= 2
    adjusted_ranked = cp.minimum(adjusted_ranked, cp.float64(1.0))
    adjusted = cp.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def bh_dynamic_threshold(
    accuracy_scores: Any,
    variance_scores: Any,
    *,
    alpha: float,
    previous_threshold: float,
    rules: str,
) -> float:
    cp = _cupy()
    accuracy = _float64_vector(accuracy_scores, name="accuracy_scores")
    variance = _float64_vector(variance_scores, name="variance_scores")
    if accuracy.size != variance.size:
        raise ValueError("accuracy and variance score arrays must align")

    def candidate(scores: Any) -> float:
        adjusted = benjamini_hochberg(scores)
        passing = scores[adjusted < alpha]
        if passing.size == 0:
            return previous_threshold
        largest = cp.max(passing)
        sorted_scores = cp.sort(scores)
        # R max(which(sort(scores) == max(passing))) uses the last tied rank.
        last_rank = cp.max(cp.where(sorted_scores == largest)[0]) + 1
        return float((last_rank / scores.size * alpha).item())

    accuracy_candidate = candidate(accuracy)
    variance_candidate = candidate(variance)
    if rules == "choir_v0_3_0":
        combined = variance_candidate
    elif rules == "corrected":
        combined = max(accuracy_candidate, variance_candidate)
    else:
        raise ValueError(f"unsupported rules: {rules}")
    return min(previous_threshold, combined)
