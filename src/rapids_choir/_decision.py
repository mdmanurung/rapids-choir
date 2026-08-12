"""Pure decision rules shared by runtime and exact logic tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ._enums import Decision, ReasonCode
from .config import ComparisonConfig
from .results import ComparisonMetrics


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    decision: Decision
    reason_code: ReasonCode
    maximum_tail_score: float


def _tail_pass(value: float | None, threshold: float) -> bool:
    return value is not None and isfinite(value) and value < threshold


def _tail_value(value: float | None) -> float:
    """Return a finite score or the conservative upper bound."""

    return value if value is not None and isfinite(value) else 1.0


def decide_comparison(
    metrics: ComparisonMetrics,
    config: ComparisonConfig,
    *,
    cluster1_size: int,
    cluster2_size: int,
) -> DecisionOutcome:
    accuracy_pass = _tail_pass(
        metrics.accuracy_reference_tail_probability, config.alpha
    )
    variance_pass = not config.use_variance or _tail_pass(
        metrics.variance_reference_tail_probability, config.alpha
    )
    if (
        metrics.mean_observed_accuracy >= config.min_accuracy
        and accuracy_pass
        and variance_pass
    ):
        maximum = metrics.accuracy_reference_tail_probability
        if config.use_variance:
            maximum = max(maximum, metrics.variance_reference_tail_probability)
        return DecisionOutcome(Decision.SPLIT, ReasonCode.SPLIT, maximum)

    if config.max_repeat_errors > 0:
        modified_accuracy_pass = _tail_pass(
            metrics.modified_accuracy_reference_tail_probability, config.alpha
        )
        modified_variance_pass = not config.use_variance or _tail_pass(
            metrics.modified_variance_reference_tail_probability, config.alpha
        )
        repeated_error_limits_pass = metrics.repeated_error_cells_cluster1 <= min(
            config.max_repeat_errors, 0.3 * cluster1_size
        ) and metrics.repeated_error_cells_cluster2 <= min(
            config.max_repeat_errors, 0.3 * cluster2_size
        )
        if (
            metrics.mean_observed_accuracy >= config.min_accuracy
            and _tail_pass(
                metrics.accuracy_reference_tail_probability, 2 * config.alpha
            )
            and (
                not config.use_variance
                or _tail_pass(
                    metrics.variance_reference_tail_probability, 2 * config.alpha
                )
            )
            and modified_accuracy_pass
            and modified_variance_pass
            and repeated_error_limits_pass
        ):
            accuracy_max = min(
                metrics.accuracy_reference_tail_probability,
                _tail_value(metrics.modified_accuracy_reference_tail_probability),
            )
            maximum = accuracy_max
            if config.use_variance:
                variance_max = min(
                    metrics.variance_reference_tail_probability,
                    _tail_value(metrics.modified_variance_reference_tail_probability),
                )
                maximum = max(accuracy_max, variance_max)
            return DecisionOutcome(
                Decision.SPLIT, ReasonCode.SPLIT_REPEAT_ERROR, maximum
            )

        if (
            metrics.mean_observed_accuracy < config.min_accuracy
            and accuracy_pass
            and variance_pass
        ):
            maximum = metrics.accuracy_reference_tail_probability
            if config.use_variance:
                maximum = max(maximum, metrics.variance_reference_tail_probability)
            return DecisionOutcome(
                Decision.MERGE, ReasonCode.MERGE_MIN_ACCURACY, maximum
            )

    return DecisionOutcome(Decision.MERGE, ReasonCode.MERGE, 1.0)
