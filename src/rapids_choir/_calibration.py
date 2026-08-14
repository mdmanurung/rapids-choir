"""Pure calibration ranking and Gate 2 acceptance arithmetic."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from statistics import NormalDist
from typing import Any

import numpy as np


def wilson_lower_bound(
    successes: int,
    total: int,
    *,
    confidence: float = 0.95,
) -> float:
    """Return the two-sided Wilson interval lower bound."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise TypeError("successes and total must be integers")
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    centre = proportion + z2 / (2 * total)
    radius = z * sqrt(
        proportion * (1 - proportion) / total + z2 / (4 * total * total)
    )
    return (centre - radius) / denominator


def majority_decision(decisions: Sequence[str]) -> str:
    """Return a strict majority from an odd number of merge/split decisions."""

    if len(decisions) == 0 or len(decisions) % 2 == 0:
        raise ValueError("majority aggregation requires an odd number of decisions")
    invalid = set(decisions) - {"merge", "split"}
    if invalid:
        raise ValueError(f"unsupported decisions: {sorted(invalid)}")
    split_count = sum(value == "split" for value in decisions)
    return "split" if split_count > len(decisions) // 2 else "merge"


def macro_decision_concordance(
    reference: Sequence[str], predicted: Sequence[str]
) -> float:
    """Macro-average concordance across reference merge/split classes."""

    if len(reference) != len(predicted) or not reference:
        raise ValueError(
            "reference and predicted decisions must align and be non-empty"
        )
    classes = sorted(set(reference))
    if set(classes) - {"merge", "split"} or set(predicted) - {"merge", "split"}:
        raise ValueError("decisions must be merge or split")
    class_scores = []
    for decision in classes:
        positions = [
            index for index, value in enumerate(reference) if value == decision
        ]
        class_scores.append(
            sum(predicted[index] == decision for index in positions) / len(positions)
        )
    return sum(class_scores) / len(class_scores)


def macro_decision_reason_concordance(
    reference_decisions: Sequence[str],
    predicted_decisions: Sequence[str],
    reference_reasons: Sequence[str],
    predicted_reasons: Sequence[str],
) -> float:
    """Macro concordance across joint reference decision/reason strata."""

    lengths = {
        len(reference_decisions),
        len(predicted_decisions),
        len(reference_reasons),
        len(predicted_reasons),
    }
    if len(lengths) != 1 or not reference_decisions:
        raise ValueError("decision/reason records must align and be non-empty")
    reference = list(zip(reference_decisions, reference_reasons, strict=True))
    predicted = list(zip(predicted_decisions, predicted_reasons, strict=True))
    strata = sorted(set(reference))
    scores = []
    for stratum in strata:
        positions = [
            index for index, value in enumerate(reference) if value == stratum
        ]
        scores.append(
            sum(predicted[index] == stratum for index in positions) / len(positions)
        )
    return sum(scores) / len(scores)


def accuracy_distribution_error(
    reference: Sequence[float],
    predicted: Sequence[float],
) -> float:
    """Mean absolute difference over a frozen 101-point quantile grid."""

    left = np.asarray(reference, dtype=float)
    right = np.asarray(predicted, dtype=float)
    if left.size == 0 or right.size == 0:
        raise ValueError("accuracy distributions must be non-empty")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any((left < 0) | (left > 1))
        or np.any((right < 0) | (right > 1))
    ):
        raise ValueError("accuracy distributions must be finite values in [0, 1]")
    quantiles = np.linspace(0.0, 1.0, 101)
    return float(
        np.mean(
            np.abs(
                np.quantile(left, quantiles, method="linear")
                - np.quantile(right, quantiles, method="linear")
            )
        )
    )


def bounded_metric_margin(
    absolute_differences: Sequence[float],
    *,
    quantile: float = 0.975,
    maximum: float = 0.03,
) -> dict[str, float | bool]:
    """Derive a capped margin and mark excess reference variability ineligible."""

    values = np.asarray(absolute_differences, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("absolute differences must be finite, non-negative values")
    if not 0 < quantile < 1 or maximum <= 0:
        raise ValueError("quantile and maximum must be within their valid ranges")
    raw = float(np.quantile(values, quantile, method="linear"))
    return {
        "raw_margin": raw,
        "margin": min(raw, maximum),
        "eligible": raw <= maximum,
    }


def profile_rank_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Frozen lexicographic RF ranking key (smaller is better)."""

    profile = summary["profile"]
    depth_key = 0 if profile["max_depth"] == 16 else 1
    return (
        -float(summary["macro_decision_reason_concordance"]),
        float(summary["median_accuracy_distribution_error"]),
        float(summary["median_runtime_seconds"]),
        int(profile["n_bins"]),
        depth_key,
        int(profile["n_streams"]),
    )


def rank_profiles(summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort all calibration profiles and attach one-based frozen ranks."""

    ranked = [
        dict(summary) for summary in summaries if bool(summary.get("eligible", True))
    ]
    if not ranked:
        raise ValueError("at least one eligible RF profile summary is required")
    ranked.sort(key=profile_rank_key)
    for index, summary in enumerate(ranked, start=1):
        summary["rank"] = index
    return ranked


def comparator_acceptance_checks(
    *,
    overall_concordance: float,
    per_dataset_concordance: Mapping[str, float],
    wilson_lower_95: float,
    median_accuracy_difference: float,
    p95_accuracy_difference: float,
    maximum_unapproved_post_ingest_transfer_bytes: int,
    minimum_vram_headroom: float,
    thresholds: Mapping[str, float | int],
) -> dict[str, bool]:
    """Evaluate every frozen Gate 2 comparator threshold independently."""

    return {
        "overall_decision_concordance": overall_concordance
        >= float(thresholds["overall_decision_concordance_min"]),
        "every_dataset_decision_concordance": bool(per_dataset_concordance)
        and min(per_dataset_concordance.values())
        >= float(thresholds["per_dataset_decision_concordance_min"]),
        "overall_wilson_lower_95": wilson_lower_95
        >= float(thresholds["overall_wilson_lower_95_min"]),
        "median_abs_observed_accuracy_difference": median_accuracy_difference
        <= float(thresholds["median_abs_observed_accuracy_difference_max"]),
        "p95_abs_observed_accuracy_difference": p95_accuracy_difference
        <= float(thresholds["p95_abs_observed_accuracy_difference_max"]),
        "unapproved_post_ingest_transfer_bytes": (
            maximum_unapproved_post_ingest_transfer_bytes
            <= int(thresholds["unapproved_post_ingest_transfer_bytes_max"])
        ),
        "vram_headroom": minimum_vram_headroom
        >= float(thresholds["vram_headroom_min"]),
    }
