"""Pure calibration ranking and Gate 2 acceptance arithmetic."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import sqrt
from statistics import NormalDist
from typing import Any


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


def profile_rank_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    """Frozen lexicographic RF ranking key (smaller is better)."""

    profile = summary["profile"]
    depth = profile["max_depth"]
    depth_key = float("inf") if depth is None else int(depth)
    return (
        -float(summary["macro_decision_concordance"]),
        float(summary["median_abs_observed_accuracy_difference"]),
        float(summary["p95_abs_observed_accuracy_difference"]),
        float(summary["median_runtime_seconds"]),
        int(summary["median_peak_device_bytes"]),
        int(profile["n_bins"]),
        depth_key,
        int(profile["n_streams"]),
    )


def rank_profiles(summaries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sort all calibration profiles and attach one-based frozen ranks."""

    ranked = [dict(summary) for summary in summaries]
    if not ranked:
        raise ValueError("at least one RF profile summary is required")
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
