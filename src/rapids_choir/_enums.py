"""Stable public vocabulary."""

from __future__ import annotations

from enum import StrEnum


class RuleSet(StrEnum):
    CORRECTED = "corrected"
    CHOIR_V0_3_0 = "choir_v0_3_0"


class InputPolicy(StrEnum):
    TRANSFER_ONCE = "transfer_once"
    DEVICE_ONLY = "device_only"


class Multiplicity(StrEnum):
    NONE = "none"
    BONFERRONI = "bonferroni"
    BH = "bh"


class Decision(StrEnum):
    SPLIT = "split"
    MERGE = "merge"


class ReasonCode(StrEnum):
    SPLIT = "split"
    SPLIT_MIN_CONNECTIONS = "split_min_connections"
    SPLIT_REPEAT_ERROR = "split_repeat_error"
    SPLIT_DISTANCE = "split_distance"
    MERGE = "merge"
    MERGE_BATCH_DEPENDENT = "merge_batch_dependent"
    MERGE_MINIMUM_READS = "merge_minimum_reads"
    MERGE_MIN_ACCURACY = "merge_min_accuracy"
    MERGE_ADJUSTMENT = "merge_adjustment"


REFERENCE_REASON_ALIASES: dict[ReasonCode, str] = {
    ReasonCode.SPLIT: "split",
    ReasonCode.SPLIT_MIN_CONNECTIONS: "split: min connections",
    ReasonCode.SPLIT_REPEAT_ERROR: "split: repeat error",
    ReasonCode.SPLIT_DISTANCE: "split: distance",
    ReasonCode.MERGE: "merge",
    ReasonCode.MERGE_BATCH_DEPENDENT: "merge: batch-dependent",
    ReasonCode.MERGE_MINIMUM_READS: "merge: minimum reads",
    ReasonCode.MERGE_MIN_ACCURACY: "merge: min accuracy",
    ReasonCode.MERGE_ADJUSTMENT: "merge: adjustment",
}
