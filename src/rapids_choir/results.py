"""Typed result and provenance records."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ._enums import Decision, ReasonCode, RuleSet
from .tree import HierarchyTree


@dataclass(frozen=True, slots=True)
class ComparisonMetrics:
    mean_observed_accuracy: float
    variance_observed_accuracy: float
    mean_null_accuracy: float
    variance_null_accuracy: float
    accuracy_reference_tail_probability: float
    variance_reference_tail_probability: float
    mean_modified_accuracy: float | None = None
    variance_modified_accuracy: float | None = None
    modified_accuracy_reference_tail_probability: float | None = None
    modified_variance_reference_tail_probability: float | None = None
    repeated_error_cells_cluster1: int = 0
    repeated_error_cells_cluster2: int = 0
    batch_mean_accuracies: dict[str, float] = field(default_factory=dict)
    batch_variance_accuracies: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SampleCounts:
    cluster1_cells: int
    cluster2_cells: int
    per_class_per_half: int
    split_rounds: int
    observed_accuracies: int
    null_accuracies: int
    forest_fits: int


@dataclass(frozen=True, slots=True)
class FeatureCounts:
    available: int
    selected: int


@dataclass(frozen=True, slots=True)
class SeedRecord:
    root: int
    operation: int
    round_seeds: tuple[int, ...] = ()
    fit_seeds: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TimingRecord:
    total_seconds: float
    components_seconds: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransferRecord:
    direction: str
    bytes: int
    purpose: str
    approved: bool


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    estimated_bytes: int
    free_bytes_at_preflight: int
    total_device_bytes: int
    required_headroom_bytes: int
    peak_device_bytes: int | None = None
    transfers: tuple[TransferRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    decision: Decision
    reason_code: ReasonCode
    rules: RuleSet
    cluster1_id: str
    cluster2_id: str
    connectivity: float | None
    metrics: ComparisonMetrics | None
    samples: SampleCounts
    features: FeatureCounts
    seeds: SeedRecord
    timings: TimingRecord
    memory: MemoryRecord
    compatibility_aliases: dict[str, Any] = field(default_factory=dict)
    pair_kind: str = "sibling"
    input_membership_sha256: tuple[str, str] | None = None
    root_distance: float | None = None
    subtree_distance: float | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecordsManifest:
    schema_version: str
    uri: str
    sha256: str
    row_count: int
    run_id: str
    input_sha256: str


@dataclass(frozen=True, slots=True)
class ChoirResult:
    key_added: str
    labels: Any
    records: tuple[ComparisonResult, ...]
    provenance: dict[str, Any]
    tree: HierarchyTree
    records_manifest: RecordsManifest | None = None
