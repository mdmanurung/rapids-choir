"""Transfer accounting and fixed-workload VRAM preflight."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

from .exceptions import InputResidencyError, InsufficientDeviceMemoryError
from .results import TransferRecord

UNAPPROVED_TRANSFER_LIMIT_BYTES = 1024 * 1024


@dataclass(slots=True)
class TransferLedger:
    records: list[TransferRecord] = field(default_factory=list)
    ingest_complete: bool = False

    def record(
        self,
        *,
        direction: str,
        bytes: int,
        purpose: str,
        approved: bool = False,
    ) -> None:
        if bytes < 0:
            raise ValueError("transfer size cannot be negative")
        if direction not in {"host_to_device", "device_to_host"}:
            raise ValueError("unsupported transfer direction")
        if (
            self.ingest_complete
            and bytes > UNAPPROVED_TRANSFER_LIMIT_BYTES
            and not approved
        ):
            raise InputResidencyError(
                f"unapproved post-ingest transfer of {bytes} bytes exceeds 1 MiB"
            )
        self.records.append(
            TransferRecord(
                direction=direction,
                bytes=bytes,
                purpose=purpose,
                approved=approved,
            )
        )

    def finish_ingest(self) -> None:
        self.ingest_complete = True

    def freeze(self) -> tuple[TransferRecord, ...]:
        return tuple(self.records)


def estimate_comparison_bytes(
    *,
    sampled_rows: int,
    selected_features: int,
    n_trees: int,
    max_depth: int | None,
    n_iterations: int,
) -> int:
    """Conservative peak estimate for sequential cuML fits and GPU statistics."""

    if sampled_rows < 1 or selected_features < 1:
        raise ValueError("sampled_rows and selected_features must be positive")
    dense_bytes = sampled_rows * selected_features * 4
    # Dense input, training workspace, and inference workspace coexist at fit peak.
    fit_workspace = dense_bytes * 3
    label_and_index_bytes = sampled_rows * (4 + 4 + 4 + 1)
    if max_depth is None:
        max_nodes_per_tree = max(1, 2 * sampled_rows - 1)
    else:
        max_nodes_per_tree = min(2 ** (max_depth + 1) - 1, 2 * sampled_rows - 1)
    # Feature, threshold, children, leaf value, and bookkeeping per possible node.
    forest_bytes = n_trees * max_nodes_per_tree * 40
    statistic_bytes = max(1, n_iterations) * max(1, n_iterations) * 8
    safety_metadata = 8 * 1024 * 1024
    return int(
        fit_workspace
        + label_and_index_bytes
        + forest_bytes
        + statistic_bytes
        + safety_metadata
    )


def preflight_device_memory(
    *,
    estimated_bytes: int,
    free_bytes: int,
    headroom_fraction: float,
) -> int:
    """Return required bytes or raise without modifying the workload."""

    required_bytes = ceil(estimated_bytes / (1.0 - headroom_fraction))
    if required_bytes > free_bytes:
        raise InsufficientDeviceMemoryError(
            estimated_bytes=estimated_bytes,
            free_bytes=free_bytes,
            headroom_fraction=headroom_fraction,
            required_bytes=required_bytes,
        )
    return required_bytes
