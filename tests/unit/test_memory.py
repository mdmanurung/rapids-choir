from __future__ import annotations

import pytest

from rapids_choir._memory import (
    TransferLedger,
    estimate_comparison_bytes,
    preflight_device_memory,
)
from rapids_choir.exceptions import InputResidencyError, InsufficientDeviceMemoryError


def test_preflight_preserves_twenty_percent_headroom() -> None:
    assert (
        preflight_device_memory(
            estimated_bytes=800,
            free_bytes=1000,
            headroom_fraction=0.2,
        )
        == 1000
    )
    with pytest.raises(InsufficientDeviceMemoryError) as captured:
        preflight_device_memory(
            estimated_bytes=801,
            free_bytes=1000,
            headroom_fraction=0.2,
        )
    assert captured.value.required_bytes == 1002


def test_memory_estimate_increases_with_workload() -> None:
    small = estimate_comparison_bytes(
        sampled_rows=100,
        selected_features=50,
        n_trees=10,
        max_depth=16,
        n_iterations=10,
    )
    large = estimate_comparison_bytes(
        sampled_rows=200,
        selected_features=50,
        n_trees=10,
        max_depth=16,
        n_iterations=10,
    )
    assert large > small


def test_transfer_ledger_blocks_large_unapproved_post_ingest_transfer() -> None:
    ledger = TransferLedger()
    ledger.record(
        direction="host_to_device", bytes=10_000_000, purpose="initial ingest"
    )
    ledger.finish_ingest()
    ledger.record(direction="device_to_host", bytes=128, purpose="scalar metric")
    with pytest.raises(InputResidencyError, match="exceeds 1 MiB"):
        ledger.record(
            direction="device_to_host", bytes=2_000_000, purpose="hidden copy"
        )
