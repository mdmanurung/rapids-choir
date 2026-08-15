from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from rapids_choir._enums import Decision, ReasonCode, RuleSet
from rapids_choir._persistence import (
    read_comparison_records,
    write_comparison_records,
)
from rapids_choir.exceptions import ArtifactCollisionError, RecordsUnavailableError
from rapids_choir.results import (
    ComparisonMetrics,
    ComparisonResult,
    FeatureCounts,
    MemoryRecord,
    SampleCounts,
    SeedRecord,
    TimingRecord,
)


def _record() -> ComparisonResult:
    return ComparisonResult(
        decision=Decision.SPLIT,
        reason_code=ReasonCode.SPLIT,
        rules=RuleSet.CORRECTED,
        cluster1_id="a",
        cluster2_id="b",
        connectivity=1.0,
        metrics=ComparisonMetrics(0.8, 0.01, 0.5, 0.01, 0.01, 0.01),
        samples=SampleCounts(4, 4, 2, 2, 2, 2, 8),
        features=FeatureCounts(10, 5),
        seeds=SeedRecord(1, 2, (3, 4), (5, 6)),
        timings=TimingRecord(0.1, {"fit": 0.05}),
        memory=MemoryRecord(1, 2, 3, 1),
    )


@pytest.mark.parametrize("records", [(), (_record(),)])
def test_records_round_trip_with_canonical_schema(
    tmp_path: Path, records: tuple[ComparisonResult, ...]
) -> None:
    destination = tmp_path / "records.parquet"
    manifest = write_comparison_records(
        destination,
        records,
        run_id="run-fixture",
        input_sha256="a" * 64,
        overwrite=False,
    )
    frame = read_comparison_records(manifest)
    assert len(frame) == len(records)
    assert manifest.row_count == len(records)
    if records:
        assert frame.loc[0, "decision"] == "split"


def test_records_reject_collisions_and_manifest_tampering(tmp_path: Path) -> None:
    destination = tmp_path / "records.parquet"
    manifest = write_comparison_records(
        destination,
        (_record(),),
        run_id="run-fixture",
        input_sha256="a" * 64,
        overwrite=False,
    )
    with pytest.raises(ArtifactCollisionError):
        write_comparison_records(
            destination,
            (_record(),),
            run_id="run-fixture",
            input_sha256="a" * 64,
            overwrite=False,
        )
    changed = asdict(manifest)
    changed["run_id"] = "run-tampered"
    with pytest.raises(RecordsUnavailableError, match="metadata mismatch"):
        read_comparison_records(changed)


def test_records_reject_checksum_changes(tmp_path: Path) -> None:
    destination = tmp_path / "records.parquet"
    manifest = write_comparison_records(
        destination,
        (_record(),),
        run_id="run-fixture",
        input_sha256="a" * 64,
        overwrite=False,
    )
    destination.write_bytes(destination.read_bytes() + b"tampered")
    with pytest.raises(RecordsUnavailableError, match="checksum"):
        read_comparison_records(manifest)
