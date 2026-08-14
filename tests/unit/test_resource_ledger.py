from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "resource_ledger", SCRIPTS / "resource_ledger.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def budget(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "limits": {
                    "l40s_hours": 250.0,
                    "scratch_bytes": 2_000_000_000_000,
                    "cloud_spend_usd": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def append(
    ledger: Path,
    budget_path: Path,
    *,
    action: str,
    hours: float,
    scratch: int,
    reservation_id: str | None = None,
) -> dict[str, object]:
    return MODULE.append_entry(
        ledger=ledger,
        budget_path=budget_path,
        action=action,
        phase="comparator_development",
        candidate_commit="a" * 40,
        candidate_attempt=1,
        l40s_hours=hours,
        scratch_bytes=scratch,
        reservation_id=reservation_id,
    )


def test_reservation_reconciles_to_actual_use(tmp_path: Path) -> None:
    budget_path = budget(tmp_path / "budget.yaml")
    ledger = tmp_path / "ledger.jsonl"
    reservation = append(
        ledger, budget_path, action="reserve", hours=120, scratch=500
    )

    reconciliation = append(
        ledger,
        budget_path,
        action="reconcile",
        hours=20,
        scratch=200,
        reservation_id=str(reservation["reservation_id"]),
    )

    assert reconciliation["l40s_hours_delta"] == -100
    assert reconciliation["totals_after"] == {
        "l40s_hours": 20,
        "scratch_bytes": 200,
    }
    with ledger.open(encoding="utf-8") as handle:
        assert len(MODULE.read_entries(handle)) == 2


def test_reservation_fails_before_program_limit_is_exceeded(tmp_path: Path) -> None:
    budget_path = budget(tmp_path / "budget.yaml")
    ledger = tmp_path / "ledger.jsonl"
    append(ledger, budget_path, action="reserve", hours=240, scratch=500)

    with pytest.raises(RuntimeError, match="250 L40S-hour"):
        append(ledger, budget_path, action="reserve", hours=11, scratch=500)


def test_ledger_checksum_detects_mutation(tmp_path: Path) -> None:
    budget_path = budget(tmp_path / "budget.yaml")
    ledger = tmp_path / "ledger.jsonl"
    append(ledger, budget_path, action="reserve", hours=1, scratch=1)
    text = ledger.read_text(encoding="utf-8").replace(
        '"l40s_hours_delta": 1', '"l40s_hours_delta": 2'
    )
    ledger.write_text(text, encoding="utf-8")

    with (
        ledger.open(encoding="utf-8") as handle,
        pytest.raises(RuntimeError, match="checksum"),
    ):
        MODULE.read_entries(handle)
