#!/usr/bin/env python3
"""Append hash-chained reservations and usage to the v0.1 resource ledger."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import yaml
from gate2_common import immutable_json


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _entry_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def read_entries(handle: TextIO) -> list[dict[str, Any]]:
    handle.seek(0)
    entries = []
    previous: str | None = None
    for line_number, line in enumerate(handle, start=1):
        if not line.endswith("\n"):
            raise RuntimeError(f"ledger line {line_number} is incomplete")
        entry = json.loads(line)
        recorded_hash = entry.pop("entry_sha256", None)
        if entry.get("previous_entry_sha256") != previous:
            raise RuntimeError(f"ledger chain breaks at line {line_number}")
        actual_hash = _entry_hash(entry)
        if recorded_hash != actual_hash:
            raise RuntimeError(f"ledger checksum fails at line {line_number}")
        entry["entry_sha256"] = recorded_hash
        entries.append(entry)
        previous = recorded_hash
    return entries


def totals(entries: list[dict[str, Any]]) -> tuple[float, int]:
    hours = sum(float(entry["l40s_hours_delta"]) for entry in entries)
    scratch = sum(int(entry["scratch_bytes_delta"]) for entry in entries)
    return hours, scratch


def validate_reservation(
    *,
    ledger: Path,
    reservation_path: Path,
    phase: str,
    candidate_commit: str,
    candidate_attempt: int,
) -> dict[str, Any]:
    reservation = json.loads(reservation_path.read_text(encoding="utf-8"))
    with ledger.open(encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        entries = read_entries(handle)
    matches = [
        entry
        for entry in entries
        if entry["entry_sha256"] == reservation.get("entry_sha256")
    ]
    if len(matches) != 1 or matches[0] != reservation:
        raise RuntimeError("resource reservation is absent from the valid ledger")
    if (
        reservation.get("action") != "reserve"
        or reservation.get("phase") != phase
        or reservation.get("candidate_commit") != candidate_commit
        or reservation.get("candidate_attempt") != candidate_attempt
    ):
        raise RuntimeError("resource reservation identity mismatch")
    reconciled = any(
        entry["action"] == "reconcile"
        and entry["reservation_id"] == reservation["reservation_id"]
        for entry in entries
    )
    if reconciled:
        raise RuntimeError("resource reservation has already been reconciled")
    return reservation


def _budget(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise RuntimeError("unsupported resource-budget schema")
    limits = payload.get("limits", {})
    if float(limits.get("cloud_spend_usd", -1)) != 0:
        raise RuntimeError("v0.1 budget must forbid cloud spend")
    return payload


def append_entry(
    *,
    ledger: Path,
    budget_path: Path,
    action: str,
    phase: str,
    candidate_commit: str,
    candidate_attempt: int,
    l40s_hours: float,
    scratch_bytes: int,
    reservation_id: str | None = None,
    failure_classification: str | None = None,
) -> dict[str, Any]:
    if action not in {"reserve", "reconcile"}:
        raise RuntimeError("resource action must be reserve or reconcile")
    if isinstance(candidate_attempt, bool) or candidate_attempt < 1:
        raise RuntimeError("candidate attempt must be positive")
    if l40s_hours < 0 or scratch_bytes < 0:
        raise RuntimeError("resource quantities cannot be negative")
    budget = _budget(budget_path)
    limits = budget["limits"]
    ledger.parent.mkdir(parents=True, exist_ok=True)

    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        entries = read_entries(handle)
        prior_hours, prior_scratch = totals(entries)
        if action == "reserve":
            if reservation_id is not None:
                raise RuntimeError("reserve creates its own reservation identifier")
            delta_hours = l40s_hours
            delta_scratch = scratch_bytes
            reservation_id = str(uuid.uuid4())
        else:
            if reservation_id is None:
                raise RuntimeError("reconcile requires a reservation identifier")
            reservations = [
                entry
                for entry in entries
                if entry["action"] == "reserve"
                and entry["reservation_id"] == reservation_id
            ]
            reconciliations = [
                entry
                for entry in entries
                if entry["action"] == "reconcile"
                and entry["reservation_id"] == reservation_id
            ]
            if len(reservations) != 1 or reconciliations:
                raise RuntimeError("reservation is missing, duplicated, or reconciled")
            reservation = reservations[0]
            if reservation["phase"] != phase:
                raise RuntimeError("reconciliation phase differs from reservation")
            if reservation["candidate_commit"] != candidate_commit or int(
                reservation["candidate_attempt"]
            ) != candidate_attempt:
                raise RuntimeError("reconciliation candidate differs from reservation")
            reserved_hours = float(reservation["l40s_hours_delta"])
            reserved_scratch = int(reservation["scratch_bytes_delta"])
            if l40s_hours > reserved_hours or scratch_bytes > reserved_scratch:
                raise RuntimeError("actual use exceeded the frozen reservation")
            delta_hours = l40s_hours - reserved_hours
            delta_scratch = scratch_bytes - reserved_scratch

        projected_hours = prior_hours + delta_hours
        projected_scratch = prior_scratch + delta_scratch
        if projected_hours < -1e-9 or projected_scratch < 0:
            raise RuntimeError("resource reconciliation would make totals negative")
        if projected_hours > float(limits["l40s_hours"]):
            raise RuntimeError("250 L40S-hour program limit would be exceeded")
        if projected_scratch > int(limits["scratch_bytes"]):
            raise RuntimeError("2 TB scratch program limit would be exceeded")

        previous = entries[-1]["entry_sha256"] if entries else None
        entry: dict[str, Any] = {
            "schema_version": 1,
            "entry_id": str(uuid.uuid4()),
            "generated_at": datetime.now(UTC).isoformat(),
            "action": action,
            "phase": phase,
            "candidate_commit": candidate_commit,
            "candidate_attempt": candidate_attempt,
            "reservation_id": reservation_id,
            "l40s_hours_delta": delta_hours,
            "scratch_bytes_delta": delta_scratch,
            "actual_l40s_hours": l40s_hours if action == "reconcile" else None,
            "actual_scratch_bytes": scratch_bytes if action == "reconcile" else None,
            "failure_classification": failure_classification,
            "budget_sha256": hashlib.sha256(budget_path.read_bytes()).hexdigest(),
            "previous_entry_sha256": previous,
            "totals_after": {
                "l40s_hours": projected_hours,
                "scratch_bytes": projected_scratch,
            },
        }
        entry_hash = _entry_hash(entry)
        published = {**entry, "entry_sha256": entry_hash}
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(published, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        return published


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("reserve", "reconcile"))
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=Path)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-attempt", required=True, type=int)
    parser.add_argument("--l40s-hours", required=True, type=float)
    parser.add_argument("--scratch-bytes", required=True, type=int)
    parser.add_argument("--reservation-id")
    parser.add_argument("--failure-classification")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entry = append_entry(
        ledger=args.ledger,
        budget_path=args.budget,
        action=args.action,
        phase=args.phase,
        candidate_commit=args.candidate_commit,
        candidate_attempt=args.candidate_attempt,
        l40s_hours=args.l40s_hours,
        scratch_bytes=args.scratch_bytes,
        reservation_id=args.reservation_id,
        failure_classification=args.failure_classification,
    )
    immutable_json(args.output, entry)


if __name__ == "__main__":
    main()
