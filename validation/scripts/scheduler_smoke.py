#!/usr/bin/env python3
"""Non-evidence typed-L40S scheduler smoke before candidate Gate 1."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from gate2_common import (
    cli_path_argument,
    file_hash,
    immutable_json,
    preserve_failure_record,
    validate_candidate_manifest,
)
from resource_ledger import validate_reservation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--resource-reservation", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    candidate = validate_candidate_manifest(root, args.candidate_manifest)
    reservation = validate_reservation(
        ledger=args.resource_ledger,
        reservation_path=args.resource_reservation,
        phase="gate1_gpu",
        candidate_commit=str(candidate["candidate_commit"]),
        candidate_attempt=int(candidate["candidate_attempt"]),
    )
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("scheduler smoke must run inside a Slurm allocation")
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    device = completed.stdout.strip()
    checks = {
        "nvidia_smi": completed.returncode == 0,
        "typed_l40s": "L40S" in device,
        "one_visible_device": len(
            [line for line in device.splitlines() if line.strip()]
        )
        == 1,
        "python_3_12": sys.version_info[:2] == (3, 12),
    }
    payload = {
        "schema_version": 1,
        "scope": "non_evidence_scheduler_smoke",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_commit": candidate["candidate_commit"],
        "candidate_attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "candidate_manifest_sha256": file_hash(args.candidate_manifest),
        "reservation_sha256": file_hash(args.resource_reservation),
        "reservation_id": reservation["reservation_id"],
        "checks": checks,
        "slurm": {
            "job_id": job_id,
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gres": os.environ.get("SLURM_JOB_GRES"),
        },
        "device": device,
        "stderr": completed.stderr,
        "failure_classification": (
            None if all(checks.values()) else "infrastructure_pre_method"
        ),
    }
    immutable_json(args.output, payload)
    if payload["status"] != "PASS":
        preserve_failure_record(
            args.output,
            scope="non_evidence_scheduler_smoke",
            stage="preflight",
            method_started=False,
            failure_classification="infrastructure_pre_method",
            evidence=payload,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if (output := cli_path_argument("--output")) is not None:
            try:
                preserve_failure_record(
                    output,
                    scope="non_evidence_scheduler_smoke",
                    stage="preflight",
                    method_started=False,
                    failure_classification="custody_or_preflight_rejection",
                    error=error,
                )
            except Exception as preservation_error:  # pragma: no cover
                print(
                    f"could not preserve scheduler-smoke failure: "
                    f"{preservation_error}",
                    file=sys.stderr,
                )
        raise
