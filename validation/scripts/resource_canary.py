#!/usr/bin/env python3
"""Allocate the frozen worst-shard estimate on one real L40S."""

from __future__ import annotations

import argparse
import gc
import os
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

from rapids_choir._cuda import initialize_cuda
from rapids_choir._memory import (
    estimate_comparison_bytes,
    preflight_device_memory,
)

RUN_STAGE = "preflight"


def main() -> None:
    global RUN_STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--resource-reservation", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--cells", required=True, type=int)
    parser.add_argument("--features", required=True, type=int)
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
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("resource canary must run inside Slurm")
    if args.cells < 1 or args.features < 1:
        raise RuntimeError("canary dimensions must be positive")

    context = initialize_cuda("cuda:0")
    if "L40S" not in context.device_name:
        raise RuntimeError(f"resource canary requires L40S, got {context.device_name}")
    RUN_STAGE = "allocation_canary"
    cp = context.cupy
    estimated = estimate_comparison_bytes(
        sampled_rows=args.cells,
        selected_features=args.features,
        n_trees=50,
        max_depth=None,
        n_iterations=100,
    )
    free_before, total = context.memory_info()
    required = preflight_device_memory(
        estimated_bytes=estimated,
        free_bytes=free_before,
        headroom_fraction=0.20,
    )
    allocation = cp.empty(estimated, dtype=cp.uint8)
    allocation.fill(0)
    cp.cuda.runtime.deviceSynchronize()
    free_during, _ = context.memory_info()
    del allocation
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.runtime.deviceSynchronize()
    free_after, _ = context.memory_info()

    payload = {
        "schema_version": 1,
        "scope": "non_evidence_worst_shard_resource_canary",
        "status": "PASS",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_commit": candidate["candidate_commit"],
        "candidate_attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "candidate_manifest_sha256": file_hash(args.candidate_manifest),
        "reservation_sha256": file_hash(args.resource_reservation),
        "reservation_id": reservation["reservation_id"],
        "shape": {"sampled_rows": args.cells, "selected_features": args.features},
        "rf_shape": {"n_trees": 50, "max_depth": None, "n_iterations": 100},
        "estimated_device_bytes": estimated,
        "required_bytes_with_headroom": required,
        "free_bytes_before": free_before,
        "free_bytes_during": free_during,
        "free_bytes_after": free_after,
        "total_device_bytes": total,
        "device": context.device_name,
        "slurm": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gres": os.environ.get("SLURM_JOB_GRES"),
        },
        "failure_classification": None,
    }
    immutable_json(args.output, payload)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if (output := cli_path_argument("--output")) is not None:
            try:
                preserve_failure_record(
                    output,
                    scope="non_evidence_worst_shard_resource_canary",
                    stage=RUN_STAGE,
                    method_started=False,
                    failure_classification=(
                        "resource_canary_outcome"
                        if RUN_STAGE == "allocation_canary"
                        else "custody_or_infrastructure_preflight"
                    ),
                    error=error,
                )
            except Exception as preservation_error:  # pragma: no cover
                print(
                    f"could not preserve resource-canary failure: "
                    f"{preservation_error}",
                    file=sys.stderr,
                )
        raise
