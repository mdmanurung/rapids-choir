#!/usr/bin/env python3
"""Run Gate 1 functional checks on a real typed L40S allocation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gate2_common import (
    cli_path_argument,
    file_hash,
    immutable_json,
    immutable_text,
    preserve_failure_record,
    validate_candidate_manifest,
)
from resource_ledger import validate_reservation

RUN_STAGE = "preflight"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(root: Path, command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def conda_packages(prefix: Path) -> dict[str, dict[str, str]]:
    packages: dict[str, dict[str, str]] = {}
    for path in (prefix / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        packages[record["name"]] = {
            "version": record["version"],
            "build": record["build"],
        }
    return packages


def main() -> None:
    global RUN_STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--resource-reservation", type=Path, required=True)
    parser.add_argument("--resource-ledger", type=Path, required=True)
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
    python = Path(sys.executable).resolve()
    prefix = python.parent.parent
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("Gate 1 GPU evidence must run inside a Slurm allocation")

    RUN_STAGE = "gpu_functional_checks"
    doctor = execute(
        root,
        [str(python), "-m", "rapids_choir", "doctor", "--strict", "--json"],
    )
    doctor_payload: dict[str, Any] = {}
    if doctor["stdout"]:
        doctor_payload = json.loads(doctor["stdout"])
    tests = execute(
        root,
        [
            str(python),
            "-m",
            "pytest",
            "tests/gpu/test_statistics.py",
            "tests/gpu/test_sampling.py",
            "tests/gpu/test_doctor.py",
            "tests/gpu/test_comparator.py",
            "-q",
        ],
    )
    nvidia_smi = execute(
        root,
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
    )
    revision = execute(root, ["git", "rev-parse", "HEAD"])
    tracked_status = execute(
        root, ["git", "status", "--porcelain", "--untracked-files=no"]
    )

    packages = conda_packages(prefix)
    expected_conda = {
        "cuda-version": "12.8",
        "rapids": "26.06.01",
        "cuml": "26.06.00",
        "cugraph": "26.06.00",
        "rmm": "26.06.00",
    }
    actual_conda = {
        name: packages.get(name, {}).get("version") for name in expected_conda
    }
    cuda13_packages = sorted(
        name
        for name, record in packages.items()
        if "cuda13" in f"{name}-{record['build']}".lower()
    )
    rsc_version = importlib.metadata.version("rapids-singlecell-cu12")

    device_name = ""
    cuda_path = ""
    toolkit_inside_env = False
    if doctor_payload:
        for check in doctor_payload.get("checks", []):
            if check.get("name") == "cuda_device_and_rmm":
                device_name = check.get("details", {}).get("name", "")
            if check.get("name") == "cuda_toolkit_locality":
                toolkit_inside_env = bool(check.get("passed"))
                cuda_path = check.get("details", {}).get("cuda_path", "")

    status = "PASS"
    failure_reasons: list[str] = []
    conditions = {
        "doctor": doctor["returncode"] == 0 and doctor_payload.get("status") == "PASS",
        "gpu_tests": tests["returncode"] == 0,
        "nvidia_smi": nvidia_smi["returncode"] == 0,
        "typed_l40s": "L40S" in device_name,
        "conda_versions": actual_conda == expected_conda,
        "rapids_singlecell": rsc_version == "0.16.1",
        "single_cuda_major": not cuda13_packages,
        # Package versions cannot see a toolkit resolved from outside the
        # environment, so this condition is not redundant with the one above.
        "cuda_toolkit_inside_env": toolkit_inside_env,
        "clean_tracked_tree": tracked_status["returncode"] == 0
        and not tracked_status["stdout"].strip(),
        "commit_resolved": revision["returncode"] == 0,
    }
    for name, passed in conditions.items():
        if not passed:
            failure_reasons.append(name)
    if failure_reasons:
        status = "FAIL"
    failure_classification = None
    if status == "FAIL":
        infrastructure_conditions = {"nvidia_smi", "typed_l40s"}
        failure_classification = (
            "infrastructure_pre_method"
            if infrastructure_conditions.intersection(failure_reasons)
            else "technical_outcome"
        )

    lock = root / "workflow/envs/rapids-cu128-linux-64.lock"
    pip_lock = root / "workflow/envs/rapids-cu128-pip.lock"
    payload = {
        "schema_version": 1,
        "gate": 1,
        "scope": "real_gpu_functional",
        "status": status,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": revision["stdout"].strip(),
        "candidate": {
            "commit": candidate["candidate_commit"],
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": file_hash(args.candidate_manifest),
            "dirty_state": candidate["dirty_state"],
        },
        "resource_reservation": {
            "reservation_id": reservation["reservation_id"],
            "sha256": file_hash(args.resource_reservation),
        },
        "slurm": {
            "job_id": slurm_job_id,
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gres": os.environ.get("SLURM_JOB_GRES"),
        },
        "device": {
            "doctor_name": device_name,
            "nvidia_smi": nvidia_smi["stdout"].strip(),
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": str(python),
            "conda": actual_conda,
            "rapids_singlecell_cu12": rsc_version,
            "cuda13_packages": cuda13_packages,
            "cuda_path": cuda_path,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "locks": {
                lock.relative_to(root).as_posix(): sha256(lock),
                pip_lock.relative_to(root).as_posix(): sha256(pip_lock),
            },
        },
        "conditions": conditions,
        "failure_reasons": failure_reasons,
        "failure_classification": failure_classification,
        "doctor": doctor_payload,
        "gpu_tests": {
            "returncode": tests["returncode"],
            "seconds": tests["seconds"],
            "stdout": tests["stdout"],
            "stderr": tests["stderr"],
        },
    }
    immutable_text(
        args.log,
        (
            f"DOCTOR\n{doctor['stdout']}{doctor['stderr']}\n"
            f"GPU TESTS\n{tests['stdout']}{tests['stderr']}\n"
            f"NVIDIA-SMI\n{nvidia_smi['stdout']}{nvidia_smi['stderr']}"
        ),
    )
    immutable_json(args.output, payload)
    if status != "PASS":
        # Snakemake deletes a failed job's declared outputs, which would destroy
        # the only record of why the gate failed. This sibling is not a rule
        # output, so it survives.
        preserved = args.output.with_name(args.output.stem + ".FAILED.json")
        immutable_json(preserved, payload)
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if (output := cli_path_argument("--output")) is not None:
            try:
                preserve_failure_record(
                    output,
                    scope="gate1_gpu",
                    stage=RUN_STAGE,
                    method_started=RUN_STAGE == "gpu_functional_checks",
                    failure_classification=(
                        "technical_outcome"
                        if RUN_STAGE == "gpu_functional_checks"
                        else "custody_or_infrastructure_preflight"
                    ),
                    error=error,
                )
            except Exception as preservation_error:  # pragma: no cover
                print(
                    f"could not preserve Gate 1 GPU failure: "
                    f"{preservation_error}",
                    file=sys.stderr,
                )
        raise
