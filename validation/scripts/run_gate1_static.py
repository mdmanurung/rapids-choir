#!/usr/bin/env python3
"""Run and record the CUDA-independent part of Gate 1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import tarfile
import time
import zipfile
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

EXPECTED_CONDA_PACKAGES = {
    "rapids": "26.06.01",
    "cuml": "26.06.00",
    "cugraph": "26.06.00",
    "rmm": "26.06.00",
}
RUN_STAGE = "preflight"


def forbidden_packaged_members(members: list[str]) -> list[str]:
    """Return generated, private, or unreviewed paths forbidden in artifacts."""

    forbidden = []
    for member in members:
        normalized = "/" + member.lstrip("/")
        authorization_path = "/workflow/authorization/" in normalized
        authorization_readme = normalized.endswith(
            "/workflow/authorization/README.md"
        )
        if (
            "/docs/wayfinder/" in normalized
            or "/validation/evidence/candidates/" in normalized
            or "/validation/logs/" in normalized
            or normalized.endswith(".FAILED.json")
            or normalized.endswith(
                "/validation/evidence/governance/resource_ledger.jsonl"
            )
            or (authorization_path and not authorization_readme)
        ):
            forbidden.append(member)
    return sorted(forbidden)


def archive_members(path: Path) -> list[str]:
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            return archive.getnames()
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    raise RuntimeError(f"unsupported package artifact: {path}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(root: Path, command: list[str]) -> dict[str, Any]:
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


def conda_versions(prefix: Path) -> dict[str, str | None]:
    versions: dict[str, str] = {}
    for path in (prefix / "conda-meta").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        versions[record["name"]] = record["version"]
    return {name: versions.get(name) for name in EXPECTED_CONDA_PACKAGES}


def main() -> None:
    global RUN_STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    candidate = validate_candidate_manifest(root, args.candidate_manifest)
    python = Path(sys.executable).resolve()
    prefix = python.parent.parent
    lock = root / "workflow/envs/rapids-cu128-linux-64.lock"
    pip_lock = root / "workflow/envs/rapids-cu128-pip.lock"
    lock_text = lock.read_text(encoding="utf-8")
    if "cuda-version-12.8" not in lock_text:
        raise RuntimeError("explicit lock does not pin cuda-version 12.8")
    if "cuda13" in lock_text.lower():
        raise RuntimeError("explicit lock contains a CUDA 13 package")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(f"Gate 1 requires Python 3.12, got {sys.version}")

    versions = conda_versions(prefix)
    if versions != EXPECTED_CONDA_PACKAGES:
        raise RuntimeError(
            "locked distribution drift: "
            f"expected={EXPECTED_CONDA_PACKAGES}, got={versions}"
        )
    rsc_version = importlib.metadata.version("rapids-singlecell-cu12")
    if rsc_version != "0.16.1":
        raise RuntimeError(f"rapids-singlecell drift: {rsc_version}")

    commands = [
        [
            str(python.parent / "ruff"),
            "check",
            "src",
            "tests",
            "validation/scripts",
            "workflow",
        ],
        [str(python.parent / "mypy"), "src/rapids_choir"],
        [str(python), "-m", "pytest", "tests/unit", "-q"],
        [str(python), "-m", "build", "--no-isolation"],
        [
            str(python),
            "-c",
            (
                "import json,sys,rapids_choir; "
                "names=('cupy','rmm','cuml','cugraph','rapids_singlecell'); "
                "print(json.dumps([n for n in names if n in sys.modules]))"
            ),
        ],
    ]
    RUN_STAGE = "static_checks"
    checks = [run(root, command) for command in commands]
    commands_failed = any(check["returncode"] != 0 for check in checks)
    cuda_imported = checks[-1]["stdout"].strip() != "[]"
    status = "FAIL" if commands_failed or cuda_imported else "PASS"

    artifacts = []
    forbidden_members: dict[str, list[str]] = {}
    for path in sorted((root / "dist").glob("rapids_choir-0.1.0.dev0*")):
        forbidden = forbidden_packaged_members(archive_members(path))
        if forbidden:
            forbidden_members[path.name] = forbidden
        artifacts.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    if len(artifacts) != 2:
        status = "FAIL"
    if forbidden_members:
        status = "FAIL"

    revision = run(root, ["git", "rev-parse", "HEAD"])
    payload = {
        "schema_version": 1,
        "gate": 1,
        "scope": "cpu_static_only",
        "status": status,
        "functional_validation": False,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit": revision["stdout"].strip() if revision["returncode"] == 0 else None,
        "candidate": {
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": file_hash(args.candidate_manifest),
            "dirty_state": candidate["dirty_state"],
        },
        "python": sys.version.split()[0],
        "python_executable": str(python),
        "conda_packages": versions,
        "rapids_singlecell_cu12": rsc_version,
        "locks": {
            lock.relative_to(root).as_posix(): sha256(lock),
            pip_lock.relative_to(root).as_posix(): sha256(pip_lock),
        },
        "artifacts": artifacts,
        "forbidden_package_members": forbidden_members,
        "checks": checks,
        "limitations": [
            "This evidence covers only imports, schemas, logic, typing, and packaging.",
            "It is not GPU functional or scientific acceptance evidence.",
        ],
    }
    immutable_text(
        args.log,
        "\n\n".join(
            f"$ {' '.join(check['command'])}\n{check['stdout']}{check['stderr']}"
            for check in checks
        ),
    )
    immutable_json(args.output, payload)
    if status != "PASS":
        preserve_failure_record(
            args.output,
            scope="gate1_static",
            stage=RUN_STAGE,
            method_started=True,
            failure_classification="technical_outcome",
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
                    scope="gate1_static",
                    stage=RUN_STAGE,
                    method_started=RUN_STAGE == "static_checks",
                    failure_classification=(
                        "technical_outcome"
                        if RUN_STAGE == "static_checks"
                        else "custody_or_preflight_rejection"
                    ),
                    error=error,
                )
            except Exception as preservation_error:  # pragma: no cover
                print(
                    f"could not preserve Gate 1 static failure: "
                    f"{preservation_error}",
                    file=sys.stderr,
                )
        raise
