#!/usr/bin/env python3
"""Shared file/provenance helpers for Gate 2 scripts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

RUNTIME_PATHS = (
    ".gitignore",
    "CITATION.cff",
    "containers/",
    "LICENSE",
    "NOTICE",
    "README.md",
    "Snakefile",
    "docs/architecture/",
    "docs/decisions/",
    "docs/validation/",
    "figures/",
    "manuscript/",
    "pyproject.toml",
    "renv.lock",
    "release/",
    "scripts/",
    "smoke/",
    "src/",
    "tests/",
    "tables/",
    "supplement/",
    "validation/contracts/",
    "validation/fixtures/specs/",
    "validation/governance/",
    "validation/reference/choir_v0_3_0/",
    "validation/scripts/",
    "workflow/",
)


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _is_runtime_path(path: str) -> bool:
    return any(
        path == prefix or (prefix.endswith("/") and path.startswith(prefix))
        for prefix in RUNTIME_PATHS
    )


def runtime_tree_digest(root: Path) -> tuple[str, list[str]]:
    """Hash names and bytes of every tracked repository file."""

    untracked = git_output(
        root, "ls-files", "--others", "--exclude-standard"
    ).splitlines()
    untracked_runtime = sorted(
        path for path in untracked if _is_runtime_path(path)
    )
    if untracked_runtime:
        raise RuntimeError(
            "runtime-controlled files must be tracked before candidate freeze: "
            f"{untracked_runtime}"
        )
    tracked = git_output(root, "ls-files").splitlines()
    paths = sorted(tracked)
    if not paths:
        raise RuntimeError("runtime-controlled tree is empty")
    digest = hashlib.sha256()
    for relative in paths:
        encoded = relative.encode("utf-8")
        payload = (root / relative).read_bytes()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest(), paths


def validate_candidate_manifest(root: Path, path: Path) -> dict[str, Any]:
    """Validate a candidate seal against the current clean tracked tree."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN":
        raise RuntimeError("candidate manifest is not FROZEN")
    commit = git_output(root, "rev-parse", "HEAD")
    if payload.get("candidate_commit") != commit:
        raise RuntimeError("candidate manifest commit does not match HEAD")
    dirty = git_output(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("candidate execution requires a clean tracked tree")
    digest, paths = runtime_tree_digest(root)
    if payload.get("runtime_tree_sha256") != digest:
        raise RuntimeError("candidate runtime tree digest does not match")
    if payload.get("runtime_file_count") != len(paths):
        raise RuntimeError("candidate runtime file count does not match")
    return payload


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path: Path, payload: Any) -> None:
    """Atomically create checksum-bearing evidence without overwriting it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    install_immutable_file(temporary, path)


def immutable_text(path: Path, value: str) -> None:
    """Atomically create a UTF-8 text artifact without overwriting it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{uuid4().hex}")
    temporary.write_text(value, encoding="utf-8")
    install_immutable_file(temporary, path)


def install_immutable_file(temporary: Path, destination: Path) -> None:
    """Install a completed same-filesystem temporary file without replacement."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise RuntimeError(
            f"immutable evidence already exists: {destination}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)


def cli_path_argument(
    name: str,
    argv: Sequence[str] | None = None,
) -> Path | None:
    """Resolve a single path-valued CLI option for last-resort failure capture."""

    values = sys.argv if argv is None else argv
    try:
        index = values.index(name)
    except ValueError:
        return None
    if index + 1 >= len(values):
        return None
    return Path(values[index + 1])


def preserve_failure_record(
    output: Path,
    *,
    scope: str,
    stage: str,
    method_started: bool,
    failure_classification: str,
    error: BaseException | None = None,
    evidence: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a machine-readable sibling that Snakemake will not delete."""

    failure = output.with_name(output.stem + ".FAILED.json")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAIL",
        "scope": scope,
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": stage,
        "method_started": method_started,
        "failure_classification": failure_classification,
        "exception_type": type(error).__name__ if error is not None else None,
        "exception_message": str(error) if error is not None else None,
        "traceback": (
            "".join(traceback.format_exception(error))
            if error is not None
            else None
        ),
        "evidence": dict(evidence) if evidence is not None else None,
        "context": dict(context) if context is not None else None,
    }
    immutable_json(failure, payload)
    return failure


def identifier_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def dataset_number(dataset_id: str) -> int:
    try:
        return int(dataset_id.rsplit(" ", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"unsupported frozen dataset id: {dataset_id}") from error


def dataset_slug(dataset_id: str) -> str:
    return f"simulated_dataset_{dataset_number(dataset_id):03d}"
