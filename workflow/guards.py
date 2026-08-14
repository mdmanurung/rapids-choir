"""Fail-closed checks applied before Snakemake constructs the workflow DAG."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def _option_value(argv: Sequence[str], name: str) -> str | None:
    """Return one CLI option value, accepting ``--name x`` and ``--name=x``."""

    prefix = f"{name}="
    for index, argument in enumerate(argv):
        if argument.startswith(prefix):
            return argument[len(prefix) :]
        if argument == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def _profile_executor(profile: str, *, base_dir: Path) -> str | None:
    path = Path(profile).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    candidates = (path, path / "config.yaml", path / "config.yml")
    config_path = next((value for value in candidates if value.is_file()), None)
    if config_path is None:
        return None
    payload: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return None
    executor = payload.get("executor")
    return str(executor).strip().lower() if executor is not None else None


def requested_executor(
    argv: Sequence[str],
    environ: Mapping[str, str],
    *,
    base_dir: Path,
) -> str | None:
    """Resolve only enough CLI/profile state to distinguish Slurm from local."""

    direct = _option_value(argv, "--executor") or environ.get(
        "SNAKEMAKE_EXECUTOR"
    )
    if direct:
        return direct.strip().lower()
    profile = _option_value(argv, "--profile") or environ.get("SNAKEMAKE_PROFILE")
    if profile:
        return _profile_executor(profile, base_dir=base_dir)
    return None


def enforce_executor_context(
    argv: Sequence[str],
    environ: Mapping[str, str] | None = None,
    *,
    base_dir: Path | None = None,
) -> None:
    """Reject a Slurm *driver* launched from inside an existing allocation.

    Snakemake's per-rule worker invocation is local and therefore remains
    permitted. Inside an allocation, operators may also deliberately use the
    local executor with ``--cores``.
    """

    values = os.environ if environ is None else environ
    job_id = values.get("SLURM_JOB_ID")
    if not job_id:
        return
    executor = requested_executor(
        argv,
        values,
        base_dir=Path.cwd() if base_dir is None else base_dir,
    )
    if executor == "slurm":
        raise RuntimeError(
            "refusing to launch the Snakemake Slurm executor from inherited "
            f"allocation SLURM_JOB_ID={job_id}; launch the driver outside an "
            "allocation, or use the local executor with --cores inside it"
        )
