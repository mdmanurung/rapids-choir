from __future__ import annotations

from pathlib import Path

import pytest

from workflow.guards import enforce_executor_context, requested_executor


def test_slurm_executor_is_rejected_inside_allocation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="SLURM_JOB_ID=25418412"):
        enforce_executor_context(
            ["snakemake", "--executor", "slurm"],
            {"SLURM_JOB_ID": "25418412"},
            base_dir=tmp_path,
        )


def test_local_executor_is_allowed_inside_allocation(tmp_path: Path) -> None:
    enforce_executor_context(
        ["snakemake", "--cores", "4"],
        {"SLURM_JOB_ID": "25418412"},
        base_dir=tmp_path,
    )


def test_slurm_profile_is_resolved_and_rejected(tmp_path: Path) -> None:
    profile = tmp_path / "profiles" / "slurm"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("executor: slurm\n", encoding="utf-8")
    assert requested_executor(
        ["snakemake", "--profile", "profiles/slurm"],
        {"SLURM_JOB_ID": "1"},
        base_dir=tmp_path,
    ) == "slurm"
    with pytest.raises(RuntimeError, match="launch the driver outside"):
        enforce_executor_context(
            ["snakemake", "--profile", "profiles/slurm"],
            {"SLURM_JOB_ID": "1"},
            base_dir=tmp_path,
        )


def test_slurm_executor_is_allowed_without_inherited_allocation(
    tmp_path: Path,
) -> None:
    enforce_executor_context(
        ["snakemake", "--executor=slurm"],
        {},
        base_dir=tmp_path,
    )
