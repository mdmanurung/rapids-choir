from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "freeze_candidate", SCRIPTS / "freeze_candidate.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def initialise_repository(root: Path) -> tuple[str, Path]:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    snakefile = root / "Snakefile"
    snakefile.write_text("rule all:\n    input: []\n", encoding="utf-8")
    subprocess.run(["git", "add", "Snakefile"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Candidate Test",
            "-c",
            "user.email=candidate@example.invalid",
            "commit",
            "-qm",
            "freeze candidate",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, snakefile


def test_candidate_freeze_records_clean_runtime_identity(tmp_path: Path) -> None:
    commit, snakefile = initialise_repository(tmp_path)

    payload = MODULE.freeze_candidate(
        root=tmp_path,
        expected_commit=commit,
        attempt=2,
        controlled_files=[snakefile],
    )

    assert payload["status"] == "FROZEN"
    assert payload["candidate_commit"] == commit
    assert payload["candidate_attempt"] == 2
    assert payload["dirty_state"]["tracked"] is False
    assert payload["workflow_sha256"] == MODULE.file_hash(snakefile)
    assert payload["runtime_scope"] == "all_tracked_repository_files"


def test_candidate_freeze_rejects_tracked_drift(tmp_path: Path) -> None:
    commit, snakefile = initialise_repository(tmp_path)
    snakefile.write_text("rule all:\n    input: ['changed']\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean tracked tree"):
        MODULE.freeze_candidate(
            root=tmp_path,
            expected_commit=commit,
            attempt=1,
            controlled_files=[snakefile],
        )


def test_candidate_freeze_rejects_untracked_runtime_files(tmp_path: Path) -> None:
    commit, snakefile = initialise_repository(tmp_path)
    untracked = tmp_path / "validation/scripts/untracked.py"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("print('not frozen')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must be tracked"):
        MODULE.freeze_candidate(
            root=tmp_path,
            expected_commit=commit,
            attempt=1,
            controlled_files=[snakefile, untracked],
        )
