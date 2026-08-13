from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "check_profile_seal", SCRIPTS / "check_profile_seal.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
EXTRACT_SPEC = importlib.util.spec_from_file_location(
    "extract_datasets", SCRIPTS / "extract_datasets.py"
)
assert EXTRACT_SPEC is not None and EXTRACT_SPEC.loader is not None
EXTRACT_MODULE = importlib.util.module_from_spec(EXTRACT_SPEC)
EXTRACT_SPEC.loader.exec_module(EXTRACT_MODULE)


def run(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_dataset_member_selection_requires_counts_and_metadata(tmp_path: Path) -> None:
    inventory = {
        "members": [
            {
                "name": "Simulated_Datasets/Simulated_Dataset_1/RNA_matrix_raw.rds",
                "type": "file",
                "bytes": 101,
            },
            {
                "name": "Simulated_Datasets/Simulated_Dataset_1/cell_metadata.csv",
                "type": "file",
                "bytes": 102,
            },
            {
                "name": "Simulated_Datasets/Simulated_Dataset_10/RNA_matrix_raw.rds",
                "type": "file",
                "bytes": 201,
            },
            {
                "name": "Simulated_Datasets/Simulated_Dataset_10/cell_metadata.csv",
                "type": "file",
                "bytes": 202,
            },
        ]
    }

    selected = EXTRACT_MODULE.select_members(
        inventory, ["Simulated Dataset 1"], tmp_path
    )

    assert set(selected) == {
        "Simulated_Datasets/Simulated_Dataset_1/RNA_matrix_raw.rds",
        "Simulated_Datasets/Simulated_Dataset_1/cell_metadata.csv",
    }
    assert {record["kind"] for record in selected.values()} == {
        "counts",
        "metadata",
    }
    assert {record["output"].name for record in selected.values()} == {
        "simulated_dataset_001.rds",
        "simulated_dataset_001.metadata.csv",
    }


def test_dataset_member_selection_fails_closed_without_metadata(tmp_path: Path) -> None:
    inventory = {
        "members": [
            {
                "name": "Simulated_Datasets/Simulated_Dataset_1/RNA_matrix_raw.rds",
                "type": "file",
                "bytes": 101,
            }
        ]
    }

    with pytest.raises(RuntimeError, match="exactly one metadata"):
        EXTRACT_MODULE.select_members(inventory, ["Simulated Dataset 1"], tmp_path)


def test_dataset_member_match_rejects_traversal() -> None:
    assert not EXTRACT_MODULE.member_matches(
        "../Simulated_Dataset_1/RNA_matrix_raw.rds",
        "Simulated Dataset 1",
        "counts",
    )


def test_holdout_profile_must_be_tracked_committed_and_unchanged(
    tmp_path: Path,
) -> None:
    run(tmp_path, "init", "-q")
    profile = tmp_path / "validation/evidence/rf_profile.json"
    profile.parent.mkdir(parents=True)
    payload = {"status": "FROZEN", "winner": {"profile": {"n_bins": 128}}}
    profile.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        MODULE.verify_profile_seal(tmp_path, profile)

    run(tmp_path, "add", profile.relative_to(tmp_path).as_posix())
    run(
        tmp_path,
        "-c",
        "user.name=Gate Test",
        "-c",
        "user.email=gate@example.invalid",
        "commit",
        "-qm",
        "freeze profile",
    )
    evidence = MODULE.verify_profile_seal(tmp_path, profile)
    assert evidence["status"] == "PASS"
    seal = tmp_path / "rf_profile.sealed.json"
    seal.write_text(json.dumps(evidence), encoding="utf-8")
    assert EXTRACT_MODULE.validate_holdout_seal(tmp_path, seal) == evidence

    profile.write_text(json.dumps({**payload, "changed": True}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted or modified"):
        MODULE.verify_profile_seal(tmp_path, profile)

    profile.write_text(json.dumps(payload), encoding="utf-8")
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("new head\n", encoding="utf-8")
    run(tmp_path, "add", "unrelated.txt")
    run(
        tmp_path,
        "-c",
        "user.name=Gate Test",
        "-c",
        "user.email=gate@example.invalid",
        "commit",
        "-qm",
        "advance head",
    )
    with pytest.raises(RuntimeError, match="stale"):
        EXTRACT_MODULE.validate_holdout_seal(tmp_path, seal)
