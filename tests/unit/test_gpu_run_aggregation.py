from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "aggregate_gpu_comparisons", SCRIPTS / "aggregate_gpu_comparisons.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def initialise_repository(root: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    marker = root / "marker.txt"
    marker.write_text("frozen source\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate@example.invalid",
            "commit",
            "-qm",
            "freeze source",
        ],
        cwd=root,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_runs(
    root: Path,
    *,
    datasets: list[str],
    commit: str,
) -> list[tuple[Path, Path]]:
    runs: list[tuple[Path, Path]] = []
    for dataset in datasets:
        for repeat in range(1, 6):
            stem = f"{dataset}.repeat{repeat}"
            parquet = root / f"{stem}.parquet"
            frame = pd.DataFrame.from_records(
                [
                    {
                        "dataset_id": dataset,
                        "pair_id": f"{dataset}:pair-1",
                        "repeat": repeat,
                        "profile_id": "bins128_depth16_streams1",
                    }
                ]
            )
            frame.to_parquet(parquet, index=False)
            summary = root / f"{stem}.json"
            summary.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "source_commit": commit,
                        "profile_id": "bins128_depth16_streams1",
                        "profile": {
                            "n_bins": 128,
                            "max_depth": 16,
                            "n_streams": 1,
                        },
                        "profile_evidence": None,
                        "repeat_indices": [repeat],
                        "row_count": len(frame),
                        "results_parquet_sha256": MODULE.file_hash(parquet),
                        "inputs": [
                            {
                                "dataset_id": dataset,
                                "h5ad_sha256": f"h5ad-{dataset}",
                            }
                        ],
                        "slurm": {"job_id": f"{dataset}-{repeat}"},
                        "device": {
                            "name": "NVIDIA L40S",
                            "compute_capability": [8, 9],
                            "free_bytes_after": 40_000,
                            "total_bytes": 48_000,
                        },
                        "environment": {"python": "3.12.11", "cuml": "26.06"},
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            runs.append((summary, parquet))
    return runs


def test_aggregate_runs_requires_complete_five_repeat_l40s_grid(
    tmp_path: Path,
) -> None:
    commit = initialise_repository(tmp_path)
    datasets = ["Dataset A", "Dataset B"]
    runs = make_runs(tmp_path, datasets=datasets, commit=commit)

    payload = MODULE.aggregate_runs(
        root=tmp_path,
        runs=runs,
        expected_datasets=datasets,
        expected_repeats=5,
        output_parquet=tmp_path / "combined.parquet",
        output_summary=tmp_path / "combined.json",
    )

    assert payload["status"] == "PASS"
    assert payload["row_count"] == 10
    assert payload["repeat_indices"] == [1, 2, 3, 4, 5]
    assert payload["device"]["all_chunks_typed_l40s"] is True
    assert payload["slurm"]["chunk_count"] == 10
    combined = pd.read_parquet(tmp_path / "combined.parquet")
    assert (
        combined.groupby(["dataset_id", "pair_id"])["repeat"].apply(list).tolist()
        == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    )


def test_aggregate_runs_rejects_non_l40s_shard(tmp_path: Path) -> None:
    commit = initialise_repository(tmp_path)
    runs = make_runs(tmp_path, datasets=["Dataset A"], commit=commit)
    summary_path, _ = runs[0]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["device"]["name"] = "MIG 1g.24gb"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(RuntimeError, match="typed L40S"):
        MODULE.aggregate_runs(
            root=tmp_path,
            runs=runs,
            expected_datasets=["Dataset A"],
            expected_repeats=5,
            output_parquet=tmp_path / "combined.parquet",
            output_summary=tmp_path / "combined.json",
        )
