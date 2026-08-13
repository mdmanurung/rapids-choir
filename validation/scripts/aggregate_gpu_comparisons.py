#!/usr/bin/env python3
"""Combine dataset-by-repeat L40S shards into one governed GPU run."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from gate2_common import atomic_json, file_hash


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_run(value: str) -> tuple[Path, Path]:
    fields = value.split("\t")
    if len(fields) != 2:
        raise ValueError("--run requires SUMMARY<TAB>PARQUET")
    return Path(fields[0]), Path(fields[1])


def _same_json(left: object, right: object) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def aggregate_runs(
    *,
    root: Path,
    runs: list[tuple[Path, Path]],
    expected_datasets: list[str],
    expected_repeats: int,
    output_parquet: Path,
    output_summary: Path,
) -> dict[str, Any]:
    if len(expected_datasets) != len(set(expected_datasets)):
        raise RuntimeError("expected dataset identifiers are not unique")
    if expected_repeats != 5:
        raise RuntimeError("Gate 2 aggregation requires exactly five repeats")
    expected_grid = {
        (dataset_id, repeat)
        for dataset_id in expected_datasets
        for repeat in range(1, expected_repeats + 1)
    }
    if len(runs) != len(expected_grid):
        raise RuntimeError(
            f"expected {len(expected_grid)} GPU shards, got {len(runs)}"
        )

    frames: list[pd.DataFrame] = []
    component_grid: set[tuple[str, int]] = set()
    provenance: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}
    source_commits: set[str] = set()
    profile_id: str | None = None
    profile: dict[str, Any] | None = None
    profile_evidence: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    device_names: set[str] = set()
    compute_capabilities: set[tuple[int, ...]] = set()
    free_after: list[int] = []
    total_bytes: list[int] = []

    for summary_path, parquet_path in runs:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "PASS":
            raise RuntimeError(f"GPU shard is not PASS: {summary_path}")
        if file_hash(parquet_path) != summary.get("results_parquet_sha256"):
            raise RuntimeError(f"GPU shard checksum mismatch: {parquet_path}")
        frame = pd.read_parquet(parquet_path)
        if frame.empty or len(frame) != summary.get("row_count"):
            raise RuntimeError(f"GPU shard row-count mismatch: {parquet_path}")

        datasets = set(frame["dataset_id"].astype(str))
        repeats = {int(value) for value in frame["repeat"]}
        if len(datasets) != 1 or len(repeats) != 1:
            raise RuntimeError("each GPU shard must contain one dataset and one repeat")
        dataset_id = next(iter(datasets))
        repeat = next(iter(repeats))
        if summary.get("repeat_indices") != [repeat]:
            raise RuntimeError("GPU shard repeat metadata does not match its rows")
        component = (dataset_id, repeat)
        if component in component_grid:
            raise RuntimeError(f"duplicate GPU shard: {dataset_id}/repeat-{repeat}")
        component_grid.add(component)

        shard_profile_id = str(summary["profile_id"])
        shard_profile = summary["profile"]
        shard_profile_evidence = summary.get("profile_evidence")
        shard_environment = summary["environment"]
        if profile_id is None:
            profile_id = shard_profile_id
            profile = shard_profile
            profile_evidence = shard_profile_evidence
            environment = shard_environment
        elif (
            shard_profile_id != profile_id
            or not _same_json(shard_profile, profile)
            or not _same_json(shard_profile_evidence, profile_evidence)
            or not _same_json(shard_environment, environment)
        ):
            raise RuntimeError("GPU shard profile or environment drift detected")

        shard_inputs = summary.get("inputs", [])
        if len(shard_inputs) != 1 or shard_inputs[0]["dataset_id"] != dataset_id:
            raise RuntimeError("GPU shard input provenance does not match its dataset")
        if dataset_id in inputs and not _same_json(inputs[dataset_id], shard_inputs[0]):
            raise RuntimeError(f"GPU shard input drift detected for {dataset_id}")
        inputs[dataset_id] = shard_inputs[0]

        source_commits.add(str(summary["source_commit"]))
        device = summary["device"]
        device_name = str(device["name"])
        if "L40S" not in device_name:
            raise RuntimeError(f"GPU shard did not run on typed L40S: {device_name}")
        device_names.add(device_name)
        compute_capabilities.add(tuple(device["compute_capability"]))
        free_after.append(int(device["free_bytes_after"]))
        total_bytes.append(int(device["total_bytes"]))
        provenance.append(
            {
                "dataset_id": dataset_id,
                "repeat": repeat,
                "summary_sha256": file_hash(summary_path),
                "results_parquet_sha256": file_hash(parquet_path),
                "slurm": summary["slurm"],
                "device": device,
            }
        )
        frames.append(frame)

    if component_grid != expected_grid:
        missing = sorted(expected_grid - component_grid)
        unexpected = sorted(component_grid - expected_grid)
        raise RuntimeError(
            f"GPU shard grid mismatch: missing={missing}, extra={unexpected}"
        )
    current_commit = git_revision(root)
    if source_commits != {current_commit}:
        raise RuntimeError(
            "GPU shards do not share the current source commit: "
            f"{sorted(source_commits)}"
        )
    if set(inputs) != set(expected_datasets):
        raise RuntimeError("GPU shard inputs do not cover the expected datasets")
    if profile_id is None or profile is None or environment is None:
        raise RuntimeError("GPU shard aggregation produced no profile metadata")

    combined = pd.concat(frames, ignore_index=True)
    key = ["dataset_id", "pair_id", "repeat"]
    if combined.duplicated(key).any():
        raise RuntimeError("duplicate dataset/pair/repeat rows in GPU shards")
    for (dataset_id, pair_id), group in combined.groupby(
        ["dataset_id", "pair_id"], sort=True
    ):
        observed = sorted(int(value) for value in group["repeat"])
        if observed != list(range(1, expected_repeats + 1)):
            raise RuntimeError(
                f"incomplete repeats for {dataset_id}/{pair_id}: {observed}"
            )

    combined = combined.sort_values(key, kind="stable").reset_index(drop=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parquet.with_suffix(output_parquet.suffix + ".tmp")
    combined.to_parquet(temporary, index=False)
    temporary.replace(output_parquet)
    provenance.sort(key=lambda record: (record["dataset_id"], record["repeat"]))
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": current_commit,
        "profile_id": profile_id,
        "profile": profile,
        "profile_evidence": profile_evidence,
        "repeats": expected_repeats,
        "repeat_indices": list(range(1, expected_repeats + 1)),
        "random_state": 1,
        "rules": "choir_v0_3_0",
        "row_count": len(combined),
        "results_parquet": str(output_parquet.resolve()),
        "results_parquet_sha256": file_hash(output_parquet),
        "inputs": [inputs[dataset_id] for dataset_id in expected_datasets],
        "slurm": {
            "execution_model": "typed_l40s_dataset_repeat_shards",
            "chunk_count": len(provenance),
            "chunk_jobs": provenance,
        },
        "device": {
            "name": next(iter(device_names)) if len(device_names) == 1 else "L40S",
            "compute_capabilities": [
                list(value) for value in sorted(compute_capabilities)
            ],
            "minimum_free_bytes_after": min(free_after),
            "minimum_total_bytes": min(total_bytes),
            "all_chunks_typed_l40s": True,
        },
        "environment": environment,
    }
    atomic_json(output_summary, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--expected-dataset", action="append", required=True)
    parser.add_argument("--expected-repeats", default=5, type=int)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    args = parser.parse_args()
    aggregate_runs(
        root=args.root,
        runs=[parse_run(value) for value in args.run],
        expected_datasets=args.expected_dataset,
        expected_repeats=args.expected_repeats,
        output_parquet=args.output_parquet,
        output_summary=args.output_summary,
    )


if __name__ == "__main__":
    main()
