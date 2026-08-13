#!/usr/bin/env python3
"""Run one frozen RF profile for five same-seed L40S repeats."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gate2_common import atomic_json, file_hash

from rapids_choir._comparator import ChoirComparator
from rapids_choir._cuda import initialize_cuda
from rapids_choir.config import ComparisonConfig, RFProfile


def parse_dataset(value: str) -> dict[str, Path | str]:
    fields = value.split("\t")
    if len(fields) != 5:
        raise ValueError(
            "--dataset requires ID<TAB>H5AD<TAB>data manifest<TAB>pairs<TAB>reference"
        )
    return {
        "id": fields[0],
        "h5ad": Path(fields[1]),
        "data_manifest": Path(fields[2]),
        "pairs": Path(fields[3]),
        "reference": Path(fields[4]),
    }


def to_device_csr(matrix: Any, cp: Any, cps: Any) -> Any:
    host = sp.csr_matrix(matrix, dtype=np.float32)
    return cps.csr_matrix(
        (
            cp.asarray(host.data, dtype=cp.float32),
            cp.asarray(host.indices, dtype=cp.int32),
            cp.asarray(host.indptr, dtype=cp.int32),
        ),
        shape=host.shape,
    )


def source_bytes(matrix: Any) -> int:
    if sp.issparse(matrix):
        csr = matrix.tocsr(copy=False)
        return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
    return int(np.asarray(matrix).nbytes)


def git_revision(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--max-depth")
    parser.add_argument("--n-streams", type=int)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--repeats", default=5, type=int)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    args = parser.parse_args()

    if args.repeats != 5:
        raise RuntimeError("Gate 2 requires exactly five GPU repeats")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 2 GPU comparisons must run inside Slurm")
    if args.profile_json is not None:
        profile_payload = json.loads(args.profile_json.read_text(encoding="utf-8"))
        profile = RFProfile.from_mapping(profile_payload["winner"]["profile"])
    else:
        if args.n_bins is None or args.max_depth is None or args.n_streams is None:
            raise RuntimeError(
                "explicit n_bins/max_depth/n_streams or --profile-json is required"
            )
        max_depth = (
            None if args.max_depth.lower() == "none" else int(args.max_depth)
        )
        profile = RFProfile(
            n_bins=args.n_bins,
            max_depth=max_depth,
            n_streams=args.n_streams,
        )
    profile_id = (
        f"bins{profile.n_bins}_depth"
        f"{'none' if profile.max_depth is None else profile.max_depth}_"
        f"streams{profile.n_streams}"
    )
    context = initialize_cuda("cuda:0")
    if "L40S" not in context.device_name:
        raise RuntimeError(f"Gate 2 requires a typed L40S, got {context.device_name}")
    cp = context.cupy
    cps = __import__("cupyx.scipy.sparse", fromlist=["sparse"])

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for dataset_value in args.dataset:
        dataset = parse_dataset(dataset_value)
        h5ad_path = Path(dataset["h5ad"])
        pair_path = Path(dataset["pairs"])
        reference_path = Path(dataset["reference"])
        data_manifest_path = Path(dataset["data_manifest"])
        pair_manifest = json.loads(pair_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_by_pair = {
            result["pair_id"]: result for result in reference["results"]
        }
        adata = ad.read_h5ad(h5ad_path)
        ingest_bytes = source_bytes(adata.X) + source_bytes(
            adata.obsp["choir_nn"]
        )
        features = to_device_csr(adata.X, cp, cps)
        graph = to_device_csr(adata.obsp["choir_nn"], cp, cps)
        feature_mask = cp.asarray(
            adata.var["highly_variable"].to_numpy(dtype=bool, copy=False),
            dtype=cp.bool_,
        )
        name_to_position = {
            str(name): index for index, name in enumerate(adata.obs_names)
        }
        memberships = {
            pair["pair_id"]: (
                cp.asarray(
                    [name_to_position[name] for name in pair["cluster1_cells"]],
                    dtype=cp.int64,
                ),
                cp.asarray(
                    [name_to_position[name] for name in pair["cluster2_cells"]],
                    dtype=cp.int64,
                ),
            )
            for pair in pair_manifest["pairs"]
        }
        ingest_bytes += int(feature_mask.nbytes) + sum(
            int(first.nbytes + second.nbytes)
            for first, second in memberships.values()
        )
        inputs.append(
            {
                "dataset_id": dataset["id"],
                "h5ad_sha256": file_hash(h5ad_path),
                "data_manifest_sha256": file_hash(data_manifest_path),
                "pair_manifest_sha256": file_hash(pair_path),
                "reference_sha256": file_hash(reference_path),
                "initial_host_to_device_ingest_bytes": ingest_bytes,
            }
        )
        for repeat_index in range(1, args.repeats + 1):
            config = ComparisonConfig(
                rules="choir_v0_3_0",
                input_policy="device_only",
                random_state=1,
                rf_profile=profile,
            )
            comparator = ChoirComparator(config)
            for pair in pair_manifest["pairs"]:
                first, second = memberships[pair["pair_id"]]
                result = comparator.compare(
                    features,
                    first,
                    second,
                    graph,
                    feature_mask=feature_mask,
                    cluster1_id=pair["cluster1_label"],
                    cluster2_id=pair["cluster2_label"],
                )
                reference_result = reference_by_pair[pair["pair_id"]]
                unapproved_transfer_bytes = sum(
                    transfer.bytes
                    for transfer in result.memory.transfers
                    if not transfer.approved
                )
                rows.append(
                    {
                        "dataset_id": dataset["id"],
                        "pair_id": pair["pair_id"],
                        "pair_kind": pair["kind"],
                        "expected_decision": pair["expected_decision"],
                        "profile_id": profile_id,
                        "repeat": repeat_index,
                        "reference_decision": reference_result["decision"],
                        "reference_reason": reference_result["reason"],
                        "reference_mean_observed_accuracy": reference_result[
                            "mean_observed_accuracy"
                        ],
                        "gpu_decision": result.decision.value,
                        "gpu_reason": result.reason_code.value,
                        "gpu_mean_observed_accuracy": (
                            result.metrics.mean_observed_accuracy
                            if result.metrics is not None
                            else None
                        ),
                        "runtime_seconds": result.timings.total_seconds,
                        "forest_fits": result.samples.forest_fits,
                        "estimated_device_bytes": result.memory.estimated_bytes,
                        "peak_device_bytes": result.memory.peak_device_bytes,
                        "free_bytes_at_preflight": (
                            result.memory.free_bytes_at_preflight
                        ),
                        "unapproved_post_ingest_transfer_bytes": (
                            unapproved_transfer_bytes
                        ),
                    }
                )
            del comparator
        del memberships, feature_mask, graph, features, adata
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()

    frame = pd.DataFrame.from_records(rows)
    output_parquet = args.output_parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parquet.with_suffix(output_parquet.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output_parquet)
    free_bytes, total_bytes = context.memory_info()
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": git_revision(args.root),
        "profile_id": profile_id,
        "profile": profile.to_record(),
        "profile_evidence": (
            {
                "path": str(args.profile_json.resolve()),
                "sha256": file_hash(args.profile_json),
            }
            if args.profile_json is not None
            else None
        ),
        "repeats": args.repeats,
        "random_state": 1,
        "rules": "choir_v0_3_0",
        "row_count": len(frame),
        "results_parquet": str(output_parquet.resolve()),
        "results_parquet_sha256": file_hash(output_parquet),
        "inputs": inputs,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gres": os.environ.get("SLURM_JOB_GRES"),
        },
        "device": {
            "name": context.device_name,
            "compute_capability": list(context.compute_capability),
            "free_bytes_after": free_bytes,
            "total_bytes": total_bytes,
        },
        "environment": {
            "python": sys.version.split()[0],
            "python_executable": str(Path(sys.executable).resolve()),
            "cupy": cp.__version__,
            "cuml": __import__("cuml").__version__,
        },
    }
    atomic_json(args.output_summary, summary)


if __name__ == "__main__":
    main()
