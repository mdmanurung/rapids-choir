#!/usr/bin/env python3
"""Run one frozen RF profile for five same-seed L40S repeats."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from gate2_common import (
    file_hash,
    immutable_json,
    install_immutable_file,
    validate_candidate_manifest,
)
from resource_ledger import validate_reservation

from rapids_choir._comparator import ChoirComparator
from rapids_choir._cuda import initialize_cuda
from rapids_choir._enums import REFERENCE_REASON_ALIASES
from rapids_choir.config import ComparisonConfig, RFProfile

SUPPORTED_BOUNDED_METRICS = (
    "mean_observed_accuracy",
    "variance_observed_accuracy",
    "mean_null_accuracy",
    "variance_null_accuracy",
    "accuracy_reference_tail_probability",
    "variance_reference_tail_probability",
)
RUN_STAGE = "preflight"


def _cli_path(name: str) -> Path | None:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        return None
    return Path(sys.argv[index + 1])


def _preserve_failure(error: Exception) -> None:
    output = _cli_path("--output-summary")
    if output is None:
        return
    message = str(error).lower()
    if RUN_STAGE == "preflight" and any(
        token in message for token in ("l40s", "cuda device", "nvidia")
    ):
        classification = "infrastructure_pre_method"
    elif RUN_STAGE == "preflight":
        classification = "custody_or_preflight_rejection"
    else:
        classification = "technical_outcome"
    payload = {
        "schema_version": 1,
        "status": "FAIL",
        "scope": "gpu_comparator_shard_failure",
        "generated_at": datetime.now(UTC).isoformat(),
        "stage": RUN_STAGE,
        "method_started": RUN_STAGE == "method",
        "failure_classification": classification,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node": os.environ.get("SLURMD_NODENAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "gres": os.environ.get("SLURM_JOB_GRES"),
        },
        "candidate_manifest": (
            str(path.resolve())
            if (path := _cli_path("--candidate-manifest")) is not None
            else None
        ),
        "resource_reservation": (
            str(path.resolve())
            if (path := _cli_path("--resource-reservation")) is not None
            else None
        ),
    }
    failure = output.with_name(output.stem + ".FAILED.json")
    try:
        immutable_json(failure, payload)
    except Exception as preservation_error:  # pragma: no cover - last resort
        print(
            f"could not preserve GPU failure record: {preservation_error}",
            file=sys.stderr,
        )


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
    global RUN_STAGE
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--max-depth")
    parser.add_argument("--n-streams", type=int)
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--comparator-contract", required=True, type=Path)
    parser.add_argument("--authorization-seal", type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--resource-reservation", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--repeat-index", type=int)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    candidate = validate_candidate_manifest(root, args.candidate_manifest)

    contract = yaml.safe_load(
        args.comparator_contract.read_text(encoding="utf-8")
    )
    bounded_metrics = tuple(contract["reference_variability"]["bounded_fields"])
    if bounded_metrics != SUPPORTED_BOUNDED_METRICS:
        raise RuntimeError(
            "comparator contract bounded metrics are unsupported or reordered"
        )

    if args.repeat_index is None:
        repeats = 5 if args.repeats is None else args.repeats
        if repeats != 5:
            raise RuntimeError("Gate 2 requires exactly five GPU repeats")
        repeat_indices = list(range(1, repeats + 1))
    else:
        if args.repeats is not None:
            raise RuntimeError("--repeats and --repeat-index are mutually exclusive")
        if args.repeat_index not in range(1, 6):
            raise RuntimeError("--repeat-index must be one of 1, 2, 3, 4, or 5")
        repeat_indices = [args.repeat_index]
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Gate 2 GPU comparisons must run inside Slurm")
    if args.profile_json is not None:
        expected_phase = "comparator_confirm"
        profile_payload = json.loads(args.profile_json.read_text(encoding="utf-8"))
        if profile_payload.get("status") != "FROZEN":
            raise RuntimeError("holdout profile manifest is not FROZEN")
        if profile_payload.get("comparator_contract", {}).get(
            "sha256"
        ) != file_hash(args.comparator_contract):
            raise RuntimeError("profile and comparator contract checksums differ")
        if args.authorization_seal is None:
            raise RuntimeError("frozen-profile runs require an authorization seal")
        authorization = json.loads(
            args.authorization_seal.read_text(encoding="utf-8")
        )
        if authorization.get("status") != "AUTHORIZED" or authorization.get(
            "target"
        ) != "comparator_confirm":
            raise RuntimeError("comparator confirmation is not authorized")
        if authorization.get("candidate_commit") != git_revision(root):
            raise RuntimeError("authorization candidate does not match HEAD")
        if authorization.get("runtime_tree_sha256") != candidate.get(
            "runtime_tree_sha256"
        ):
            raise RuntimeError("authorization runtime does not match candidate")
        if authorization.get("attempt") != candidate.get("candidate_attempt"):
            raise RuntimeError("authorization attempt does not match candidate")
        if authorization.get("rf_profile_sha256") != file_hash(args.profile_json):
            raise RuntimeError("authorization RF profile checksum does not match")
        expires = datetime.fromisoformat(str(authorization["expires_at"]))
        if expires.astimezone(UTC) <= datetime.now(UTC):
            raise RuntimeError("comparator authorization has expired")
        profile = RFProfile.from_mapping(profile_payload["winner"]["profile"])
    else:
        expected_phase = "comparator_development"
        if args.authorization_seal is not None:
            raise RuntimeError(
                "development profile runs cannot consume confirmatory authorization"
            )
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
    reservation = validate_reservation(
        ledger=args.resource_ledger,
        reservation_path=args.resource_reservation,
        phase=expected_phase,
        candidate_commit=str(candidate["candidate_commit"]),
        candidate_attempt=int(candidate["candidate_attempt"]),
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
    RUN_STAGE = "input_ingest"
    for dataset_value in args.dataset:
        dataset = parse_dataset(dataset_value)
        h5ad_path = Path(dataset["h5ad"])
        pair_path = Path(dataset["pairs"])
        reference_path = Path(dataset["reference"])
        data_manifest_path = Path(dataset["data_manifest"])
        h5ad_hash = file_hash(h5ad_path)
        data_manifest_hash = file_hash(data_manifest_path)
        pair_hash = file_hash(pair_path)
        data_manifest = json.loads(
            data_manifest_path.read_text(encoding="utf-8")
        )
        pair_manifest = json.loads(pair_path.read_text(encoding="utf-8"))
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        candidate_identity = {
            "commit": candidate["candidate_commit"],
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": file_hash(args.candidate_manifest),
            "dirty_state": candidate["dirty_state"],
        }
        if (
            data_manifest.get("status") != "PASS"
            or data_manifest.get("dataset_id") != dataset["id"]
            or Path(str(data_manifest.get("path"))).resolve()
            != h5ad_path.resolve()
            or data_manifest.get("sha256") != h5ad_hash
            or data_manifest.get("candidate") != candidate_identity
        ):
            raise RuntimeError(
                f"H5AD manifest identity mismatch: {data_manifest_path}"
            )
        if (
            pair_manifest.get("status") != "FROZEN"
            or pair_manifest.get("dataset_id") != dataset["id"]
            or pair_manifest.get("h5ad_sha256") != h5ad_hash
            or pair_manifest.get("data_manifest_sha256") != data_manifest_hash
            or pair_manifest.get("candidate") != candidate_identity
        ):
            raise RuntimeError(f"pair manifest identity mismatch: {pair_path}")
        if reference.get("status") != "PASS" or reference.get(
            "dataset_id"
        ) != dataset["id"]:
            raise RuntimeError(f"reference identity mismatch: {reference_path}")
        if (
            reference.get("h5ad_sha256") != h5ad_hash
            or reference.get("pair_manifest_sha256") != pair_hash
            or reference.get("candidate") != candidate_identity
        ):
            raise RuntimeError(
                f"reference input or candidate checksum mismatch: {reference_path}"
            )
        reference_by_pair = {
            result["pair_id"]: result for result in reference["results"]
        }
        pair_by_id = {pair["pair_id"]: pair for pair in pair_manifest["pairs"]}
        if set(reference_by_pair) != set(pair_by_id):
            raise RuntimeError(
                f"reference pair set does not match manifest: {dataset['id']}"
            )
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
                "h5ad_sha256": h5ad_hash,
                "data_manifest_sha256": data_manifest_hash,
                "pair_manifest_sha256": pair_hash,
                "reference_sha256": file_hash(reference_path),
                "initial_host_to_device_ingest_bytes": ingest_bytes,
            }
        )
        for repeat_index in repeat_indices:
            config = ComparisonConfig(
                rules="choir_v0_3_0",
                input_policy="device_only",
                random_state=1,
                rf_profile=profile,
                rf_profile_manifest=(
                    str(args.profile_json) if args.profile_json is not None else None
                ),
            )
            comparator = ChoirComparator(config)
            for pair in pair_manifest["pairs"]:
                first, second = memberships[pair["pair_id"]]
                RUN_STAGE = "method"
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
                if (
                    reference_result.get("pair_kind") != pair["kind"]
                    or reference_result.get("expected_decision")
                    != pair["expected_decision"]
                ):
                    raise RuntimeError(
                        f"reference metadata drift: {dataset['id']}/{pair['pair_id']}"
                    )
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
                        "reference_pair_kind": reference_result["pair_kind"],
                        "reference_expected_decision": reference_result[
                            "expected_decision"
                        ],
                        "gpu_decision": result.decision.value,
                        "gpu_reason": result.reason_code.value,
                        "gpu_reference_reason": REFERENCE_REASON_ALIASES[
                            result.reason_code
                        ],
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
                for field in bounded_metrics:
                    rows[-1][f"reference_{field}"] = reference_result.get(field)
                    rows[-1][f"gpu_{field}"] = (
                        getattr(result.metrics, field)
                        if result.metrics is not None
                        else None
                    )
            del comparator
        del memberships, feature_mask, graph, features, adata
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()

    frame = pd.DataFrame.from_records(rows)
    output_parquet = args.output_parquet
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_parquet.with_suffix(
        output_parquet.suffix + f".tmp.{os.getpid()}"
    )
    frame.to_parquet(temporary, index=False)
    install_immutable_file(temporary, output_parquet)
    free_bytes, total_bytes = context.memory_info()
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "failure_classification": None,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": git_revision(root),
        "candidate": {
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": file_hash(args.candidate_manifest),
            "dirty_state": candidate["dirty_state"],
        },
        "resource_reservation": {
            "reservation_id": reservation["reservation_id"],
            "sha256": file_hash(args.resource_reservation),
        },
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
        "comparator_contract": {
            "path": str(args.comparator_contract.resolve()),
            "sha256": file_hash(args.comparator_contract),
        },
        "authorization": (
            {
                "path": str(args.authorization_seal.resolve()),
                "sha256": file_hash(args.authorization_seal),
            }
            if args.authorization_seal is not None
            else None
        ),
        "repeats": len(repeat_indices),
        "repeat_indices": repeat_indices,
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
    immutable_json(args.output_summary, summary)
    RUN_STAGE = "complete"


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _preserve_failure(error)
        raise
