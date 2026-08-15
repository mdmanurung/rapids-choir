#!/usr/bin/env python3
"""Run validation-only Python hierarchy tracing on one prepared AnnData."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anndata as ad
from gate2_common import file_hash, validate_candidate_manifest
from resource_ledger import validate_reservation

from rapids_choir._enums import InputPolicy
from rapids_choir._hashing import anndata_input_hashes, canonical_json_bytes
from rapids_choir._semantic_trace import (
    TraceRecorder,
    semantic_trace_reference,
)
from rapids_choir.config import ChoirConfig, ComparisonConfig, RFProfile, RNAConfig
from rapids_choir.tl._choir import (
    _choir_impl,
    _prepare_if_requested,
    _resolve_choir_config,
    _run_identity,
)

RESULT_SCHEMA = "rapids_choir.hierarchy-result/v1"


def _write_immutable_json(path: Path, payload: dict[str, Any], identity: str) -> None:
    if path.exists():
        raise FileExistsError(f"immutable JSON artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{identity}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary JSON artifact exists: {temporary}")
    try:
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"immutable JSON artifact already exists: {path}"
            ) from error
    finally:
        if temporary.exists():
            temporary.unlink()


def _config(payload: dict[str, Any]) -> ChoirConfig:
    values = dict(payload)
    raw_preprocessing = values.get("preprocessing")
    if raw_preprocessing is not None:
        values["preprocessing"] = RNAConfig(**raw_preprocessing)
    raw_comparison = values.get("comparison")
    if raw_comparison is not None:
        comparison_values = dict(raw_comparison)
        raw_profile = comparison_values.get("rf_profile")
        if raw_profile is not None:
            comparison_values["rf_profile"] = RFProfile(**raw_profile)
        comparison_values["input_policy"] = InputPolicy.DEVICE_ONLY
        values["comparison"] = ComparisonConfig(**comparison_values)
    return ChoirConfig(**values)


def _validate_rf_profile_config(payload: dict[str, Any]) -> None:
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        raise ValueError("semantic hierarchy config lacks comparator settings")
    manifest_value = comparison.get("rf_profile_manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise ValueError("semantic hierarchy config lacks frozen RF profile")
    manifest_path = Path(manifest_value).expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError("semantic hierarchy RF-profile manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") != "FROZEN"
        or not isinstance(manifest.get("winner"), dict)
        or comparison.get("rf_profile") != manifest["winner"].get("profile")
    ):
        raise ValueError("semantic hierarchy RF profile has drifted")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--resource-reservation", required=True, type=Path)
    parser.add_argument("--resource-ledger", required=True, type=Path)
    parser.add_argument("--resource-phase", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-result", required=True, type=Path)
    parser.add_argument("--output-parquet", type=Path)
    parser.add_argument("--parquet-manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.output_jsonl.exists():
        raise FileExistsError(
            f"semantic trace already exists: {arguments.output_jsonl}"
        )
    if not arguments.dataset_id:
        raise ValueError("--dataset-id cannot be empty")
    if arguments.output_result.exists():
        raise FileExistsError(
            f"hierarchy result already exists: {arguments.output_result}"
        )
    if (arguments.output_parquet is None) != (arguments.parquet_manifest is None):
        raise ValueError(
            "--output-parquet and --parquet-manifest must be supplied together"
        )
    if arguments.parquet_manifest is not None and arguments.parquet_manifest.exists():
        raise FileExistsError(
            f"semantic trace manifest exists: {arguments.parquet_manifest}"
        )

    adata = ad.read_h5ad(arguments.h5ad)
    candidate = validate_candidate_manifest(
        arguments.root.resolve(), arguments.candidate_manifest
    )
    reservation = validate_reservation(
        ledger=arguments.resource_ledger,
        reservation_path=arguments.resource_reservation,
        phase=arguments.resource_phase,
        candidate_commit=str(candidate["candidate_commit"]),
        candidate_attempt=int(candidate["candidate_attempt"]),
    )
    config_payload = json.loads(arguments.config.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("semantic hierarchy config must be a JSON object")
    _validate_rf_profile_config(config_payload)
    config = _resolve_choir_config(_config(config_payload))
    preprocessed = config.preprocessing is not None
    if preprocessed:
        _prepare_if_requested(adata, config)
    input_sha256, _ = anndata_input_hashes(
        adata,
        layer=config.layer,
        use_rep=config.use_rep,
        nn_graph_key=config.nn_graph_key,
        snn_graph_key=config.snn_graph_key,
        feature_mask_key=config.feature_mask_key,
        batch_key=config.batch_key,
    )
    recorder = TraceRecorder(_run_identity(input_sha256, config))
    result = _choir_impl(
        adata,
        config,
        trace=recorder,
        preprocessed=preprocessed,
    )
    recorder.write_jsonl(arguments.output_jsonl)
    if arguments.output_parquet is not None:
        manifest = recorder.write_parquet(arguments.output_parquet)
        assert arguments.parquet_manifest is not None
        _write_immutable_json(
            arguments.parquet_manifest,
            asdict(manifest),
            result.provenance["run_id"],
        )
    trace_reference = semantic_trace_reference(
        arguments.output_jsonl,
        format="jsonl",
        relative_to=arguments.output_result.parent,
    )
    provenance = dict(result.provenance)
    provenance["resource_reservation"] = {
        "reservation_id": reservation["reservation_id"],
        "sha256": file_hash(arguments.resource_reservation),
        "phase": arguments.resource_phase,
    }
    payload = {
        "schema_version": RESULT_SCHEMA,
        "candidate": {
            "commit": candidate["candidate_commit"],
            "attempt": candidate["candidate_attempt"],
            "runtime_tree_sha256": candidate["runtime_tree_sha256"],
            "manifest_sha256": file_hash(arguments.candidate_manifest),
        },
        "dataset_id": arguments.dataset_id,
        "run_id": result.provenance["run_id"],
        "cell_ids": list(result.tree.cell_ids),
        "labels": result.labels.tolist(),
        "tree": result.tree.to_record(),
        "provenance": provenance,
        "semantic_trace": trace_reference,
    }
    _write_immutable_json(
        arguments.output_result,
        payload,
        result.provenance["run_id"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
