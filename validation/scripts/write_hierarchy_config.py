#!/usr/bin/env python3
"""Create one immutable compatibility-mode config for a hierarchy object."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rapids_choir.config import RFProfile
from validation.scripts.gate2_common import immutable_json


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(contract: dict[str, Any], dataset_id: str) -> dict[str, Any]:
    matches = [
        record
        for record in contract.get("objects", [])
        if record.get("dataset_id") == dataset_id
    ]
    if len(matches) != 1:
        raise ValueError("dataset ID must select one hierarchy development object")
    return dict(matches[0])


def build_config(
    object_contract_path: Path,
    dataset_id: str,
    profile_path: Path,
    object_manifest_path: Path,
) -> dict[str, Any]:
    contract = json.loads(object_contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != 1
        or contract.get("evidence_role") != "development_only"
        or contract.get("promotable") is not False
    ):
        raise ValueError("hierarchy development object contract is invalid")
    record = _object(contract, dataset_id)
    object_manifest = json.loads(
        object_manifest_path.read_text(encoding="utf-8")
    )
    if (
        object_manifest.get("status") != "GENERATED"
        or object_manifest.get("dataset_id") != dataset_id
        or object_manifest.get("role") != record.get("role")
        or object_manifest.get("evidence_role") != "development_only"
        or object_manifest.get("promotable") is not False
        or object_manifest.get("contract_sha256") != _sha256(object_contract_path)
    ):
        raise ValueError("hierarchy development object manifest has drifted")
    artifacts = object_manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"recipe", "h5ad"}:
        raise ValueError("hierarchy development artifacts are incomplete")
    for artifact in artifacts.values():
        if not isinstance(artifact, dict):
            raise ValueError("hierarchy development artifact manifest is invalid")
        path = Path(str(artifact.get("path")))
        if not path.is_file() or artifact.get("sha256") != _sha256(path):
            raise ValueError("hierarchy development artifact is stale")
    defaults = contract.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("hierarchy development defaults are missing")

    def setting(name: str) -> Any:
        return record.get(name, defaults.get(name))

    profile_manifest = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile_manifest.get("status") != "FROZEN":
        raise ValueError("hierarchy execution requires a frozen RF profile")
    raw_profile = profile_manifest.get("winner", {}).get("profile")
    if not isinstance(raw_profile, dict):
        raise ValueError("frozen RF profile has no canonical winner")
    profile = RFProfile.from_mapping(raw_profile)
    random_state = int(contract["reference_seeds"][0])
    multiplicity = str(setting("multiplicity"))
    return {
        "key_added": "choir",
        "layer": "log1p",
        "use_rep": "X_pca",
        "nn_graph_key": "choir_nn",
        "snn_graph_key": "choir_snn",
        "feature_mask_key": "highly_variable",
        "batch_key": None,
        "preprocessing": {
            "counts_layer": "counts",
            "normalized_layer": "log1p",
            "target_sum": 10000.0,
            "hvg_flavor": "seurat_v3",
            "n_top_genes": int(setting("n_top_genes")),
            "n_pcs": int(setting("n_pcs")),
            "n_neighbors": int(setting("n_neighbors")),
        },
        "rules": "choir_v0_3_0",
        "multiplicity": multiplicity,
        "max_clusters": setting("max_clusters"),
        "min_cluster_depth": int(setting("min_cluster_depth")),
        "group_singletons": bool(setting("group_singletons")),
        "max_resolution_steps": 512,
        "input_policy": "transfer_once",
        "inplace": True,
        "overwrite": False,
        "device": "cuda:0",
        "vram_headroom": 0.2,
        "random_state": random_state,
        "comparison": {
            "alpha": 0.05,
            "n_iterations": 100,
            "n_trees": 50,
            "use_variance": True,
            "min_accuracy": 0.5,
            "min_connections": 1.0,
            "max_repeat_errors": 20,
            "sample_max": None,
            "downsampling_rate": 1.0,
            "min_reads": None,
            "rules": "choir_v0_3_0",
            "device": "cuda:0",
            "vram_headroom": 0.2,
            "random_state": random_state,
            "rf_profile": profile.to_record(),
            "rf_profile_manifest": str(profile_path.resolve()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--object-contract", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--rf-profile", required=True, type=Path)
    parser.add_argument("--object-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    immutable_json(
        arguments.output,
        build_config(
            arguments.object_contract,
            arguments.dataset_id,
            arguments.rf_profile,
            arguments.object_manifest,
        ),
    )


if __name__ == "__main__":
    main()
