from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rapids_choir.config import RFProfile
from validation.scripts.run_python_semantic_trace import (
    _validate_rf_profile_config,
)
from validation.scripts.write_hierarchy_config import build_config

ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hierarchy_config_binds_generated_object_and_frozen_profile(
    tmp_path: Path,
) -> None:
    contract = ROOT / "validation/contracts/hierarchy_development_objects_v1.json"
    recipe = tmp_path / "object.rds"
    h5ad = tmp_path / "object.h5ad"
    recipe.write_bytes(b"recipe")
    h5ad.write_bytes(b"h5ad")
    manifest = tmp_path / "object.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "GENERATED",
                "dataset_id": "splatter-development-01",
                "role": "starting-resolution-up",
                "evidence_role": "development_only",
                "promotable": False,
                "contract_sha256": _sha256(contract),
                "artifacts": {
                    "recipe": {"path": str(recipe), "sha256": _sha256(recipe)},
                    "h5ad": {"path": str(h5ad), "sha256": _sha256(h5ad)},
                },
            }
        ),
        encoding="utf-8",
    )
    profile = tmp_path / "rf_profile.json"
    profile.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "winner": {"profile": RFProfile().to_record()},
            }
        ),
        encoding="utf-8",
    )
    config = build_config(
        contract,
        "splatter-development-01",
        profile,
        manifest,
    )
    assert config["rules"] == "choir_v0_3_0"
    assert config["random_state"] == 104729
    assert config["comparison"]["rf_profile"] == RFProfile().to_record()
    _validate_rf_profile_config(config)

    config["comparison"]["rf_profile"]["n_bins"] = 256
    with pytest.raises(ValueError, match="has drifted"):
        _validate_rf_profile_config(config)
