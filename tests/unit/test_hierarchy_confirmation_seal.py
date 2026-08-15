from __future__ import annotations

import json
from pathlib import Path

import pytest

from validation.scripts import seal_hierarchy_confirmation as module

ROOT = Path(__file__).resolve().parents[2]


def test_hierarchy_confirmation_is_copied_once_under_signed_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_payload = {
        "candidate_commit": "a" * 40,
        "candidate_attempt": 1,
        "runtime_tree_sha256": "b" * 64,
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "validate_candidate_manifest",
        lambda root, path: candidate_payload,
    )
    context = {
        "commit": candidate_payload["candidate_commit"],
        "attempt": 1,
        "runtime_tree_sha256": candidate_payload["runtime_tree_sha256"],
        "manifest_sha256": module.file_hash(candidate_path),
    }
    sources = {
        "h5ad": tmp_path / "source.h5ad",
        "recipe": tmp_path / "source.rds",
        "python_config": tmp_path / "source.config.json",
        "custody_record": tmp_path / "source.custody.json",
    }
    sources["h5ad"].write_bytes(b"h5ad")
    sources["recipe"].write_bytes(b"recipe")
    profile = tmp_path / "rf_profile.json"
    selected_profile = {"n_bins": 128, "max_depth": 16, "n_streams": 1}
    profile.write_text(
        json.dumps(
            {"status": "FROZEN", "winner": {"profile": selected_profile}}
        ),
        encoding="utf-8",
    )
    sources["python_config"].write_text(
        json.dumps(
            {
                "rules": "choir_v0_3_0",
                "device": "cuda:0",
                "input_policy": "transfer_once",
                "comparison": {
                    "rules": "choir_v0_3_0",
                    "device": "cuda:0",
                    "rf_profile": selected_profile,
                    "rf_profile_manifest": str(profile.resolve()),
                },
            }
        ),
        encoding="utf-8",
    )
    sources["custody_record"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "dataset_id": "Arigoni",
                "deviation": "none",
                "recorded_by": "mdmanurung",
                "recorded_at": "2026-08-15T00:00:00+00:00",
                "source_provenance": {},
            }
        ),
        encoding="utf-8",
    )
    data_roles = ROOT / "validation/contracts/data_roles_v2.json"
    margins = tmp_path / "margins.json"
    margins.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "eligible": True,
                "maximum_margin": 0.03,
                "candidate": context,
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "AUTHORIZED",
                "authorization_id": "fixture",
                "target": "hierarchy_confirm_Arigoni",
                "dataset_roles": ["hierarchy_parity_confirmation:Arigoni"],
                "candidate_commit": context["commit"],
                "attempt": 1,
                "runtime_tree_sha256": context["runtime_tree_sha256"],
                "bound_artifacts": {
                    **{
                        name: module.file_hash(path)
                        for name, path in sources.items()
                    },
                    "hierarchy_margins": module.file_hash(margins),
                    "data_roles": module.file_hash(data_roles),
                    "rf_profile": module.file_hash(profile),
                },
            }
        ),
        encoding="utf-8",
    )
    destinations = {
        name: tmp_path / "sealed" / path.name for name, path in sources.items()
    }
    output = tmp_path / "opening.json"
    report = module.seal(
        root=tmp_path,
        candidate_path=candidate_path,
        authorization_path=authorization,
        margins_path=margins,
        data_roles_path=data_roles,
        profile_path=profile,
        dataset_id="Arigoni",
        sources=sources,
        destinations=destinations,
        output_path=output,
    )
    assert report["opening_count"] == 1
    assert report["independent_hierarchy_validation"] is False
    assert Path(report["opening_attempt"]["path"]).is_file()
    assert all(
        destinations[name].read_bytes() == sources[name].read_bytes()
        for name in sources
    )
    with pytest.raises(RuntimeError, match="already been opened"):
        module.seal(
            root=tmp_path,
            candidate_path=candidate_path,
            authorization_path=authorization,
            margins_path=margins,
            data_roles_path=data_roles,
            profile_path=profile,
            dataset_id="Arigoni",
            sources=sources,
            destinations=destinations,
            output_path=output,
        )


def test_failed_hierarchy_opening_is_preserved_and_cannot_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_payload = {
        "candidate_commit": "a" * 40,
        "candidate_attempt": 1,
        "runtime_tree_sha256": "b" * 64,
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "validate_candidate_manifest",
        lambda root, path: candidate_payload,
    )
    context = {
        "commit": candidate_payload["candidate_commit"],
        "attempt": 1,
        "runtime_tree_sha256": candidate_payload["runtime_tree_sha256"],
        "manifest_sha256": module.file_hash(candidate_path),
    }
    sources = {
        "h5ad": tmp_path / "source.h5ad",
        "recipe": tmp_path / "source.rds",
        "python_config": tmp_path / "source.config.json",
        "custody_record": tmp_path / "source.custody.json",
    }
    for path in sources.values():
        path.write_bytes(path.name.encode("utf-8"))
    data_roles = ROOT / "validation/contracts/data_roles_v2.json"
    profile = tmp_path / "rf_profile.json"
    profile.write_text(
        json.dumps({"status": "FROZEN", "winner": {"profile": {}}}),
        encoding="utf-8",
    )
    margins = tmp_path / "margins.json"
    margins.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "eligible": True,
                "maximum_margin": 0.03,
                "candidate": context,
            }
        ),
        encoding="utf-8",
    )
    authorization = tmp_path / "authorization.json"
    bindings = {name: module.file_hash(path) for name, path in sources.items()}
    bindings["h5ad"] = "f" * 64
    authorization.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "status": "AUTHORIZED",
                "authorization_id": "fixture-failure",
                "target": "hierarchy_confirm_Arigoni",
                "dataset_roles": ["hierarchy_parity_confirmation:Arigoni"],
                "candidate_commit": context["commit"],
                "attempt": 1,
                "runtime_tree_sha256": context["runtime_tree_sha256"],
                "bound_artifacts": {
                    **bindings,
                    "hierarchy_margins": module.file_hash(margins),
                    "data_roles": module.file_hash(data_roles),
                    "rf_profile": module.file_hash(profile),
                },
            }
        ),
        encoding="utf-8",
    )
    destinations = {
        name: tmp_path / "sealed" / path.name for name, path in sources.items()
    }
    output = tmp_path / "opening.json"
    with pytest.raises(RuntimeError, match="checksum drift"):
        module.seal(
            root=tmp_path,
            candidate_path=candidate_path,
            authorization_path=authorization,
            margins_path=margins,
            data_roles_path=data_roles,
            profile_path=profile,
            dataset_id="Arigoni",
            sources=sources,
            destinations=destinations,
            output_path=output,
        )
    assert (tmp_path / "opening.ATTEMPTED.json").is_file()
    assert (tmp_path / "opening.FAILED.json").is_file()
    with pytest.raises(RuntimeError, match="already been opened"):
        module.seal(
            root=tmp_path,
            candidate_path=candidate_path,
            authorization_path=authorization,
            margins_path=margins,
            data_roles_path=data_roles,
            profile_path=profile,
            dataset_id="Arigoni",
            sources=sources,
            destinations=destinations,
            output_path=output,
        )
