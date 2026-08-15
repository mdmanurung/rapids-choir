from __future__ import annotations

import json
from pathlib import Path

from validation.scripts import assemble_publication_evidence as publication_module
from validation.scripts import assemble_release_candidate as release_module
from validation.scripts.aggregate_sant_reproduction import aggregate as aggregate_sant
from validation.scripts.aggregate_secondary_evidence import (
    aggregate as aggregate_secondary,
)
from validation.scripts.evaluate_l40s_performance import evaluate

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = {
    "commit": "a" * 40,
    "attempt": 1,
    "runtime_tree_sha256": "b" * 64,
    "manifest_sha256": "c" * 64,
}


def test_secondary_results_are_mode_separated_and_never_pooled(
    tmp_path: Path,
) -> None:
    contract_path = ROOT / "validation/contracts/secondary_evidence_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    paths = []
    for dataset in contract["datasets"]:
        path = tmp_path / f"{dataset['dataset_id']}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "VALID",
                    "runtime_schema_status": "PASS",
                    "dataset_id": dataset["dataset_id"],
                    "candidate": CANDIDATE,
                    "modes": ["compatibility", "corrected"],
                    "constructs": {
                        construct: {"compatibility": 0.2, "corrected": 0.1}
                        for construct in dataset["constructs"]
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = aggregate_secondary(contract_path, paths)
    assert result["status"] == "VALID"
    assert result["pooled_biological_validity_claim"] is False
    assert result["may_determine_publication_eligibility"] is False


def test_sant_failures_are_accounted_without_blocking_primary_claim(
    tmp_path: Path,
) -> None:
    contract_path = ROOT / "validation/contracts/sant_reproduction_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    paths = []
    for index, analysis_id in enumerate(contract["analyses"]):
        path = tmp_path / f"sant-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "analysis_id": analysis_id,
                    "classification": "failed" if index == 0 else "reproduced",
                    "candidate": CANDIDATE,
                    "attempts": [{"status": "recorded"}],
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = aggregate_sant(contract_path, paths)
    assert result["status"] == "ACCOUNTED"
    assert result["completeness_claim"] == "WITHHELD"
    assert result["primary_claim_blocked"] is False


def test_l40s_claim_requires_every_workload_lower_bound_and_correctness(
    tmp_path: Path,
) -> None:
    contract_path = ROOT / "validation/contracts/l40s_performance_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    paths = []
    for workload_id in contract["workloads"]:
        pairs = []
        for index, order in enumerate(
            contract["pair_order"]["schedule"][workload_id], start=1
        ):
            correctness = tmp_path / f"{workload_id}-correctness-{index}.json"
            correctness.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "PASS",
                        "workload_id": workload_id,
                        "pair_index": index,
                        "candidate": CANDIDATE,
                        "checks": {"hierarchy_contract": True},
                        "r_output_sha256": "f" * 64,
                        "python_output_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            pairs.append(
                {
                    "pair_index": index,
                    "order": order,
                    "r_process_id": f"{workload_id}-r-{index}",
                    "python_process_id": f"{workload_id}-python-{index}",
                    "input_sha256": "d" * 64,
                    "settings_sha256": "e" * 64,
                    "r_wall_seconds": 3.0,
                    "python_wall_seconds": 1.0,
                    "correctness_report": {
                        "path": str(correctness),
                        "sha256": __import__("hashlib").sha256(
                            correctness.read_bytes()
                        ).hexdigest(),
                    },
                    "r_host_peak_bytes": 100,
                    "r_device_peak_bytes": 0,
                    "python_host_peak_bytes": 80,
                    "python_device_peak_bytes": 200,
                    "fresh_process": True,
                    "full_io_included": True,
                    "durable_output": True,
                }
            )
        path = tmp_path / f"{workload_id}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "workload_id": workload_id,
                    "hardware": "NVIDIA L40S",
                    "genes": 10000,
                    "warmup": {
                        "r": {
                            "fresh_process": True,
                            "excluded_from_measurement": True,
                            "wall_seconds": 1.0,
                            "process_id": f"{workload_id}-r-warmup",
                        },
                        "python": {
                            "fresh_process": True,
                            "excluded_from_measurement": True,
                            "wall_seconds": 1.0,
                            "process_id": f"{workload_id}-python-warmup",
                        },
                    },
                    "timing_boundary": "input_open_to_durable_output_fsync",
                    "candidate": CANDIDATE,
                    "pairs": pairs,
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = evaluate(contract_path, paths)
    assert result["status"] == "CLAIM_ELIGIBLE"
    assert all(
        workload["one_sided_95_percent_lower_bound"] >= 2
        for workload in result["workloads"]
    )


def test_publication_package_binds_a_parity_ready_release_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    candidate_manifest = {
        "candidate_commit": CANDIDATE["commit"],
        "candidate_attempt": CANDIDATE["attempt"],
        "runtime_tree_sha256": CANDIDATE["runtime_tree_sha256"],
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    candidate = {
        **CANDIDATE,
        "manifest_sha256": release_module.file_hash(candidate_path),
    }
    monkeypatch.setattr(
        release_module,
        "validate_candidate_manifest",
        lambda root, path: candidate_manifest,
    )
    monkeypatch.setattr(
        publication_module,
        "validate_candidate_manifest",
        lambda root, path: candidate_manifest,
    )
    (tmp_path / "src/rapids_choir").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (tmp_path / "src/rapids_choir/__init__.py").write_text(
        '__version__ = "0.1.0"\n', encoding="utf-8"
    )
    parity_path = tmp_path / "parity.json"
    parity_path.write_text(
        json.dumps({"status": "READY", "candidate": candidate}),
        encoding="utf-8",
    )
    stable_profile = tmp_path / "rf_profile.json"
    installed_profile = tmp_path / "installed_rf_profile.json"
    stable_profile.write_text('{"status":"FROZEN"}\n', encoding="utf-8")
    installed_profile.write_bytes(stable_profile.read_bytes())
    release_contract_path = ROOT / "validation/contracts/release_candidate_v1.json"
    release_contract = json.loads(
        release_contract_path.read_text(encoding="utf-8")
    )
    release_artifacts: dict[str, Path] = {}
    for name in release_contract["required_artifacts"]:
        path = tmp_path / f"release-{name}.json"
        if name in {
            "rights_audit",
            "credential_audit",
            "restricted_data_audit",
            "stale_evidence_audit",
        }:
            payload = {"status": "PASS", "candidate": candidate}
        elif name == "release_manifest":
            payload = {
                "clean_checkout_reproduction": "PASS",
                "candidate": candidate,
            }
        else:
            payload = {"artifact": name}
        path.write_text(json.dumps(payload), encoding="utf-8")
        release_artifacts[name] = path
    release = release_module.assemble(
        root=tmp_path,
        candidate_path=candidate_path,
        contract_path=release_contract_path,
        parity_path=parity_path,
        stable_profile_path=stable_profile,
        installed_profile_path=installed_profile,
        artifacts=release_artifacts,
    )
    assert release["status"] == "READY"
    assert release["parity_promotion_sha256"] == release_module.file_hash(
        parity_path
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")

    secondary_path = tmp_path / "secondary.json"
    secondary_path.write_text(
        json.dumps(
            {
                "status": "VALID",
                "candidate": candidate,
                "pooled_biological_validity_claim": False,
            }
        ),
        encoding="utf-8",
    )
    sant_path = tmp_path / "sant.json"
    sant_path.write_text(
        json.dumps(
            {
                "status": "ACCOUNTED",
                "candidate": candidate,
                "completeness_claim": "WITHHELD",
            }
        ),
        encoding="utf-8",
    )
    performance_path = tmp_path / "performance.json"
    performance_path.write_text(
        json.dumps({"status": "CLAIM_WITHHELD", "candidate": candidate}),
        encoding="utf-8",
    )
    publication_artifacts: dict[str, Path] = {}
    for name in publication_module.REQUIRED_ARTIFACTS:
        path = tmp_path / f"publication-{name}.txt"
        path.write_text(f"{name}\n", encoding="utf-8")
        publication_artifacts[name] = path
    publication = publication_module.assemble(
        root=tmp_path,
        candidate_path=candidate_path,
        claims_path=ROOT / "validation/contracts/publication_claims_v2.json",
        data_roles_path=ROOT / "validation/contracts/data_roles_v2.json",
        parity_path=parity_path,
        secondary_path=secondary_path,
        sant_path=sant_path,
        performance_path=performance_path,
        release_path=release_path,
        artifacts=publication_artifacts,
    )
    assert publication["status"] == "READY"
    assert publication["evidence"]["release_candidate"] == (
        publication_module.file_hash(release_path)
    )
