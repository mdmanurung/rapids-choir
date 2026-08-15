from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/scripts"))
SPEC = importlib.util.spec_from_file_location(
    "require_parity_framework",
    ROOT / "validation/scripts/require_parity_framework.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_framework_seal_requires_real_l40s_and_reference_equivalence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text("{}\n", encoding="utf-8")
    candidate = {
        "candidate_commit": "a" * 40,
        "candidate_attempt": 1,
        "runtime_tree_sha256": "b" * 64,
    }
    monkeypatch.setattr(
        module,
        "validate_candidate_manifest",
        lambda root, path: candidate,
    )
    context = {
        "commit": candidate["candidate_commit"],
        "attempt": 1,
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": module.file_hash(candidate_path),
    }
    contract_names = {
        "data_roles",
        "hierarchy_development",
        "hierarchy_margins",
        "hierarchy_result_schema",
        "hierarchy_validation",
        "l40s_performance",
        "ledger",
        "mutations",
        "publication_claims",
        "release_candidate",
        "sant_reproduction",
        "scope",
        "secondary_evidence",
        "semantic_schema",
    }
    gate0 = tmp_path / "gate0.json"
    gate0.write_text(
        json.dumps(
            {
                "status": "PASS",
                "candidate": context,
                "parity_assurance": {
                    "contracts": {name: {} for name in contract_names},
                    "unclassified_count": 0,
                    "r_function_count": 45,
                    "ledger_entry_count": 45,
                    "promotion_ready": False,
                },
            }
        ),
        encoding="utf-8",
    )
    static = tmp_path / "gate1-static.json"
    checks = [
        (
            ["python", "validation/scripts/verify_parity_ledger.py"],
            {"passed": True, "unclassified_count": 0, "ledger_entry_count": 45},
        ),
        (
            ["python", "validation/scripts/run_mutation_controls.py"],
            {
                "status": "PASS",
                "mutation_count": 12,
                "detected_count": 12,
                "source_tree_unchanged": True,
            },
        ),
        (
            ["Rscript", "validation/scripts/run_reference_hierarchy.R"],
            {
                "status": "PASS",
                "results_identical": True,
                "r_version": "4.5.1",
                "observed_boundaries": [".relabelSingletons"],
            },
        ),
    ]
    static.write_text(
        json.dumps(
            {
                "status": "PASS",
                "candidate": context,
                "checks": [
                    {
                        "command": command,
                        "returncode": 0,
                        "stdout": json.dumps(stdout),
                    }
                    for command, stdout in checks
                ],
            }
        ),
        encoding="utf-8",
    )
    gpu = tmp_path / "gate1-gpu.json"
    gpu.write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "real_gpu_functional",
                "candidate": context,
                "conditions": {"typed_l40s": True},
                "slurm": {"job_id": "123"},
            }
        ),
        encoding="utf-8",
    )
    report = module.verify(tmp_path, candidate_path, gate0, static, gpu)
    assert report["status"] == "READY"
    assert report["real_l40s_verified"] is True

    failed_gpu = json.loads(gpu.read_text(encoding="utf-8"))
    failed_gpu["conditions"]["typed_l40s"] = False
    gpu.write_text(json.dumps(failed_gpu), encoding="utf-8")
    with pytest.raises(RuntimeError, match="real typed L40S"):
        module.verify(tmp_path, candidate_path, gate0, static, gpu)
