#!/usr/bin/env python3
"""Issue a fail-closed readiness report for the complete v0.1 parity claim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import (
    file_hash,
    immutable_json,
    validate_candidate_manifest,
)
from validation.scripts.verify_parity_ledger import inventory, validate_ledger

PERMITTED_CLAIM = (
    "A faithful Python reimplementation of the unimodal-RNA CHOIR v0.3.0 "
    "algorithm in compatibility mode within prespecified fidelity margins."
)
HIERARCHY_EVIDENCE_ENTRIES = {
    "comparison-add-distance",
    "comparison-check-distance",
    "centroid-distance",
    "starting-resolution",
    "tree-dispatch",
    "tree-full",
    "tree-silhouette",
    "tree-subtree-pending",
    "tree-optimization-pending",
    "rna-dimensional-reduction-pending",
    "choir-orchestration",
    "build-tree",
    "prune-tree",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"promotion evidence must be an object: {path}")
    return value


def _candidate_context(candidate: dict[str, Any], manifest: Path) -> dict[str, Any]:
    return {
        "commit": candidate["candidate_commit"],
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": file_hash(manifest),
    }


def _gate_candidate_matches(
    evidence: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    observed = evidence.get("candidate", {})
    return isinstance(observed, dict) and all(
        observed.get(key) == value for key, value in context.items()
    )


def unresolved_ledger_entries(
    ledger: dict[str, Any],
    *,
    hierarchy_complete: bool,
    margins_frozen: bool,
) -> list[str]:
    """Resolve only predeclared external entries through full hierarchy evidence."""

    unresolved: list[str] = []
    for entry in ledger["entries"]:
        entry_id = str(entry["entry_id"])
        externally_pending = (
            entry["evidence"]["status"] == "pending"
            or entry["classification"] == "unavailable_input"
            or any(
                branch["classification"] == "unavailable_input"
                for branch in entry["meaningful_branches"]
            )
        )
        if externally_pending and not (
            entry_id in HIERARCHY_EVIDENCE_ENTRIES
            and hierarchy_complete
            and margins_frozen
        ):
            unresolved.append(entry_id)
    return sorted(set(unresolved))


def assess(
    *,
    root: Path,
    candidate_path: Path,
    gate0_path: Path,
    gate1_static_path: Path,
    gate1_gpu_path: Path,
    framework_path: Path,
    comparator_path: Path,
    comparator_seal_path: Path,
    hierarchy_path: Path,
    margin_path: Path,
    margin_reproduction_path: Path,
    rf_profile_reproduction_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    candidate = validate_candidate_manifest(root, candidate_path)
    context = _candidate_context(candidate, candidate_path)
    gate0 = _read(gate0_path)
    gate1_static = _read(gate1_static_path)
    gate1_gpu = _read(gate1_gpu_path)
    framework = _read(framework_path)
    comparator = _read(comparator_path)
    comparator_seal = _read(comparator_seal_path)
    hierarchy = _read(hierarchy_path)
    margins = _read(margin_path)
    margin_reproduction = _read(margin_reproduction_path)
    rf_profile_reproduction = _read(rf_profile_reproduction_path)
    ledger = _read(ledger_path)

    functions = inventory(root / "validation/reference/choir_v0_3_0/R", root)
    ledger_errors = validate_ledger(ledger, functions=functions, repo_root=root)
    if ledger_errors:
        raise RuntimeError(f"parity ledger verification failed: {ledger_errors}")

    hierarchy_complete = (
        hierarchy.get("schema_version")
        == "rapids_choir.hierarchy-fidelity-aggregate/v1"
        and hierarchy.get("status") == "PASS"
        and hierarchy.get("candidate") == context
        and all(hierarchy.get("checks", {}).values())
    )
    margins_frozen = (
        margins.get("status") == "FROZEN"
        and margins.get("eligible") is True
        and float(margins.get("maximum_margin", -1)) == 0.03
        and margins.get("candidate") == context
    )
    margins_reproduced = (
        margin_reproduction.get("status") == "REPRODUCED"
        and margin_reproduction.get("candidate") == context
        and margin_reproduction.get("exact_margin_payload") is True
        and margin_reproduction.get("reproduced", {}).get("sha256")
        == file_hash(margin_path)
    )
    rf_profile_reproduced = (
        rf_profile_reproduction.get("status") == "REPRODUCED"
        and rf_profile_reproduction.get("candidate") == context
        and rf_profile_reproduction.get("exact_winning_profile") is True
        and rf_profile_reproduction.get("exact_reference_variability_margins")
        is True
        and rf_profile_reproduction.get("protocol_checks")
        == {
            "ranking_order": True,
            "comparator_contract_sha256": True,
            "environment_lock_sha256": True,
        }
    )
    unresolved = unresolved_ledger_entries(
        ledger,
        hierarchy_complete=hierarchy_complete,
        margins_frozen=margins_frozen,
    )

    comparator_checks = comparator.get("checks", {})
    comparator_pass = (
        comparator.get("schema_version") == 2
        and comparator.get("status") == "PASS"
        and comparator.get("scope")
        == "sealed_holdout_comparator_component_fidelity"
        and comparator.get("source_commit") == candidate["candidate_commit"]
        and comparator.get("authorization", {}).get("attempt")
        == candidate["candidate_attempt"]
        and isinstance(comparator_checks, dict)
        and comparator_checks
        and all(comparator_checks.values())
        and comparator_seal.get("status") == "PASS"
        and comparator_seal.get("scope") == "comparator_confirm"
        and comparator_seal.get("source_sha256") == file_hash(comparator_path)
        and comparator.get("rf_profile", {}).get(
            "final_candidate_reproduction_sha256"
        )
        == file_hash(rf_profile_reproduction_path)
    )
    framework_pass = (
        framework.get("status") == "READY"
        and framework.get("scope") == "parity_assurance_framework"
        and framework.get("candidate", {}).get("runtime_tree_sha256")
        == candidate["runtime_tree_sha256"]
        and framework.get("mutation_count") == 12
        and framework.get("ledger_entry_count") == 45
    )
    checks = {
        "gate0_same_candidate_pass": gate0.get("status") == "PASS"
        and _gate_candidate_matches(gate0, context),
        "static_gate1_same_candidate_pass": gate1_static.get("status") == "PASS"
        and _gate_candidate_matches(gate1_static, context),
        "real_l40s_gate1_same_candidate_pass": gate1_gpu.get("status") == "PASS"
        and gate1_gpu.get("scope") == "real_gpu_functional"
        and _gate_candidate_matches(gate1_gpu, context),
        "assurance_framework_same_candidate_ready": framework_pass,
        "comparator_confirmation_same_candidate_pass": comparator_pass,
        "rf_profile_reproduced_by_final_candidate": rf_profile_reproduced,
        "hierarchy_margins_same_candidate_frozen": margins_frozen,
        "hierarchy_margins_reproduce_frozen_contract": margins_reproduced,
        "hierarchy_development_and_confirmation_same_candidate_pass": (
            hierarchy_complete
        ),
        "zero_unresolved_parity_ledger_entries": not unresolved,
    }
    ready = all(checks.values())
    return {
        "schema_version": "rapids_choir.parity-promotion/v1",
        "status": "READY" if ready else "BLOCKED",
        "candidate": context,
        "checks": checks,
        "unresolved_ledger_entry_ids": unresolved,
        "permitted_claim": PERMITTED_CLAIM if ready else None,
        "prohibited_claims": [
            "bitwise identity",
            "source-code identity",
            "drop-in R-package equivalence",
            "parity for excluded adapters, plotting, or multimodal behavior",
            "R parity for corrected mode",
        ],
        "evidence": {
            "gate0": file_hash(gate0_path),
            "gate1_static": file_hash(gate1_static_path),
            "gate1_gpu": file_hash(gate1_gpu_path),
            "assurance_framework": file_hash(framework_path),
            "comparator": file_hash(comparator_path),
            "comparator_seal": file_hash(comparator_seal_path),
            "hierarchy": file_hash(hierarchy_path),
            "hierarchy_margins": file_hash(margin_path),
            "hierarchy_margin_reproduction": file_hash(
                margin_reproduction_path
            ),
            "rf_profile_reproduction": file_hash(
                rf_profile_reproduction_path
            ),
            "parity_ledger": file_hash(ledger_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--gate0", required=True, type=Path)
    parser.add_argument("--gate1-static", required=True, type=Path)
    parser.add_argument("--gate1-gpu", required=True, type=Path)
    parser.add_argument("--framework", required=True, type=Path)
    parser.add_argument("--comparator", required=True, type=Path)
    parser.add_argument("--comparator-seal", required=True, type=Path)
    parser.add_argument("--hierarchy", required=True, type=Path)
    parser.add_argument("--hierarchy-margins", required=True, type=Path)
    parser.add_argument(
        "--hierarchy-margin-reproduction", required=True, type=Path
    )
    parser.add_argument("--rf-profile-reproduction", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    report = assess(
        root=arguments.root.resolve(),
        candidate_path=arguments.candidate_manifest,
        gate0_path=arguments.gate0,
        gate1_static_path=arguments.gate1_static,
        gate1_gpu_path=arguments.gate1_gpu,
        framework_path=arguments.framework,
        comparator_path=arguments.comparator,
        comparator_seal_path=arguments.comparator_seal,
        hierarchy_path=arguments.hierarchy,
        margin_path=arguments.hierarchy_margins,
        margin_reproduction_path=arguments.hierarchy_margin_reproduction,
        rf_profile_reproduction_path=arguments.rf_profile_reproduction,
        ledger_path=arguments.ledger,
    )
    immutable_json(arguments.output, report)
    if report["status"] != "READY":
        blocked = arguments.output.with_name(
            f"{arguments.output.stem}.BLOCKED.json"
        )
        immutable_json(blocked, report)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
