#!/usr/bin/env python3
"""Seal the assurance framework before any comparator holdout is authorized."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gate2_common import file_hash, immutable_json, validate_candidate_manifest


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"assurance evidence must be an object: {path}")
    return value


def verify(
    root: Path,
    candidate_path: Path,
    gate0_path: Path,
    gate1_path: Path,
    gate1_gpu_path: Path,
) -> dict[str, Any]:
    candidate = validate_candidate_manifest(root, candidate_path)
    gate0 = _read(gate0_path)
    gate1 = _read(gate1_path)
    gate1_gpu = _read(gate1_gpu_path)
    runtime = candidate["runtime_tree_sha256"]
    expected_candidate = {
        "commit": candidate["candidate_commit"],
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": runtime,
        "manifest_sha256": file_hash(candidate_path),
    }
    if (
        gate0.get("status") != "PASS"
        or gate1.get("status") != "PASS"
        or gate1_gpu.get("status") != "PASS"
    ):
        raise RuntimeError(
            "parity framework requires passing Gate 0, static Gate 1, and "
            "real-L40S Gate 1"
        )
    if any(
        gate0.get("candidate", {}).get(key) != value
        for key, value in expected_candidate.items()
    ):
        raise RuntimeError("Gate 0 parity framework candidate drift")
    if any(
        gate1.get("candidate", {}).get(key) != value
        for key, value in expected_candidate.items()
    ):
        raise RuntimeError("static Gate 1 parity framework candidate drift")
    if any(
        gate1_gpu.get("candidate", {}).get(key) != value
        for key, value in expected_candidate.items()
    ):
        raise RuntimeError("real-L40S Gate 1 parity framework candidate drift")
    if (
        gate1_gpu.get("scope") != "real_gpu_functional"
        or gate1_gpu.get("conditions", {}).get("typed_l40s") is not True
        or not gate1_gpu.get("slurm", {}).get("job_id")
    ):
        raise RuntimeError("Gate 1 GPU evidence is not a real typed L40S run")
    assurance = gate0.get("parity_assurance")
    if not isinstance(assurance, dict):
        raise RuntimeError("Gate 0 predates the parity assurance framework")
    if (
        assurance.get("unclassified_count") != 0
        or assurance.get("r_function_count") != assurance.get("ledger_entry_count")
        or assurance.get("r_function_count") != 45
    ):
        raise RuntimeError("parity ledger is incomplete or unclassified")
    required_contracts = {
        "data_roles",
        "hierarchy_development",
        "l40s_performance",
        "release_candidate",
        "sant_reproduction",
        "secondary_evidence",
        "scope",
        "ledger",
        "mutations",
        "publication_claims",
        "semantic_schema",
        "hierarchy_result_schema",
        "hierarchy_margins",
        "hierarchy_validation",
    }
    if set(assurance.get("contracts", {})) != required_contracts:
        raise RuntimeError("Gate 0 parity contract set is incomplete")

    reports: dict[str, dict[str, Any]] = {}
    for check in gate1.get("checks", []):
        command = check.get("command", [])
        if not isinstance(command, list) or len(command) < 2:
            continue
        scripts = {Path(str(argument)).name for argument in command}
        script = next(
            (
                name
                for name in (
                    "verify_parity_ledger.py",
                    "run_mutation_controls.py",
                    "run_reference_hierarchy.R",
                )
                if name in scripts
            ),
            None,
        )
        if script is None:
            continue
        if script in {
            "verify_parity_ledger.py",
            "run_mutation_controls.py",
            "run_reference_hierarchy.R",
        }:
            try:
                reports[script] = json.loads(check.get("stdout", ""))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"static Gate 1 {script} output is invalid"
                ) from error
            if check.get("returncode") != 0:
                raise RuntimeError(f"static Gate 1 {script} failed")
    ledger = reports.get("verify_parity_ledger.py", {})
    mutations = reports.get("run_mutation_controls.py", {})
    reference_driver = reports.get("run_reference_hierarchy.R", {})
    if ledger.get("passed") is not True or ledger.get("unclassified_count") != 0:
        raise RuntimeError("static Gate 1 parity ledger check is absent or failed")
    if (
        mutations.get("status") != "PASS"
        or mutations.get("mutation_count") != 12
        or mutations.get("detected_count") != 12
        or mutations.get("source_tree_unchanged") is not True
    ):
        raise RuntimeError("static Gate 1 mutation controls are absent or incomplete")
    if (
        reference_driver.get("status") != "PASS"
        or reference_driver.get("results_identical") is not True
        or reference_driver.get("r_version") != "4.5.1"
        or ".relabelSingletons"
        not in reference_driver.get("observed_boundaries", [])
    ):
        raise RuntimeError(
            "instrumented R4_51 reference equivalence check is absent or failed"
        )
    return {
        "schema_version": 1,
        "status": "READY",
        "scope": "parity_assurance_framework",
        "candidate": expected_candidate,
        "gate0_sha256": file_hash(gate0_path),
        "gate1_static_sha256": file_hash(gate1_path),
        "gate1_gpu_sha256": file_hash(gate1_gpu_path),
        "ledger_entry_count": ledger["ledger_entry_count"],
        "mutation_count": mutations["mutation_count"],
        "reference_instrumentation_equivalent": True,
        "real_l40s_verified": True,
        "promotion_ready": assurance.get("promotion_ready") is True,
        "limitations": [
            "READY permits only separately authorized confirmatory workflows.",
            "It is not confirmation evidence or release-promotion evidence.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--gate0", required=True, type=Path)
    parser.add_argument("--gate1-static", required=True, type=Path)
    parser.add_argument("--gate1-gpu", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    payload = verify(
        arguments.root.resolve(),
        arguments.candidate_manifest,
        arguments.gate0,
        arguments.gate1_static,
        arguments.gate1_gpu,
    )
    immutable_json(arguments.output, payload)


if __name__ == "__main__":
    main()
