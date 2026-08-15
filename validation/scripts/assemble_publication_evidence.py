#!/usr/bin/env python3
"""Assemble the narrow software-article evidence without broadening its claims."""

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

REQUIRED_ARTIFACTS = {
    "claim_to_evidence_manifest",
    "manuscript",
    "figures_manifest",
    "tables_manifest",
    "supplement",
    "negative_results_appendix",
    "bmc_package",
    "venue_requirements_audit",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"publication evidence must be an object: {path}")
    return value


def _candidate(candidate: dict[str, Any], manifest: Path) -> dict[str, Any]:
    return {
        "commit": candidate["candidate_commit"],
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": file_hash(manifest),
    }


def assemble(
    *,
    root: Path,
    candidate_path: Path,
    claims_path: Path,
    data_roles_path: Path,
    parity_path: Path,
    secondary_path: Path,
    sant_path: Path,
    performance_path: Path,
    release_path: Path,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    if set(artifacts) != REQUIRED_ARTIFACTS:
        raise RuntimeError("publication artifact set is incomplete or unknown")
    if any(
        not path.is_file() or path.stat().st_size == 0
        for path in artifacts.values()
    ):
        raise RuntimeError("publication artifact is missing or empty")
    candidate = _candidate(
        validate_candidate_manifest(root, candidate_path), candidate_path
    )
    claims = _read(claims_path)
    data_roles = _read(data_roles_path)
    parity = _read(parity_path)
    secondary = _read(secondary_path)
    sant = _read(sant_path)
    performance = _read(performance_path)
    release = _read(release_path)
    if (
        claims.get("contract_id") != "narrow-software-publication-v2"
        or claims.get("target", {}).get("exclusive") is not True
        or data_roles.get("contract_id") != "publication-data-roles-v2"
        or parity.get("status") != "READY"
        or parity.get("candidate") != candidate
        or secondary.get("status") != "VALID"
        or secondary.get("candidate") != candidate
        or secondary.get("pooled_biological_validity_claim") is not False
        or sant.get("status") != "ACCOUNTED"
        or sant.get("candidate") != candidate
        or performance.get("status")
        not in {"CLAIM_ELIGIBLE", "CLAIM_WITHHELD"}
        or performance.get("candidate") != candidate
        or release.get("status") != "READY"
        or release.get("scope") != "local_release_candidate_only"
        or release.get("candidate") != candidate
    ):
        raise RuntimeError("publication evidence is incomplete, stale, or overclaimed")
    optional_claims = {
        "tested_l40s_performance": performance["status"] == "CLAIM_ELIGIBLE",
        "sant_reproduction_completeness": sant.get("completeness_claim")
        == "ELIGIBLE",
    }
    return {
        "schema_version": 1,
        "status": "READY",
        "scope": "bmc_bioinformatics_software_article",
        "candidate": candidate,
        "primary_claim": claims["primary_claim"],
        "optional_claims": optional_claims,
        "secondary_case_studies": {
            "status": "VALID",
            "publication_weight": "secondary",
            "may_determine_publication_eligibility": False,
            "sha256": file_hash(secondary_path),
        },
        "prohibited_claims": claims["prohibited_claims"],
        "evidence": {
            "claims": file_hash(claims_path),
            "data_roles": file_hash(data_roles_path),
            "parity_promotion": file_hash(parity_path),
            "secondary": file_hash(secondary_path),
            "sant": file_hash(sant_path),
            "performance": file_hash(performance_path),
            "release_candidate": file_hash(release_path),
        },
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for name, path in sorted(artifacts.items())
        },
        "external_action_authorized": False,
    }


def _artifact_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or name in result:
            raise RuntimeError("--artifact must be unique NAME=PATH values")
        result[name] = Path(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--claims", required=True, type=Path)
    parser.add_argument("--data-roles", required=True, type=Path)
    parser.add_argument("--parity", required=True, type=Path)
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--sant", required=True, type=Path)
    parser.add_argument("--performance", required=True, type=Path)
    parser.add_argument("--release-candidate", required=True, type=Path)
    parser.add_argument("--artifact", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    immutable_json(
        arguments.output,
        assemble(
            root=arguments.root.resolve(),
            candidate_path=arguments.candidate_manifest,
            claims_path=arguments.claims,
            data_roles_path=arguments.data_roles,
            parity_path=arguments.parity,
            secondary_path=arguments.secondary,
            sant_path=arguments.sant,
            performance_path=arguments.performance,
            release_path=arguments.release_candidate,
            artifacts=_artifact_arguments(arguments.artifact),
        ),
    )


if __name__ == "__main__":
    main()
