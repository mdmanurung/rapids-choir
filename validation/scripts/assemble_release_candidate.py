#!/usr/bin/env python3
"""Assemble a local release candidate; never publish, tag, or submit it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import (
    file_hash,
    immutable_json,
    validate_candidate_manifest,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"release evidence must be an object: {path}")
    return value


def _artifacts(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        name, separator, path = raw.partition("=")
        if not separator or not name or name in result:
            raise RuntimeError("--artifact must be unique NAME=PATH values")
        result[name] = Path(path)
    return result


def _source_version(root: Path) -> str:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    init = (root / "src/rapids_choir/__init__.py").read_text(encoding="utf-8")
    project_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    init_match = re.search(r'^__version__ = "([^"]+)"$', init, re.MULTILINE)
    if project_match is None or init_match is None:
        raise RuntimeError("release version is missing")
    if project_match.group(1) != init_match.group(1):
        raise RuntimeError("release source versions disagree")
    version = project_match.group(1)
    if any(token in version for token in ("dev", "a", "b", "rc", "+")):
        raise RuntimeError("release candidate source version is not final")
    return version


def assemble(
    *,
    root: Path,
    candidate_path: Path,
    contract_path: Path,
    parity_path: Path,
    stable_profile_path: Path,
    installed_profile_path: Path,
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    contract = _read(contract_path)
    required = set(contract.get("required_artifacts", []))
    if (
        contract.get("contract_id")
        != "narrow-software-release-candidate-v1"
        or contract.get("parity_promotion_required") is not True
        or set(artifacts) != required
    ):
        raise RuntimeError("release-candidate artifact contract has drifted")
    if any(
        not path.is_file() or path.stat().st_size == 0
        for path in artifacts.values()
    ):
        raise RuntimeError("release-candidate artifact is missing or empty")
    candidate_manifest = validate_candidate_manifest(root, candidate_path)
    candidate = {
        "commit": candidate_manifest["candidate_commit"],
        "attempt": candidate_manifest["candidate_attempt"],
        "runtime_tree_sha256": candidate_manifest["runtime_tree_sha256"],
        "manifest_sha256": file_hash(candidate_path),
    }
    parity = _read(parity_path)
    if (
        parity.get("status") != "READY"
        or parity.get("candidate") != candidate
    ):
        raise RuntimeError("parity promotion is not ready for this candidate")
    if (
        not stable_profile_path.is_file()
        or not installed_profile_path.is_file()
        or stable_profile_path.read_bytes() != installed_profile_path.read_bytes()
    ):
        raise RuntimeError("the frozen RF profile is not shipped byte-for-byte")
    for audit_name in (
        "rights_audit",
        "credential_audit",
        "restricted_data_audit",
        "stale_evidence_audit",
    ):
        audit = _read(artifacts[audit_name])
        if audit.get("status") != "PASS" or audit.get("candidate") != candidate:
            raise RuntimeError(f"release audit failed or drifted: {audit_name}")
    clean = _read(artifacts["release_manifest"])
    if (
        clean.get("clean_checkout_reproduction") != "PASS"
        or clean.get("candidate") != candidate
    ):
        raise RuntimeError("clean-checkout release reproduction is absent")
    version = _source_version(root)
    return {
        "schema_version": 1,
        "status": "READY",
        "scope": "local_release_candidate_only",
        "version": version,
        "candidate": candidate,
        "parity_promotion_sha256": file_hash(parity_path),
        "stable_rf_profile_sha256": file_hash(stable_profile_path),
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for name, path in sorted(artifacts.items())
        },
        "external_actions_authorized": [],
        "external_actions_requiring_separate_authorization": sorted(
            contract["external_actions"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--parity-promotion", required=True, type=Path)
    parser.add_argument("--stable-profile", required=True, type=Path)
    parser.add_argument("--installed-profile", required=True, type=Path)
    parser.add_argument("--artifact", required=True, action="append")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    immutable_json(
        arguments.output,
        assemble(
            root=arguments.root.resolve(),
            candidate_path=arguments.candidate_manifest,
            contract_path=arguments.contract,
            parity_path=arguments.parity_promotion,
            stable_profile_path=arguments.stable_profile,
            installed_profile_path=arguments.installed_profile,
            artifacts=_artifacts(arguments.artifact),
        ),
    )


if __name__ == "__main__":
    main()
