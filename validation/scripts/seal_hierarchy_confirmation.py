#!/usr/bin/env python3
"""Open one authorized hierarchy dataset once and seal immutable working copies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import (
    file_hash,
    immutable_json,
    preserve_failure_record,
    validate_candidate_manifest,
)

RAW_BINDINGS = {"h5ad", "recipe", "python_config", "custody_record"}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _copy_and_hash(source: Path, temporary: Path) -> str:
    digest = hashlib.sha256()
    with source.open("rb") as reader, temporary.open("xb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    return digest.hexdigest()


def seal(
    *,
    root: Path,
    candidate_path: Path,
    authorization_path: Path,
    margins_path: Path,
    data_roles_path: Path,
    profile_path: Path,
    dataset_id: str,
    sources: dict[str, Path],
    destinations: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    attempt_path = output_path.with_name(output_path.stem + ".ATTEMPTED.json")
    if set(sources) != RAW_BINDINGS or set(destinations) != RAW_BINDINGS:
        raise RuntimeError("hierarchy confirmation artifact roles are incomplete")
    if (
        attempt_path.exists()
        or output_path.exists()
        or any(path.exists() for path in destinations.values())
    ):
        raise RuntimeError("hierarchy confirmation has already been opened or sealed")
    if any(not path.is_file() for path in sources.values()):
        raise RuntimeError("authorized hierarchy confirmation input is missing")

    candidate = validate_candidate_manifest(root, candidate_path)
    context = {
        "commit": candidate["candidate_commit"],
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": file_hash(candidate_path),
    }
    authorization = _read(authorization_path)
    target = f"hierarchy_confirm_{dataset_id}"
    role = f"hierarchy_parity_confirmation:{dataset_id}"
    if (
        authorization.get("schema_version") != 2
        or authorization.get("status") != "AUTHORIZED"
        or authorization.get("target") != target
        or authorization.get("dataset_roles") != [role]
        or authorization.get("candidate_commit") != context["commit"]
        or authorization.get("attempt") != context["attempt"]
        or authorization.get("runtime_tree_sha256")
        != context["runtime_tree_sha256"]
    ):
        raise RuntimeError("hierarchy authorization does not bind this candidate")
    bindings = authorization.get("bound_artifacts")
    if not isinstance(bindings, dict) or set(bindings) != {
        *RAW_BINDINGS,
        "hierarchy_margins",
        "data_roles",
        "rf_profile",
    }:
        raise RuntimeError("hierarchy authorization artifact bindings are incomplete")
    if (
        bindings["hierarchy_margins"] != file_hash(margins_path)
        or bindings["data_roles"] != file_hash(data_roles_path)
        or bindings["rf_profile"] != file_hash(profile_path)
    ):
        raise RuntimeError("hierarchy authorization contract binding has drifted")
    margins = _read(margins_path)
    if (
        margins.get("status") != "FROZEN"
        or margins.get("eligible") is not True
        or margins.get("candidate") != context
        or float(margins.get("maximum_margin", -1)) != 0.03
    ):
        raise RuntimeError("hierarchy margins are not frozen for this candidate")
    data_roles = _read(data_roles_path)
    role_records = [
        record
        for record in data_roles.get("roles", [])
        if record.get("role_id") == "hierarchy_parity_confirmation"
    ]
    if (
        data_roles.get("schema_version") != 2
        or len(role_records) != 1
        or dataset_id not in role_records[0].get("datasets", [])
        or role_records[0].get("open_once") is not True
        or role_records[0].get("independent_hierarchy_validation") is not False
    ):
        raise RuntimeError("hierarchy confirmation data role has drifted")

    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    immutable_json(
        attempt_path,
        {
            "schema_version": 1,
            "status": "OPENING_STARTED",
            "scope": "one_time_hierarchy_confirmation_opening",
            "dataset_id": dataset_id,
            "candidate": context,
            "authorization_sha256": file_hash(authorization_path),
            "source_roles": sorted(sources),
        },
    )
    evidence: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(
            prefix=f"rapids-choir-{dataset_id}-open-once-",
            dir=destinations["h5ad"].parent,
        ) as temporary_raw:
            temporary_root = Path(temporary_raw)
            for name in sorted(RAW_BINDINGS):
                temporary = temporary_root / name
                digest = _copy_and_hash(sources[name], temporary)
                if digest != bindings[name]:
                    raise RuntimeError(
                        f"authorized hierarchy input checksum drift: {name}"
                    )
                os.link(temporary, destinations[name])
                evidence[name] = {
                    "source_path": str(sources[name].resolve()),
                    "sealed_path": str(destinations[name].resolve()),
                    "bytes": destinations[name].stat().st_size,
                    "sha256": digest,
                }
        custody = _read(destinations["custody_record"])
        required_custody = {
            "schema_version",
            "dataset_id",
            "deviation",
            "recorded_by",
            "recorded_at",
            "source_provenance",
        }
        if (
            set(custody) != required_custody
            or custody.get("schema_version") != 1
            or custody.get("dataset_id") != dataset_id
            or not isinstance(custody.get("deviation"), str)
            or not custody["deviation"].strip()
        ):
            raise RuntimeError("custody-deviation record is missing or invalid")
        profile = _read(profile_path)
        python_config = _read(destinations["python_config"])
        comparison = python_config.get("comparison")
        if (
            profile.get("status") != "FROZEN"
            or not isinstance(profile.get("winner"), dict)
            or not isinstance(comparison, dict)
            or python_config.get("rules") != "choir_v0_3_0"
            or python_config.get("device") != "cuda:0"
            or python_config.get("input_policy") != "transfer_once"
            or comparison.get("rules") != "choir_v0_3_0"
            or comparison.get("device") != "cuda:0"
            or comparison.get("rf_profile")
            != profile["winner"].get("profile")
            or Path(str(comparison.get("rf_profile_manifest"))).resolve()
            != profile_path.resolve()
        ):
            raise RuntimeError(
                "hierarchy confirmation config does not use the frozen "
                "compatibility profile"
            )
        report = {
            "schema_version": 1,
            "status": "SEALED",
            "scope": "one_time_hierarchy_confirmation_opening",
            "dataset_id": dataset_id,
            "role": "computational_parity_confirmation_only",
            "opening_count": 1,
            "independent_hierarchy_validation": False,
            "candidate": context,
            "authorization": {
                "path": str(authorization_path.resolve()),
                "sha256": file_hash(authorization_path),
                "authorization_id": authorization.get("authorization_id"),
            },
            "opening_attempt": {
                "path": str(attempt_path.resolve()),
                "sha256": file_hash(attempt_path),
            },
            "artifacts": evidence,
            "custody_deviation": custody["deviation"],
        }
        immutable_json(output_path, report)
        return report
    except Exception as error:
        preserve_failure_record(
            output_path,
            scope="hierarchy_confirmation",
            stage="one_time_opening",
            method_started=True,
            failure_classification="confirmation_outcome",
            error=error,
            evidence={
                "attempt_marker": str(attempt_path.resolve()),
                "attempt_marker_sha256": file_hash(attempt_path),
                "sealed_roles": sorted(evidence),
                "sealed_artifacts": evidence,
            },
            context={"dataset_id": dataset_id, "candidate": context},
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--margins", required=True, type=Path)
    parser.add_argument("--data-roles", required=True, type=Path)
    parser.add_argument("--rf-profile", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-h5ad", required=True, type=Path)
    parser.add_argument("--source-recipe", required=True, type=Path)
    parser.add_argument("--source-python-config", required=True, type=Path)
    parser.add_argument("--source-custody-record", required=True, type=Path)
    parser.add_argument("--sealed-h5ad", required=True, type=Path)
    parser.add_argument("--sealed-recipe", required=True, type=Path)
    parser.add_argument("--sealed-python-config", required=True, type=Path)
    parser.add_argument("--sealed-custody-record", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    seal(
        root=arguments.root.resolve(),
        candidate_path=arguments.candidate_manifest,
        authorization_path=arguments.authorization,
        margins_path=arguments.margins,
        data_roles_path=arguments.data_roles,
        profile_path=arguments.rf_profile,
        dataset_id=arguments.dataset_id,
        sources={
            "h5ad": arguments.source_h5ad,
            "recipe": arguments.source_recipe,
            "python_config": arguments.source_python_config,
            "custody_record": arguments.source_custody_record,
        },
        destinations={
            "h5ad": arguments.sealed_h5ad,
            "recipe": arguments.sealed_recipe,
            "python_config": arguments.sealed_python_config,
            "custody_record": arguments.sealed_custody_record,
        },
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()
