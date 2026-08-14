#!/usr/bin/env python3
"""Verify a detached SSH signature before opening confirmatory evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gate2_common import (
    file_hash,
    git_output,
    immutable_json,
    runtime_tree_digest,
)

SIGNATURE_NAMESPACE = "rapids-choir-confirmation-v1"
def _git(root: Path, *arguments: str) -> str:
    return git_output(root, *arguments)


def verify_signature(
    manifest: Path,
    signature: Path,
    allowed_signers: Path,
    *,
    operator: str,
) -> None:
    completed = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(allowed_signers),
            "-I",
            operator,
            "-n",
            SIGNATURE_NAMESPACE,
            "-s",
            str(signature),
        ],
        input=manifest.read_bytes(),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"authorization signature verification failed: {details}")


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise RuntimeError(f"authorization {field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(f"authorization {field} is invalid") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"authorization {field} must include a timezone")
    return parsed.astimezone(UTC)


def validate_authorization(
    *,
    root: Path,
    manifest: Path,
    signature: Path,
    allowed_signers: Path,
    expected_target: str,
    expected_operator: str,
    expected_roles: list[str],
    expected_attempt: int | None = None,
    expected_profile: Path | None = None,
    now: datetime | None = None,
    signature_verifier: Callable[..., None] = verify_signature,
) -> dict[str, Any]:
    """Validate signature, exact scope, candidate identity, and expiry."""

    for path in (manifest, signature, allowed_signers):
        if not path.is_file():
            raise RuntimeError(f"authorization input does not exist: {path}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("authorization manifest must be a JSON object")
    required = {
        "schema_version",
        "authorization_id",
        "authorization_kind",
        "candidate_commit",
        "runtime_tree_sha256",
        "rf_profile_sha256",
        "target",
        "operator",
        "attempt",
        "dataset_roles",
        "issued_at",
        "expires_at",
        "permit_holdout_opening",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"authorization fields missing: {missing}")
    if payload["schema_version"] != 1:
        raise RuntimeError("unsupported authorization schema version")
    if payload["authorization_kind"] != "confirmatory_opening":
        raise RuntimeError("authorization kind is not confirmatory_opening")
    if payload["target"] != expected_target:
        raise RuntimeError("authorization target does not match requested target")
    if payload["operator"] != expected_operator:
        raise RuntimeError("authorization operator does not match custodian")
    if payload["dataset_roles"] != expected_roles:
        raise RuntimeError("authorization dataset roles do not match frozen roles")
    if payload["permit_holdout_opening"] is not True:
        raise RuntimeError("authorization does not permit holdout opening")
    if expected_profile is not None and payload["rf_profile_sha256"] != file_hash(
        expected_profile
    ):
        raise RuntimeError("authorization RF profile checksum does not match")
    attempt = payload["attempt"]
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise RuntimeError("authorization attempt must be a positive integer")
    if expected_attempt is not None and attempt != expected_attempt:
        raise RuntimeError("authorization attempt does not match candidate attempt")

    issued = _timestamp(payload["issued_at"], field="issued_at")
    expires = _timestamp(payload["expires_at"], field="expires_at")
    current = datetime.now(UTC) if now is None else now.astimezone(UTC)
    if issued > current or expires <= current or expires <= issued:
        raise RuntimeError("authorization is not currently valid")

    commit = _git(root, "rev-parse", "HEAD")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("confirmatory authorization requires a clean tracked tree")
    tree_digest, runtime_paths = runtime_tree_digest(root)
    if payload["candidate_commit"] != commit:
        raise RuntimeError("authorization candidate commit does not match HEAD")
    if payload["runtime_tree_sha256"] != tree_digest:
        raise RuntimeError("authorization runtime tree digest does not match")
    if expected_profile is not None:
        profile_payload = json.loads(expected_profile.read_text(encoding="utf-8"))
        profile_runtime = profile_payload.get("calibration_candidate", {}).get(
            "runtime_tree_sha256"
        )
        if profile_payload.get("status") != "FROZEN" or not profile_runtime:
            raise RuntimeError("authorization RF profile lacks calibration identity")
        if profile_runtime != tree_digest:
            raise RuntimeError(
                "confirmation runtime differs from RF-profile calibration runtime"
            )

    signature_verifier(
        manifest,
        signature,
        allowed_signers,
        operator=expected_operator,
    )
    return {
        "schema_version": 1,
        "status": "AUTHORIZED",
        "authorization_id": payload["authorization_id"],
        "authorization_kind": payload["authorization_kind"],
        "target": expected_target,
        "operator": expected_operator,
        "attempt": attempt,
        "dataset_roles": expected_roles,
        "candidate_commit": commit,
        "runtime_tree_sha256": tree_digest,
        "rf_profile_sha256": payload["rf_profile_sha256"],
        "runtime_file_count": len(runtime_paths),
        "issued_at": issued.isoformat(),
        "expires_at": expires.isoformat(),
        "verified_at": current.isoformat(),
        "signature_namespace": SIGNATURE_NAMESPACE,
        "manifest_sha256": file_hash(manifest),
        "signature_sha256": file_hash(signature),
        "allowed_signers_sha256": file_hash(allowed_signers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--signature", required=True, type=Path)
    parser.add_argument("--allowed-signers", required=True, type=Path)
    parser.add_argument("--expected-target", required=True)
    parser.add_argument("--expected-operator", required=True)
    parser.add_argument("--expected-role", action="append", required=True)
    parser.add_argument("--expected-attempt", required=True, type=int)
    parser.add_argument("--expected-profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evidence = validate_authorization(
        root=args.root.resolve(),
        manifest=args.manifest,
        signature=args.signature,
        allowed_signers=args.allowed_signers,
        expected_target=args.expected_target,
        expected_operator=args.expected_operator,
        expected_roles=args.expected_role,
        expected_attempt=args.expected_attempt,
        expected_profile=args.expected_profile,
    )
    immutable_json(args.output, evidence)


if __name__ == "__main__":
    main()
