from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "check_authorization", SCRIPTS / "check_authorization.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _run(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _candidate(root: Path) -> tuple[str, str]:
    _run(root, "init", "-q")
    (root / "src/rapids_choir").mkdir(parents=True)
    (root / "src/rapids_choir/marker.py").write_text("VALUE = 1\n")
    (root / "Snakefile").write_text("rule all:\n    input: []\n")
    _run(root, "add", "Snakefile", "src/rapids_choir/marker.py")
    _run(
        root,
        "-c",
        "user.name=Gate Test",
        "-c",
        "user.email=gate@example.invalid",
        "commit",
        "-qm",
        "freeze candidate",
    )
    commit = _run(root, "rev-parse", "HEAD")
    digest, _ = MODULE.runtime_tree_digest(root)
    return commit, digest


def _authorization(
    root: Path,
    *,
    commit: str,
    digest: str,
    now: datetime,
) -> tuple[Path, Path, Path]:
    manifest = root / "authorization.json"
    signature = root / "authorization.sig"
    allowed = root / "allowed_signers"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "authorization_id": "test-authorization",
                "authorization_kind": "confirmatory_opening",
                "candidate_commit": commit,
                "runtime_tree_sha256": digest,
                "rf_profile_sha256": "fixture-profile-sha256",
                "target": "comparator_confirm",
                "operator": "mdmanurung",
                "attempt": 1,
                "dataset_roles": ["comparator_holdout"],
                "issued_at": (now - timedelta(minutes=1)).isoformat(),
                "expires_at": (now + timedelta(minutes=1)).isoformat(),
                "permit_holdout_opening": True,
            }
        ),
        encoding="utf-8",
    )
    signature.write_text("detached signature fixture\n", encoding="utf-8")
    allowed.write_text("mdmanurung ssh-ed25519 fixture\n", encoding="utf-8")
    return manifest, signature, allowed


def test_authorization_binds_clean_candidate_and_scope(tmp_path: Path) -> None:
    commit, digest = _candidate(tmp_path)
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    manifest, signature, allowed = _authorization(
        tmp_path, commit=commit, digest=digest, now=now
    )
    verified: list[str] = []

    def fake_verify(*args: object, operator: str) -> None:
        verified.append(operator)

    evidence = MODULE.validate_authorization(
        root=tmp_path,
        manifest=manifest,
        signature=signature,
        allowed_signers=allowed,
        expected_target="comparator_confirm",
        expected_operator="mdmanurung",
        expected_roles=["comparator_holdout"],
        now=now,
        signature_verifier=fake_verify,
    )
    assert evidence["status"] == "AUTHORIZED"
    assert evidence["candidate_commit"] == commit
    assert verified == ["mdmanurung"]


def test_authorization_rejects_candidate_drift(tmp_path: Path) -> None:
    commit, digest = _candidate(tmp_path)
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    manifest, signature, allowed = _authorization(
        tmp_path, commit=commit, digest=digest, now=now
    )
    (tmp_path / "src/rapids_choir/marker.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="clean tracked tree"):
        MODULE.validate_authorization(
            root=tmp_path,
            manifest=manifest,
            signature=signature,
            allowed_signers=allowed,
            expected_target="comparator_confirm",
            expected_operator="mdmanurung",
            expected_roles=["comparator_holdout"],
            now=now,
            signature_verifier=lambda *args, **kwargs: None,
        )


def test_immutable_json_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    MODULE.immutable_json(path, {"attempt": 1})
    with pytest.raises(RuntimeError, match="already exists"):
        MODULE.immutable_json(path, {"attempt": 2})


def test_authorization_rejects_profile_runtime_drift(tmp_path: Path) -> None:
    commit, digest = _candidate(tmp_path)
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    manifest, signature, allowed = _authorization(
        tmp_path, commit=commit, digest=digest, now=now
    )
    profile = tmp_path / "rf_profile.json"
    profile.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "calibration_candidate": {
                    "runtime_tree_sha256": "different-runtime"
                },
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rf_profile_sha256"] = MODULE.file_hash(profile)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="differs from RF-profile"):
        MODULE.validate_authorization(
            root=tmp_path,
            manifest=manifest,
            signature=signature,
            allowed_signers=allowed,
            expected_target="comparator_confirm",
            expected_operator="mdmanurung",
            expected_roles=["comparator_holdout"],
            expected_profile=profile,
            now=now,
            signature_verifier=lambda *args, **kwargs: None,
        )
