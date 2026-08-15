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


def test_authorization_requires_final_candidate_profile_reproduction(
    tmp_path: Path,
) -> None:
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
                "winner": {"profile_id": "winner", "profile": {}},
                "calibration_candidate": {
                    "runtime_tree_sha256": "different-runtime"
                },
            }
        ),
        encoding="utf-8",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rf_profile_sha256"] = MODULE.file_hash(profile)
    reproduced_profile = tmp_path / "rf_profile.final-candidate.json"
    reproduced_profile.write_text("{}\n", encoding="utf-8")
    reproduction = tmp_path / "rf_profile.REPRODUCED.json"
    reproduction.write_text(
        json.dumps(
            {
                "status": "REPRODUCED",
                "candidate": {
                    "commit": commit,
                    "attempt": 1,
                    "runtime_tree_sha256": digest,
                },
                "exact_winning_profile": True,
                "exact_reference_variability_margins": True,
                "protocol_checks": {
                    "ranking_order": True,
                    "comparator_contract_sha256": True,
                    "environment_lock_sha256": True,
                },
                "frozen": {"sha256": MODULE.file_hash(profile)},
                "reproduced": {
                    "path": str(reproduced_profile),
                    "sha256": MODULE.file_hash(reproduced_profile),
                },
            }
        ),
        encoding="utf-8",
    )
    payload["rf_profile_reproduction_sha256"] = MODULE.file_hash(reproduction)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    evidence = MODULE.validate_authorization(
        root=tmp_path,
        manifest=manifest,
        signature=signature,
        allowed_signers=allowed,
        expected_target="comparator_confirm",
        expected_operator="mdmanurung",
        expected_roles=["comparator_holdout"],
        expected_profile=profile,
        expected_profile_reproduction=reproduction,
        now=now,
        signature_verifier=lambda *args, **kwargs: None,
    )
    assert evidence["rf_profile_reproduction_sha256"] == MODULE.file_hash(
        reproduction
    )

    reproduction_payload = json.loads(reproduction.read_text(encoding="utf-8"))
    reproduction_payload["candidate"]["runtime_tree_sha256"] = "drifted"
    reproduction.write_text(json.dumps(reproduction_payload), encoding="utf-8")
    payload["rf_profile_reproduction_sha256"] = MODULE.file_hash(reproduction)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="did not reproduce"):
        MODULE.validate_authorization(
            root=tmp_path,
            manifest=manifest,
            signature=signature,
            allowed_signers=allowed,
            expected_target="comparator_confirm",
            expected_operator="mdmanurung",
            expected_roles=["comparator_holdout"],
            expected_profile=profile,
            expected_profile_reproduction=reproduction,
            now=now,
            signature_verifier=lambda *args, **kwargs: None,
        )


def test_v2_authorization_binds_hierarchy_dataset_and_artifacts(
    tmp_path: Path,
) -> None:
    commit, digest = _candidate(tmp_path)
    now = datetime(2026, 8, 14, 12, tzinfo=UTC)
    margins = tmp_path / "margins.json"
    roles = tmp_path / "roles.json"
    profile = tmp_path / "rf_profile.json"
    margins.write_text("{}\n", encoding="utf-8")
    roles.write_text("{}\n", encoding="utf-8")
    profile.write_text(
        json.dumps({"status": "FROZEN", "winner": {"profile": {}}}),
        encoding="utf-8",
    )
    raw = {
        "h5ad": "a" * 64,
        "recipe": "b" * 64,
        "python_config": "c" * 64,
        "custody_record": "d" * 64,
    }
    manifest = tmp_path / "authorization-v2.json"
    signature = tmp_path / "authorization-v2.sig"
    allowed = tmp_path / "allowed_signers-v2"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "authorization_id": "hierarchy-arigoni-test",
                "authorization_kind": "confirmatory_opening",
                "candidate_commit": commit,
                "runtime_tree_sha256": digest,
                "target": "hierarchy_confirm_Arigoni",
                "operator": "mdmanurung",
                "attempt": 1,
                "dataset_roles": ["hierarchy_parity_confirmation:Arigoni"],
                "issued_at": (now - timedelta(minutes=1)).isoformat(),
                "expires_at": (now + timedelta(minutes=1)).isoformat(),
                "permit_holdout_opening": True,
                "bound_artifacts": {
                    **raw,
                    "hierarchy_margins": MODULE.file_hash(margins),
                    "data_roles": MODULE.file_hash(roles),
                    "rf_profile": MODULE.file_hash(profile),
                },
            }
        ),
        encoding="utf-8",
    )
    signature.write_text("fixture\n", encoding="utf-8")
    allowed.write_text("fixture\n", encoding="utf-8")
    namespaces: list[str] = []

    def fake_verify(
        *args: object, operator: str, namespace: str
    ) -> None:
        assert operator == "mdmanurung"
        namespaces.append(namespace)

    evidence = MODULE.validate_authorization(
        root=tmp_path,
        manifest=manifest,
        signature=signature,
        allowed_signers=allowed,
        expected_target="hierarchy_confirm_Arigoni",
        expected_operator="mdmanurung",
        expected_roles=["hierarchy_parity_confirmation:Arigoni"],
        expected_attempt=1,
        expected_profile=profile,
        expected_bound_artifacts={
            "hierarchy_margins": margins,
            "data_roles": roles,
            "rf_profile": profile,
        },
        required_bound_roles=set(raw),
        now=now,
        signature_verifier=fake_verify,
    )
    assert evidence["status"] == "AUTHORIZED"
    assert evidence["bound_artifacts"]["h5ad"] == "a" * 64
    assert namespaces == ["rapids-choir-confirmation-v2"]
