#!/usr/bin/env python3
"""Fail closed unless the frozen RF profile is committed and unchanged."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gate2_common import (
    file_hash,
    immutable_json,
    validate_candidate_manifest,
)


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def verify_profile_seal(root: Path, profile: Path) -> dict[str, str | int]:
    profile = profile.resolve()
    relative = profile.relative_to(root.resolve()).as_posix()
    payload = json.loads(profile.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or "winner" not in payload:
        raise RuntimeError("RF profile is not a frozen calibration winner")
    git(root, "ls-files", "--error-unmatch", "--", relative)
    unchanged = git(root, "diff", "--quiet", "HEAD", "--", relative, check=False)
    if unchanged.returncode != 0:
        raise RuntimeError("RF profile is tracked but uncommitted or modified")
    commit = git(root, "log", "-1", "--format=%H", "--", relative).stdout.strip()
    if not commit:
        raise RuntimeError("RF profile has no committed history")
    return {
        "schema_version": 1,
        "status": "PASS",
        "profile": relative,
        "profile_sha256": file_hash(profile),
        "profile_commit": commit,
        "checked_at_head": git(root, "rev-parse", "HEAD").stdout.strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    candidate = validate_candidate_manifest(root, args.candidate_manifest)
    evidence = verify_profile_seal(root, args.profile)
    evidence["candidate"] = {
        "attempt": candidate["candidate_attempt"],
        "runtime_tree_sha256": candidate["runtime_tree_sha256"],
        "manifest_sha256": file_hash(args.candidate_manifest),
        "dirty_state": candidate["dirty_state"],
    }
    immutable_json(args.output, evidence)


if __name__ == "__main__":
    main()
