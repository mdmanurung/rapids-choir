#!/usr/bin/env python3
"""Seal one clean runtime tree before any candidate evidence is produced."""

from __future__ import annotations

import argparse
import socket
from datetime import UTC, datetime
from pathlib import Path

from gate2_common import (
    file_hash,
    git_output,
    immutable_json,
    runtime_tree_digest,
)


def freeze_candidate(
    *,
    root: Path,
    expected_commit: str,
    attempt: int,
    controlled_files: list[Path],
) -> dict[str, object]:
    if isinstance(attempt, bool) or attempt < 1:
        raise RuntimeError("candidate attempt must be a positive integer")
    current_commit = git_output(root, "rev-parse", "HEAD")
    if current_commit != expected_commit:
        raise RuntimeError("candidate commit changed after DAG construction")
    tracked_dirty = git_output(
        root, "status", "--porcelain", "--untracked-files=no"
    )
    if tracked_dirty:
        raise RuntimeError("candidate freeze requires a clean tracked tree")
    runtime_digest, runtime_paths = runtime_tree_digest(root)

    controlled_hashes: dict[str, str] = {}
    for path in controlled_files:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"runtime-controlled file is outside repository: {path}"
            ) from error
        if not resolved.is_file():
            raise RuntimeError(f"runtime-controlled file is missing: {relative}")
        controlled_hashes[relative] = file_hash(resolved)

    return {
        "schema_version": 1,
        "status": "FROZEN",
        "generated_at": datetime.now(UTC).isoformat(),
        "candidate_commit": current_commit,
        "candidate_attempt": attempt,
        "candidate_id": f"candidate-{current_commit}/attempt-{attempt}",
        "runtime_tree_sha256": runtime_digest,
        "runtime_file_count": len(runtime_paths),
        "workflow_sha256": controlled_hashes.get("Snakefile"),
        "controlled_file_hashes": dict(sorted(controlled_hashes.items())),
        "dirty_state": {
            "tracked": False,
            "untracked_files_excluded": True,
        },
        "runtime_scope": "all_tracked_repository_files",
        "host": socket.gethostname(),
        "failure_classification": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--controlled-file", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = freeze_candidate(
        root=args.root.resolve(),
        expected_commit=args.expected_commit,
        attempt=args.attempt,
        controlled_files=args.controlled_file,
    )
    immutable_json(args.output, payload)


if __name__ == "__main__":
    main()
