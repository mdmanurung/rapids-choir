#!/usr/bin/env python3
"""Verify the immutable Zenodo archive before any member is read."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from gate2_common import file_hash, immutable_json, validate_candidate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    parser.add_argument("--expected-md5", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    args = parser.parse_args()
    candidate = validate_candidate_manifest(
        args.root.resolve(), args.candidate_manifest
    )

    archive = args.archive.resolve()
    size = archive.stat().st_size
    if size != args.expected_bytes:
        raise RuntimeError(
            f"archive size mismatch: expected={args.expected_bytes}, got={size}"
        )
    actual_md5 = file_hash(archive, "md5")
    if actual_md5 != args.expected_md5:
        raise RuntimeError(
            f"archive MD5 mismatch: expected={args.expected_md5}, got={actual_md5}"
        )
    immutable_json(
        args.output,
        {
            "schema_version": 1,
            "status": "PASS",
            "verified_at": datetime.now(UTC).isoformat(),
            "archive": str(archive),
            "bytes": size,
            "md5": actual_md5,
            "candidate": {
                "commit": candidate["candidate_commit"],
                "attempt": candidate["candidate_attempt"],
                "runtime_tree_sha256": candidate["runtime_tree_sha256"],
                "manifest_sha256": file_hash(args.candidate_manifest),
                "dirty_state": candidate["dirty_state"],
            },
        },
    )


if __name__ == "__main__":
    main()
