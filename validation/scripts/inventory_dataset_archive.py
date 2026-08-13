#!/usr/bin/env python3
"""Write a non-extracting inventory of the verified tar archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from gate2_common import atomic_json, file_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    archive_path = args.archive.resolve()
    if verification.get("status") != "PASS":
        raise RuntimeError("archive verification evidence is not PASS")
    if Path(verification.get("archive", "")).resolve() != archive_path:
        raise RuntimeError("archive path does not match verification evidence")
    if verification.get("bytes") != archive_path.stat().st_size:
        raise RuntimeError("archive size does not match verification evidence")
    archive_md5 = file_hash(archive_path, "md5")
    if verification.get("md5") != archive_md5:
        raise RuntimeError("archive MD5 does not match verification evidence")
    members = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            members.append(
                {
                    "name": member.name,
                    "bytes": member.size,
                    "type": "file" if member.isfile() else "other",
                }
            )
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "archive": str(archive_path),
            "archive_md5": archive_md5,
            "member_count": len(members),
            "members": members,
        },
    )


if __name__ == "__main__":
    main()
