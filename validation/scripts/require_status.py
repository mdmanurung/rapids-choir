#!/usr/bin/env python3
"""Create an immutable pass seal only for an explicitly expected status."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from gate2_common import file_hash, immutable_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-status", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    observed = payload.get("status")
    if observed != args.expected_status:
        raise RuntimeError(
            f"{args.scope} requires status {args.expected_status}, got {observed}"
        )
    immutable_json(
        args.output,
        {
            "schema_version": 1,
            "status": "PASS",
            "scope": args.scope,
            "expected_source_status": args.expected_status,
            "source": str(args.input.resolve()),
            "source_sha256": file_hash(args.input),
            "sealed_at": datetime.now(UTC).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
