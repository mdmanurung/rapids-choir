#!/usr/bin/env python3
"""Shared file/provenance helpers for Gate 2 scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def identifier_hash(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def dataset_number(dataset_id: str) -> int:
    try:
        return int(dataset_id.rsplit(" ", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"unsupported frozen dataset id: {dataset_id}") from error


def dataset_slug(dataset_id: str) -> str:
    return f"simulated_dataset_{dataset_number(dataset_id):03d}"
