"""Resolve full comparison records from a result or persisted manifest."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
from typing import Any

from ..exceptions import RecordsUnavailableError
from ..results import ChoirResult


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(value: Any) -> Any:
    """Return records as a pandas DataFrame, validating persisted checksums."""

    pd: Any = importlib.import_module("pandas")

    if isinstance(value, ChoirResult):
        return pd.json_normalize(record.to_record() for record in value.records)

    uns = getattr(value, "uns", None)
    if uns is None:
        raise TypeError("records expects a ChoirResult or AnnData-like object")
    candidates = [
        entry.get("records_manifest")
        for entry in uns.values()
        if isinstance(entry, dict) and entry.get("schema") == "rapids_choir/v1"
    ]
    manifests = [manifest for manifest in candidates if manifest is not None]
    if len(manifests) != 1:
        raise RecordsUnavailableError(
            "AnnData must contain exactly one rapids-choir records manifest"
        )
    manifest = manifests[0]
    path = Path(manifest["uri"])
    if not path.is_file():
        raise RecordsUnavailableError(f"records file does not exist: {path}")
    if _sha256(path) != manifest["sha256"]:
        raise RecordsUnavailableError(f"records checksum mismatch: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != manifest["row_count"]:
        raise RecordsUnavailableError("records row count does not match manifest")
    return frame
