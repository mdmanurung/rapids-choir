"""Versioned immutable Parquet persistence for public comparison records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ._hashing import canonical_json_bytes
from .exceptions import ArtifactCollisionError, RecordsUnavailableError
from .results import ComparisonResult, RecordsManifest

RECORDS_SCHEMA = "rapids_choir.records/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_record(record: ComparisonResult) -> str:
    return canonical_json_bytes(record.to_record()).decode("utf-8")


def write_comparison_records(
    path: str | Path,
    records: tuple[ComparisonResult, ...],
    *,
    run_id: str,
    input_sha256: str,
    overwrite: bool,
) -> RecordsManifest:
    """Write one immutable record artifact and return its verified manifest."""

    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise ArtifactCollisionError(f"records artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [_canonical_record(record) for record in records]
    schema = pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("row_index", pa.int64(), nullable=False),
            pa.field("record_json", pa.string(), nullable=False),
        ]
    )
    table = pa.table(
        {
            "schema_version": [RECORDS_SCHEMA] * len(rows),
            "run_id": [run_id] * len(rows),
            "row_index": list(range(len(rows))),
            "record_json": rows,
        },
        schema=schema,
    )
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"rapids_choir_schema": RECORDS_SCHEMA.encode("ascii"),
            b"rapids_choir_run_id": run_id.encode("utf-8"),
            b"rapids_choir_input_sha256": input_sha256.encode("ascii"),
            b"rapids_choir_row_count": str(len(rows)).encode("ascii"),
        }
    )
    table = table.replace_schema_metadata(metadata)
    temporary = destination.with_name(f".{destination.name}.{run_id}.tmp")
    if temporary.exists():
        raise ArtifactCollisionError(f"temporary records artifact exists: {temporary}")
    try:
        pq.write_table(table, temporary, compression="zstd")
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ArtifactCollisionError(
                    f"records artifact already exists: {destination}"
                ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = RecordsManifest(
        schema_version=RECORDS_SCHEMA,
        uri=str(destination),
        sha256=sha256_file(destination),
        row_count=len(rows),
        run_id=run_id,
        input_sha256=input_sha256,
    )
    read_comparison_records(manifest)
    return manifest


def read_comparison_records(manifest: RecordsManifest | dict[str, Any]) -> Any:
    """Validate identity, schema metadata, run ID, and row count before reading."""

    import pandas as pd  # type: ignore[import-untyped]
    import pyarrow.parquet as pq

    values = (
        {
            "schema_version": manifest.schema_version,
            "uri": manifest.uri,
            "sha256": manifest.sha256,
            "row_count": manifest.row_count,
            "run_id": manifest.run_id,
            "input_sha256": manifest.input_sha256,
        }
        if isinstance(manifest, RecordsManifest)
        else manifest
    )
    required = {
        "schema_version",
        "uri",
        "sha256",
        "row_count",
        "run_id",
        "input_sha256",
    }
    if set(values) != required:
        raise RecordsUnavailableError(
            "records manifest schema is incomplete or unknown"
        )
    if values["schema_version"] != RECORDS_SCHEMA:
        raise RecordsUnavailableError("unsupported records schema version")
    path = Path(str(values["uri"]))
    if not path.is_file():
        raise RecordsUnavailableError(f"records file does not exist: {path}")
    if sha256_file(path) != values["sha256"]:
        raise RecordsUnavailableError(f"records checksum mismatch: {path}")
    metadata = pq.read_metadata(path)
    arrow_schema = metadata.schema.to_arrow_schema().remove_metadata()
    observed_fields = [
        (field.name, str(field.type), field.nullable) for field in arrow_schema
    ]
    expected_fields = [
        ("schema_version", "string", False),
        ("run_id", "string", False),
        ("row_index", "int64", False),
        ("record_json", "string", False),
    ]
    if observed_fields != expected_fields:
        raise RecordsUnavailableError("records Parquet schema is not canonical")
    file_metadata = metadata.metadata or {}
    expected_metadata = {
        b"rapids_choir_schema": str(values["schema_version"]).encode("ascii"),
        b"rapids_choir_run_id": str(values["run_id"]).encode("utf-8"),
        b"rapids_choir_input_sha256": str(values["input_sha256"]).encode("ascii"),
        b"rapids_choir_row_count": str(values["row_count"]).encode("ascii"),
    }
    for key, expected in expected_metadata.items():
        if file_metadata.get(key) != expected:
            raise RecordsUnavailableError(
                f"records metadata mismatch for {key.decode('ascii')}"
            )
    row_count = values["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise RecordsUnavailableError("records row count must be an integer")
    frame = pd.read_parquet(path)
    if len(frame) != row_count:
        raise RecordsUnavailableError("records row count does not match manifest")
    if not frame.empty:
        if set(frame["schema_version"]) != {values["schema_version"]}:
            raise RecordsUnavailableError("records rows contain a schema mismatch")
        if set(frame["run_id"]) != {values["run_id"]}:
            raise RecordsUnavailableError("records rows contain a run ID mismatch")
        if frame["row_index"].tolist() != list(range(len(frame))):
            raise RecordsUnavailableError("records row ordering is not canonical")
    decoded = [json.loads(value) for value in frame["record_json"].tolist()]
    return pd.json_normalize(decoded)
