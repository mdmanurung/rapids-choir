"""Validation-only canonical semantic event instrumentation.

This module is intentionally private. Trace generation is excluded from public
API guarantees and benchmark timing boundaries.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from ._hashing import canonical_json_bytes, sha256_bytes
from .exceptions import ArtifactCollisionError

TRACE_SCHEMA = "rapids_choir.semantic/v1"
TRACE_PARQUET_SCHEMA = "rapids_choir.semantic-parquet/v1"
TRACE_ARTIFACT_SCHEMA = "rapids_choir.semantic-artifact/v1"

EVENT_TYPES = frozenset(
    {
        "provenance",
        "preprocessing",
        "prepared_input_validation",
        "resolution_search",
        "resolution_transition",
        "hierarchy_node",
        "hierarchy_stop",
        "comparator_call",
        "multiplicity",
        "prune",
        "bridge",
        "merge",
        "singleton",
        "final_relabel",
    }
)
EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "operation_index",
        "event_type",
        "phase",
        "parameters",
        "seed",
        "input_membership_hashes",
        "output_membership_hashes",
        "topology_context",
        "decision",
        "reason",
        "bounded_values",
        "payload",
    }
)


@dataclass(frozen=True, slots=True)
class SemanticEvent:
    run_id: str
    operation_index: int
    event_type: str
    phase: str
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    input_membership_hashes: tuple[str, ...] = ()
    output_membership_hashes: tuple[str, ...] = ()
    topology_context: dict[str, Any] = field(default_factory=dict)
    decision: str | None = None
    reason: str | None = None
    bounded_values: dict[str, float] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TRACE_SCHEMA:
            raise ValueError("unsupported semantic event schema")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("semantic event run_id cannot be empty")
        if (
            isinstance(self.operation_index, bool)
            or not isinstance(self.operation_index, int)
            or self.operation_index < 0
        ):
            raise ValueError(
                "semantic event operation_index must be a non-negative integer"
            )
        if not isinstance(self.event_type, str) or self.event_type not in EVENT_TYPES:
            raise ValueError(f"unsupported semantic event type: {self.event_type}")
        if not isinstance(self.phase, str) or not self.phase:
            raise ValueError("semantic event phase cannot be empty")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("semantic event seed must be an integer or None")
        if self.decision is not None and not isinstance(self.decision, str):
            raise TypeError("semantic event decision must be text or None")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("semantic event reason must be text or None")
        for name, object_value in (
            ("parameters", self.parameters),
            ("topology_context", self.topology_context),
            ("bounded_values", self.bounded_values),
            ("payload", self.payload),
        ):
            if not isinstance(object_value, dict) or any(
                not isinstance(key, str) for key in object_value
            ):
                raise TypeError(f"semantic event {name} must be a string-keyed object")
        for membership_hash in (
            *self.input_membership_hashes,
            *self.output_membership_hashes,
        ):
            if (
                not isinstance(membership_hash, str)
                or len(membership_hash) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in membership_hash
                )
            ):
                raise ValueError(
                    "semantic membership identities must be lowercase SHA-256"
                )
        for name, bounded_value in self.bounded_values.items():
            if (
                isinstance(bounded_value, bool)
                or not isinstance(bounded_value, (int, float))
                or not isfinite(bounded_value)
                or not 0.0 <= bounded_value <= 1.0
            ):
                raise ValueError(f"bounded semantic value {name!r} must lie in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "operation_index": self.operation_index,
            "event_type": self.event_type,
            "phase": self.phase,
            "parameters": self.parameters,
            "seed": self.seed,
            "input_membership_hashes": list(self.input_membership_hashes),
            "output_membership_hashes": list(self.output_membership_hashes),
            "topology_context": self.topology_context,
            "decision": self.decision,
            "reason": self.reason,
            "bounded_values": self.bounded_values,
            "payload": self.payload,
        }

    @classmethod
    def from_record(cls, value: dict[str, Any]) -> SemanticEvent:
        if not isinstance(value, dict) or set(value) != EVENT_FIELDS:
            raise ValueError("semantic event record is incomplete or unknown")
        for name in ("schema_version", "run_id", "event_type", "phase"):
            if not isinstance(value[name], str):
                raise TypeError(f"semantic event {name} must be text")
        if isinstance(value["operation_index"], bool) or not isinstance(
            value["operation_index"], int
        ):
            raise TypeError("semantic event operation_index must be an integer")
        for name in ("parameters", "topology_context", "bounded_values", "payload"):
            if not isinstance(value[name], dict):
                raise TypeError(f"semantic event {name} must be an object")
        for name in ("input_membership_hashes", "output_membership_hashes"):
            if not isinstance(value[name], list):
                raise TypeError(f"semantic event {name} must be an array")
        if value["decision"] is not None and not isinstance(value["decision"], str):
            raise TypeError("semantic event decision must be text or null")
        if value["reason"] is not None and not isinstance(value["reason"], str):
            raise TypeError("semantic event reason must be text or null")
        raw_seed = value["seed"]
        if raw_seed is not None and (
            isinstance(raw_seed, bool) or not isinstance(raw_seed, int)
        ):
            raise TypeError("semantic event seed must be an integer or null")
        for name, number in value["bounded_values"].items():
            if (
                not isinstance(name, str)
                or isinstance(number, bool)
                or not isinstance(number, (int, float))
            ):
                raise TypeError("semantic bounded values must be named numbers")
        return cls(
            schema_version=value["schema_version"],
            run_id=value["run_id"],
            operation_index=value["operation_index"],
            event_type=value["event_type"],
            phase=value["phase"],
            parameters=dict(value["parameters"]),
            seed=raw_seed,
            input_membership_hashes=tuple(value["input_membership_hashes"]),
            output_membership_hashes=tuple(value["output_membership_hashes"]),
            topology_context=dict(value["topology_context"]),
            decision=value.get("decision"),
            reason=value.get("reason"),
            bounded_values={
                str(name): float(number)
                for name, number in value["bounded_values"].items()
            },
            payload=dict(value["payload"]),
        )


@dataclass(frozen=True, slots=True)
class SemanticTraceManifest:
    schema_version: str
    uri: str
    sha256: str
    row_count: int
    run_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TraceRecorder:
    """Append-only in-memory trace with canonical serialization."""

    __slots__ = ("_events", "run_id")

    def __init__(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("trace run_id cannot be empty")
        self.run_id = run_id
        self._events: list[SemanticEvent] = []

    @property
    def events(self) -> tuple[SemanticEvent, ...]:
        return tuple(self._events)

    def append(
        self,
        event_type: str,
        phase: str,
        **values: Any,
    ) -> SemanticEvent:
        event = SemanticEvent(
            run_id=self.run_id,
            operation_index=len(self._events),
            event_type=event_type,
            phase=phase,
            **values,
        )
        self._events.append(event)
        return event

    def canonical_jsonl(self) -> bytes:
        return b"".join(
            canonical_json_bytes(event.to_record()) + b"\n" for event in self._events
        )

    def sha256(self) -> str:
        return sha256_bytes(self.canonical_jsonl())

    def write_jsonl(self, path: str | Path, *, overwrite: bool = False) -> Path:
        destination = Path(path)
        if destination.exists() and not overwrite:
            raise ArtifactCollisionError(
                f"semantic trace already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{self.run_id}.tmp")
        if temporary.exists():
            raise ArtifactCollisionError(
                f"temporary semantic trace exists: {temporary}"
            )
        try:
            temporary.write_bytes(self.canonical_jsonl())
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError as error:
                    raise ArtifactCollisionError(
                        f"semantic trace already exists: {destination}"
                    ) from error
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def write_parquet(
        self, path: str | Path, *, overwrite: bool = False
    ) -> SemanticTraceManifest:
        """Write an immutable Parquet envelope around canonical event JSON."""

        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        destination = Path(path).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise ArtifactCollisionError(
                f"semantic trace already exists: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            canonical_json_bytes(event.to_record()).decode("utf-8")
            for event in self._events
        ]
        schema = pa.schema(
            [
                pa.field("schema_version", pa.string(), nullable=False),
                pa.field("run_id", pa.string(), nullable=False),
                pa.field("operation_index", pa.int64(), nullable=False),
                pa.field("event_json", pa.string(), nullable=False),
            ]
        )
        table = pa.table(
            {
                "schema_version": [TRACE_SCHEMA] * len(rows),
                "run_id": [self.run_id] * len(rows),
                "operation_index": list(range(len(rows))),
                "event_json": rows,
            },
            schema=schema,
        )
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"rapids_choir_schema": TRACE_PARQUET_SCHEMA.encode("ascii"),
                b"rapids_choir_event_schema": TRACE_SCHEMA.encode("ascii"),
                b"rapids_choir_run_id": self.run_id.encode("utf-8"),
                b"rapids_choir_row_count": str(len(rows)).encode("ascii"),
            }
        )
        table = table.replace_schema_metadata(metadata)
        temporary = destination.with_name(f".{destination.name}.{self.run_id}.tmp")
        if temporary.exists():
            raise ArtifactCollisionError(
                f"temporary semantic trace exists: {temporary}"
            )
        try:
            pq.write_table(table, temporary, compression="zstd")
            if overwrite:
                os.replace(temporary, destination)
            else:
                try:
                    os.link(temporary, destination)
                except FileExistsError as error:
                    raise ArtifactCollisionError(
                        f"semantic trace already exists: {destination}"
                    ) from error
        finally:
            if temporary.exists():
                temporary.unlink()
        manifest = SemanticTraceManifest(
            schema_version=TRACE_PARQUET_SCHEMA,
            uri=str(destination),
            sha256=_sha256_file(destination),
            row_count=len(rows),
            run_id=self.run_id,
        )
        read_trace_parquet(manifest)
        return manifest


def read_trace(path: str | Path) -> tuple[SemanticEvent, ...]:
    source = Path(path)
    events = tuple(
        SemanticEvent.from_record(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    for expected, event in enumerate(events):
        if event.operation_index != expected:
            raise ValueError("semantic trace operation order is not contiguous")
    if events and len({event.run_id for event in events}) != 1:
        raise ValueError("semantic trace contains multiple run IDs")
    return events


def read_trace_parquet(
    manifest: SemanticTraceManifest | dict[str, Any],
) -> tuple[SemanticEvent, ...]:
    """Read a trace only after checksum, metadata, schema, and order checks."""

    import pyarrow.parquet as pq

    values = (
        asdict(manifest)
        if isinstance(manifest, SemanticTraceManifest)
        else manifest
    )
    required = {"schema_version", "uri", "sha256", "row_count", "run_id"}
    if set(values) != required:
        raise ValueError("semantic trace manifest is incomplete or unknown")
    if values["schema_version"] != TRACE_PARQUET_SCHEMA:
        raise ValueError("unsupported semantic trace Parquet schema")
    row_count = values["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise ValueError("semantic trace row count must be an integer")
    path = Path(str(values["uri"]))
    if not path.is_file() or _sha256_file(path) != values["sha256"]:
        raise ValueError("semantic trace checksum or path is invalid")
    metadata = pq.read_metadata(path)
    file_metadata = metadata.metadata or {}
    expected_metadata = {
        b"rapids_choir_schema": TRACE_PARQUET_SCHEMA.encode("ascii"),
        b"rapids_choir_event_schema": TRACE_SCHEMA.encode("ascii"),
        b"rapids_choir_run_id": str(values["run_id"]).encode("utf-8"),
        b"rapids_choir_row_count": str(row_count).encode("ascii"),
    }
    if any(
        file_metadata.get(key) != expected
        for key, expected in expected_metadata.items()
    ):
        raise ValueError("semantic trace Parquet metadata mismatch")
    arrow_schema = metadata.schema.to_arrow_schema().remove_metadata()
    observed_fields = [
        (field.name, str(field.type), field.nullable) for field in arrow_schema
    ]
    expected_fields = [
        ("schema_version", "string", False),
        ("run_id", "string", False),
        ("operation_index", "int64", False),
        ("event_json", "string", False),
    ]
    if observed_fields != expected_fields:
        raise ValueError("semantic trace Parquet fields are not canonical")
    table = pq.read_table(path)
    if table.num_rows != row_count:
        raise ValueError("semantic trace row count does not match manifest")
    records = table.to_pydict()
    if records["operation_index"] != list(range(row_count)):
        raise ValueError("semantic trace operation order is not contiguous")
    if set(records["run_id"]) - {values["run_id"]}:
        raise ValueError("semantic trace rows contain a run ID mismatch")
    if set(records["schema_version"]) - {TRACE_SCHEMA}:
        raise ValueError("semantic trace rows contain a schema mismatch")
    events = tuple(
        SemanticEvent.from_record(json.loads(value))
        for value in records["event_json"]
    )
    for index, event in enumerate(events):
        if event.operation_index != index or event.run_id != values["run_id"]:
            raise ValueError("semantic trace JSON payload identity mismatch")
    return events


def semantic_trace_reference(
    path: str | Path,
    *,
    format: str,
    relative_to: str | Path | None = None,
) -> dict[str, Any]:
    """Create a content-bound reference to one canonical trace artifact."""

    source = Path(path).expanduser().resolve()
    if format == "jsonl":
        events = read_trace(source)
    elif format == "parquet_manifest":
        manifest = json.loads(source.read_text(encoding="utf-8"))
        events = read_trace_parquet(manifest)
    else:
        raise ValueError("unsupported semantic trace artifact format")
    if not events:
        raise ValueError("semantic trace artifact cannot be empty")
    run_ids = {event.run_id for event in events}
    if len(run_ids) != 1:
        raise ValueError("semantic trace artifact must contain one run ID")
    stored_path = str(source)
    if relative_to is not None:
        base = Path(relative_to).expanduser().resolve()
        stored_path = os.path.relpath(source, start=base)
    return {
        "schema_version": TRACE_ARTIFACT_SCHEMA,
        "format": format,
        "path": stored_path,
        "sha256": _sha256_file(source),
        "row_count": len(events),
        "run_id": next(iter(run_ids)),
        "event_schema": TRACE_SCHEMA,
    }


def read_trace_reference(
    reference: dict[str, Any],
    *,
    relative_to: str | Path,
) -> tuple[SemanticEvent, ...]:
    """Load a trace reference after validating identity and content custody."""

    required = {
        "schema_version",
        "format",
        "path",
        "sha256",
        "row_count",
        "run_id",
        "event_schema",
    }
    if set(reference) != required:
        raise ValueError("semantic trace reference is incomplete or unknown")
    if reference["schema_version"] != TRACE_ARTIFACT_SCHEMA:
        raise ValueError("unsupported semantic trace artifact schema")
    if reference["event_schema"] != TRACE_SCHEMA:
        raise ValueError("semantic trace event schema has drifted")
    row_count = reference["row_count"]
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise ValueError("semantic trace reference row count must be an integer")
    source = Path(str(reference["path"]))
    if not source.is_absolute():
        source = Path(relative_to).expanduser().resolve() / source
    if not source.is_file() or _sha256_file(source) != reference["sha256"]:
        raise ValueError("semantic trace reference checksum or path is invalid")
    if reference["format"] == "jsonl":
        events = read_trace(source)
    elif reference["format"] == "parquet_manifest":
        manifest = json.loads(source.read_text(encoding="utf-8"))
        events = read_trace_parquet(manifest)
    else:
        raise ValueError("unsupported semantic trace artifact format")
    if len(events) != row_count:
        raise ValueError("semantic trace reference row count mismatch")
    if not events or {event.run_id for event in events} != {reference["run_id"]}:
        raise ValueError("semantic trace reference run ID mismatch")
    return events
