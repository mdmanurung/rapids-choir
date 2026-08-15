"""Resolve complete comparison records from memory or immutable Parquet."""

from __future__ import annotations

from typing import Any

from .._hashing import anndata_input_hashes
from .._persistence import read_comparison_records
from ..exceptions import RecordsUnavailableError, StaleArtifactError
from ..results import ChoirResult


def records(value: Any, *, key: str | None = None) -> Any:
    """Return records as a DataFrame after complete manifest validation."""

    import pandas as pd  # type: ignore[import-untyped]

    if isinstance(value, ChoirResult):
        return pd.json_normalize(record.to_record() for record in value.records)

    uns = getattr(value, "uns", None)
    if uns is None:
        raise TypeError("records expects a ChoirResult or AnnData-like object")
    if key is not None:
        candidates = [uns[key]] if key in uns else []
    else:
        candidates = [
            entry
            for entry in uns.values()
            if isinstance(entry, dict)
            and entry.get("schema") == "rapids_choir.result/v1"
        ]
    if len(candidates) != 1:
        raise RecordsUnavailableError(
            "AnnData must resolve to exactly one rapids-choir result; pass key= "
            "when multiple results exist"
        )
    result = candidates[0]
    manifest = result.get("records_manifest")
    if manifest is None:
        raise RecordsUnavailableError("the selected result has no persisted records")
    if manifest.get("run_id") != result.get("run_id"):
        raise RecordsUnavailableError("records and result run IDs do not match")
    if manifest.get("input_sha256") != result.get("input_sha256"):
        raise RecordsUnavailableError(
            "records and result input identities do not match"
        )
    contract = result.get("input_contract")
    if not isinstance(contract, dict):
        raise RecordsUnavailableError("result lacks a complete input contract")
    try:
        current_sha256, _ = anndata_input_hashes(value, **contract)
    except (KeyError, TypeError, ValueError) as error:
        raise StaleArtifactError(
            f"cannot resolve recorded hierarchy inputs: {error}"
        ) from error
    if current_sha256 != result.get("input_sha256"):
        raise StaleArtifactError(
            "AnnData content has changed since the records artifact was created"
        )
    return read_comparison_records(manifest)
