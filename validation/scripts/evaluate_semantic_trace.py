#!/usr/bin/env python3
"""Compare one reference trace to one Python trace under frozen margins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rapids_choir._semantic_trace import read_trace, read_trace_parquet
from rapids_choir._trace_compare import compare_semantic_traces
from validation.scripts.gate2_common import immutable_json


def _load_trace(path: Path) -> tuple[Any, ...]:
    if path.suffix == ".jsonl":
        return read_trace(path)
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        return read_trace_parquet(manifest)
    raise ValueError("trace must be JSONL or a JSON Parquet manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    contract = json.loads(arguments.contract.read_text(encoding="utf-8"))
    comparison = compare_semantic_traces(
        _load_trace(arguments.reference),
        _load_trace(arguments.candidate),
        contract,
    )
    payload = json.dumps(
        comparison.to_record(), indent=2, sort_keys=True
    ) + "\n"
    if arguments.output is None:
        print(payload, end="")
    else:
        immutable_json(arguments.output, comparison.to_record())
    return 0 if comparison.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
