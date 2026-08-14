from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gate2_common", ROOT / "validation/scripts/gate2_common.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_failure_record_is_machine_readable_and_immutable(tmp_path: Path) -> None:
    output = tmp_path / "gate.json"
    error = RuntimeError("fixture failure")
    failure = MODULE.preserve_failure_record(
        output,
        scope="fixture_gate",
        stage="preflight",
        method_started=False,
        failure_classification="custody_or_preflight_rejection",
        error=error,
        context={"attempt": 1},
    )
    payload = json.loads(failure.read_text(encoding="utf-8"))
    assert failure.name == "gate.FAILED.json"
    assert payload["status"] == "FAIL"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["context"] == {"attempt": 1}

    with pytest.raises(RuntimeError, match="already exists"):
        MODULE.preserve_failure_record(
            output,
            scope="fixture_gate",
            stage="preflight",
            method_started=False,
            failure_classification="custody_or_preflight_rejection",
            error=error,
        )


def test_cli_path_argument_requires_one_following_value() -> None:
    assert MODULE.cli_path_argument("--output", ["tool", "--output", "x.json"]) == (
        Path("x.json")
    )
    assert MODULE.cli_path_argument("--output", ["tool", "--output"]) is None
