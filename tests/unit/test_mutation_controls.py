from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_mutation_controls", SCRIPTS / "run_mutation_controls.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

MANIFEST = ROOT / "validation/contracts/parity_mutations_v1.json"


def test_all_predeclared_parity_mutations_are_detected() -> None:
    report = MODULE.run_controls(MANIFEST)
    assert report["status"] == "PASS"
    assert report["mutation_count"] == 12
    assert report["detected_count"] == 12
    assert all(record["detected"] for record in report["results"])


def test_mutation_control_rejects_incomplete_manifest(tmp_path: Path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["mutations"] = payload["mutations"][:-1]
    changed = tmp_path / "mutations.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.run_controls(changed)
