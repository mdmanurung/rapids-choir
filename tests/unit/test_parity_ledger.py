from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from validation.scripts.assess_parity_promotion import (
    HIERARCHY_EVIDENCE_ENTRIES,
    _gate_candidate_matches,
    unresolved_ledger_entries,
)

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "verify_parity_ledger", SCRIPTS / "verify_parity_ledger.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

LEDGER_PATH = ROOT / "validation/contracts/parity_ledger_v1.json"
REFERENCE_ROOT = ROOT / "validation/reference/choir_v0_3_0/R"


def _ledger() -> dict[str, object]:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _errors(ledger: dict[str, object]) -> list[str]:
    functions = MODULE.inventory(REFERENCE_ROOT, ROOT)
    return MODULE.validate_ledger(
        ledger, functions=functions, repo_root=ROOT
    )


def test_parity_ledger_covers_every_r_function_without_unclassified_entries() -> None:
    ledger = _ledger()
    functions = MODULE.inventory(REFERENCE_ROOT, ROOT)
    assert len(functions) == 45
    assert len(ledger["entries"]) == 45  # type: ignore[arg-type]
    assert _errors(ledger) == []


def test_candidate_hierarchy_evidence_closes_only_predeclared_ledger_entries() -> None:
    ledger = _ledger()
    assert set(
        unresolved_ledger_entries(
            ledger,
            hierarchy_complete=False,
            margins_frozen=False,
        )
    ) == HIERARCHY_EVIDENCE_ENTRIES
    assert (
        unresolved_ledger_entries(
            ledger,
            hierarchy_complete=True,
            margins_frozen=True,
        )
        == []
    )


def test_promotion_rejects_drift_in_every_candidate_identity_field() -> None:
    context = {
        "commit": "a" * 40,
        "attempt": 3,
        "runtime_tree_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    evidence = {"candidate": dict(context)}
    assert _gate_candidate_matches(evidence, context)
    for field in context:
        drifted = copy.deepcopy(evidence)
        drifted["candidate"][field] = "drift"
        assert not _gate_candidate_matches(drifted, context)


def test_parity_ledger_rejects_missing_function() -> None:
    ledger = _ledger()
    ledger["entries"] = ledger["entries"][:-1]  # type: ignore[index]
    assert any("unmapped R functions" in error for error in _errors(ledger))


def test_parity_ledger_rejects_source_or_evidence_drift() -> None:
    source_drift = copy.deepcopy(_ledger())
    source_drift["entries"][0]["r_source"]["sha256"] = "0" * 64  # type: ignore[index]
    assert any(
        "source span or digest drift" in error
        for error in _errors(source_drift)
    )

    evidence_drift = copy.deepcopy(_ledger())
    evidence_drift["entries"][0]["evidence"]["sha256"] = "0" * 64  # type: ignore[index]
    assert any("evidence checksum drift" in error for error in _errors(evidence_drift))


def test_parity_ledger_rejects_unclassified_or_unexplained_branches() -> None:
    unclassified = copy.deepcopy(_ledger())
    branch = unclassified["entries"][0]["meaningful_branches"][0]  # type: ignore[index]
    branch["classification"] = "todo"
    assert any("unclassified" in error for error in _errors(unclassified))

    unexplained = copy.deepcopy(_ledger())
    branch = unexplained["entries"][0]["meaningful_branches"][0]  # type: ignore[index]
    branch["explanation"] = ""
    assert any("unexplained" in error for error in _errors(unexplained))
