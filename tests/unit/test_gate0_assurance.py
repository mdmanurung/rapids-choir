from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "check_gate0_assurance", SCRIPTS / "check_gate0.py"
)
assert SPEC is not None and SPEC.loader is not None
CHECK_GATE0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK_GATE0)


def test_gate0_verifies_complete_parity_assurance_design() -> None:
    report = CHECK_GATE0.verify_assurance_contracts(ROOT)
    assert report["r_function_count"] == 45
    assert report["ledger_entry_count"] == 45
    assert report["unclassified_count"] == 0
    assert report["margin_status"] == "DEVELOPMENT_REQUIRED"
    assert report["publication_contract"] == "narrow-software-publication-v2"
    assert report["data_role_contract"] == "publication-data-roles-v2"
    assert report["mutation_strategy"] == "temporary_source_patch"
    assert report["hierarchy_producer_count"] == 21
    assert report["secondary_dataset_count"] == 3
    assert report["sant_analysis_count"] == 4
    assert report["performance_workload_count"] == 6
    assert report["promotion_ready"] is False
