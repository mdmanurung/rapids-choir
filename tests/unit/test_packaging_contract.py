from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "run_gate1_static", SCRIPTS / "run_gate1_static.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_packaging_contract_rejects_private_and_generated_inputs() -> None:
    members = [
        "pkg/README.md",
        "pkg/workflow/authorization/README.md",
        "pkg/docs/wayfinder/private.md",
        "pkg/validation/evidence/candidates/candidate-a/gate0.json",
        "pkg/validation/logs/gate.log",
        "pkg/validation/evidence/gate1_gpu.FAILED.json",
        "pkg/workflow/authorization/comparator_confirm.json",
        "pkg/workflow/authorization/comparator_confirm.sig",
        "pkg/workflow/authorization/allowed_signers",
    ]

    forbidden = MODULE.forbidden_packaged_members(members)

    assert "pkg/README.md" not in forbidden
    assert "pkg/workflow/authorization/README.md" not in forbidden
    assert len(forbidden) == 7
