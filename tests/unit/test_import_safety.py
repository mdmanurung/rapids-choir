from __future__ import annotations

import subprocess
import sys


def test_import_does_not_load_cuda_modules() -> None:
    code = """
import json
import sys
import rapids_choir
names = ['cupy', 'rmm', 'cuml', 'cugraph', 'rapids_singlecell']
print(json.dumps([name for name in names if name in sys.modules]))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "[]"


def test_cli_help_does_not_load_cuda() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "rapids_choir", "doctor", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "validate the real GPU runtime" not in completed.stderr
    assert "--strict" in completed.stdout
