from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def run_status_script(
    tmp_path: Path, *, observed: str
) -> subprocess.CompletedProcess[str]:
    source = tmp_path / "source.json"
    output = tmp_path / "seal.json"
    source.write_text(json.dumps({"status": observed}), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            "validation/scripts/require_status.py",
            "--input",
            str(source),
            "--expected-status",
            "PASS",
            "--scope",
            "test",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[2],
        check=False,
        capture_output=True,
        text=True,
    )


def test_require_status_creates_seal_only_for_expected_status(
    tmp_path: Path,
) -> None:
    completed = run_status_script(tmp_path, observed="PASS")

    assert completed.returncode == 0
    assert json.loads((tmp_path / "seal.json").read_text())["status"] == "PASS"


def test_require_status_retains_negative_source_without_pass_seal(
    tmp_path: Path,
) -> None:
    completed = run_status_script(tmp_path, observed="FAIL")

    assert completed.returncode != 0
    assert (tmp_path / "source.json").is_file()
    assert not (tmp_path / "seal.json").exists()
