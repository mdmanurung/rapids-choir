from __future__ import annotations

import pytest

pytest.importorskip("cupy")

from rapids_choir._doctor import run_doctor


@pytest.mark.gpu
def test_strict_doctor_components_pass() -> None:
    report = run_doctor("cuda:0")
    assert report.status == "PASS", {
        check.name: check.error for check in report.checks if not check.passed
    }
