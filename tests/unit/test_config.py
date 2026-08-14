from __future__ import annotations

import json
from pathlib import Path

import pytest

from rapids_choir import ChoirConfig, ComparisonConfig, RFProfile, RuleSet
from rapids_choir.config import split_round_count
from rapids_choir.exceptions import InvalidConfigurationError, MissingCalibrationError


@pytest.mark.parametrize(
    ("requested", "rounds"),
    [(1, 0), (3, 2), (5, 2), (7, 4), (9, 4), (101, 50)],
)
def test_compatibility_rounding_matches_r_ties_to_even(
    requested: int, rounds: int
) -> None:
    config = ComparisonConfig(
        rules="choir_v0_3_0", n_iterations=requested, rf_profile=RFProfile()
    )
    assert config.split_rounds == rounds
    assert split_round_count(requested) == rounds
    assert config.forest_fits == rounds * 4


@pytest.mark.parametrize("requested", [1, 3, 5, 7, 101])
def test_corrected_rules_reject_odd_iterations(requested: int) -> None:
    with pytest.raises(InvalidConfigurationError, match="even n_iterations >= 4"):
        ComparisonConfig(n_iterations=requested, rf_profile=RFProfile())


def test_graph_roles_must_be_distinct() -> None:
    with pytest.raises(InvalidConfigurationError, match="must be distinct"):
        ChoirConfig(nn_graph_key="same", snn_graph_key="same")


def test_rf_calibration_grid_is_closed() -> None:
    assert RFProfile(n_bins=512, max_depth=None, n_streams=4).n_bins == 512
    with pytest.raises(InvalidConfigurationError, match="n_bins"):
        RFProfile(n_bins=1024)
    with pytest.raises(InvalidConfigurationError, match="max_batch_size"):
        RFProfile(max_batch_size=1024)


def test_rules_are_normalized_to_enum() -> None:
    assert (
        ComparisonConfig(rules="corrected", rf_profile=RFProfile()).rules
        is RuleSet.CORRECTED
    )


def test_repeat_error_limit_requires_integer() -> None:
    with pytest.raises(InvalidConfigurationError, match="integer >= 0"):
        ComparisonConfig(  # type: ignore[arg-type]
            max_repeat_errors=1.5,
            rf_profile=RFProfile(),
        )


def test_missing_calibration_never_falls_back_to_seed_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RAPIDS_CHOIR_RF_PROFILE", raising=False)
    with pytest.raises(MissingCalibrationError, match="no frozen RF profile"):
        ComparisonConfig()
    development = ComparisonConfig(rf_profile=RFProfile())
    assert development.rf_profile_source == "explicit_development"
    assert development.rf_profile_evidence_eligible is False


def test_frozen_profile_manifest_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rf_profile.json"
    path.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "winner": {"profile": RFProfile().to_record()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("RAPIDS_CHOIR_RF_PROFILE", str(path))
    frozen = ComparisonConfig()
    assert frozen.rf_profile == RFProfile()
    assert frozen.rf_profile_source == "frozen"
    assert frozen.rf_profile_evidence_eligible is True
