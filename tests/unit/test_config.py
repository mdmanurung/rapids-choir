from __future__ import annotations

import pytest

from rapids_choir import ChoirConfig, ComparisonConfig, RFProfile, RuleSet
from rapids_choir.config import split_round_count
from rapids_choir.exceptions import InvalidConfigurationError


@pytest.mark.parametrize(
    ("requested", "rounds"),
    [(1, 0), (3, 2), (5, 2), (7, 4), (9, 4), (101, 50)],
)
def test_compatibility_rounding_matches_r_ties_to_even(
    requested: int, rounds: int
) -> None:
    config = ComparisonConfig(rules="choir_v0_3_0", n_iterations=requested)
    assert config.split_rounds == rounds
    assert split_round_count(requested) == rounds
    assert config.forest_fits == rounds * 4


@pytest.mark.parametrize("requested", [1, 3, 5, 7, 101])
def test_corrected_rules_reject_odd_iterations(requested: int) -> None:
    with pytest.raises(InvalidConfigurationError, match="even n_iterations >= 4"):
        ComparisonConfig(n_iterations=requested)


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
    assert ComparisonConfig(rules="corrected").rules is RuleSet.CORRECTED


def test_repeat_error_limit_requires_integer() -> None:
    with pytest.raises(InvalidConfigurationError, match="integer >= 0"):
        ComparisonConfig(max_repeat_errors=1.5)  # type: ignore[arg-type]
