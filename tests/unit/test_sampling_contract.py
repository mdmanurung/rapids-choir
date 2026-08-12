from __future__ import annotations

import pytest

from rapids_choir._sampling import balanced_sample_size


@pytest.mark.parametrize(
    ("size1", "size2", "rate", "cap", "expected"),
    [
        (100, 80, 1.0, None, 40),
        (100, 80, 0.5, None, 20),
        (3, 100, 0.1, None, 1),
        (100, 80, 1.0, 12, 12),
    ],
)
def test_balanced_sample_size(
    size1: int, size2: int, rate: float, cap: int | None, expected: int
) -> None:
    assert (
        balanced_sample_size(
            size1,
            size2,
            downsampling_rate=rate,
            sample_max=cap,
        )
        == expected
    )


def test_balanced_sample_size_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="cluster sizes"):
        balanced_sample_size(0, 4)
    with pytest.raises(ValueError, match="downsampling_rate"):
        balanced_sample_size(4, 4, downsampling_rate=0)
