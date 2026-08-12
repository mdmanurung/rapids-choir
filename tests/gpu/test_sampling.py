from __future__ import annotations

import pytest

from rapids_choir._sampling import balanced_split_round, eligible_batch_codes

cp = pytest.importorskip("cupy")


@pytest.mark.gpu
@pytest.mark.parametrize("rules", ["corrected", "choir_v0_3_0"])
def test_balanced_split_round_is_device_resident_and_reproducible(rules: str) -> None:
    cluster1 = cp.arange(0, 20, dtype=cp.int64)
    cluster2 = cp.arange(20, 40, dtype=cp.int64)
    first = balanced_split_round(cluster1, cluster2, n_per_half=5, seed=17, rules=rules)
    second = balanced_split_round(
        cluster1, cluster2, n_per_half=5, seed=17, rules=rules
    )
    assert isinstance(first.row_indices, cp.ndarray)
    assert first.true_labels.dtype == cp.int32
    assert first.permuted_labels.dtype == cp.int32
    assert first.first_train_mask.dtype == cp.bool_
    cp.testing.assert_array_equal(first.row_indices, second.row_indices)
    cp.testing.assert_array_equal(first.first_train_mask, second.first_train_mask)
    cp.testing.assert_array_equal(first.permuted_labels, second.permuted_labels)
    assert int(cp.sum(first.first_train_mask[:10]).item()) == 5
    assert int(cp.sum(first.first_train_mask[10:]).item()) == 5
    assert not bool(cp.any(first.first_train_mask & first.second_train_mask).item())


@pytest.mark.gpu
def test_rule_modes_have_distinct_sampling_call_order() -> None:
    cluster1 = cp.arange(0, 20, dtype=cp.int64)
    cluster2 = cp.arange(20, 40, dtype=cp.int64)
    corrected = balanced_split_round(
        cluster1, cluster2, n_per_half=5, seed=17, rules="corrected"
    )
    compatibility = balanced_split_round(
        cluster1, cluster2, n_per_half=5, seed=17, rules="choir_v0_3_0"
    )
    assert not bool(cp.all(corrected.row_indices == compatibility.row_indices).item())


@pytest.mark.gpu
def test_batch_eligibility_aligns_explicit_identities() -> None:
    batches = cp.asarray(
        [0] * 8 + [1] * 2 + [0] * 7 + [1] * 3,
        dtype=cp.int32,
    )
    cluster1 = cp.arange(0, 10, dtype=cp.int64)
    cluster2 = cp.arange(10, 20, dtype=cp.int64)
    eligible = eligible_batch_codes(batches, cluster1, cluster2)
    cp.testing.assert_array_equal(eligible, cp.asarray([0, 1], dtype=cp.int32))


@pytest.mark.gpu
def test_batch_eligibility_can_be_empty() -> None:
    batches = cp.asarray([0] * 8 + [1] * 8, dtype=cp.int32)
    cluster1 = cp.arange(0, 8, dtype=cp.int64)
    cluster2 = cp.arange(8, 16, dtype=cp.int64)
    assert eligible_batch_codes(batches, cluster1, cluster2).size == 0
