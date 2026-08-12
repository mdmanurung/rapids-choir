from __future__ import annotations

from rapids_choir._seeding import canonical_membership_bytes, derive_seed


def test_membership_encoding_is_order_and_duplicate_independent() -> None:
    assert canonical_membership_bytes([3, 1, 3, 2]) == canonical_membership_bytes(
        [1, 2, 3]
    )


def test_child_seed_is_stable_and_namespaced() -> None:
    membership = canonical_membership_bytes([1, 5, 8])
    seed = derive_seed(1, "corrected/comparison", membership, 2, "observed-a")
    assert seed == derive_seed(1, "corrected/comparison", membership, 2, "observed-a")
    assert seed != derive_seed(
        1, "choir_v0_3_0/comparison", membership, 2, "observed-a"
    )
