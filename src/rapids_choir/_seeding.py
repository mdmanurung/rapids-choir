"""Stable hierarchical seed derivation."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable


def canonical_membership_bytes(indices: Iterable[int]) -> bytes:
    """Encode sorted unique non-negative stable cell indices."""

    canonical = sorted(set(indices))
    if any(index < 0 or index >= 2**64 for index in canonical):
        raise ValueError("membership indices must be in [0, 2**64)")
    return b"".join(struct.pack("<Q", index) for index in canonical)


def derive_seed(root_seed: int, namespace: str, *identities: object) -> int:
    """Derive a stable unsigned 32-bit child seed with BLAKE2b."""

    digest = hashlib.blake2b(digest_size=8, person=b"rch-seed")
    digest.update(str(root_seed).encode("ascii"))
    digest.update(b"\0")
    digest.update(namespace.encode("utf-8"))
    for identity in identities:
        digest.update(b"\0")
        if isinstance(identity, bytes):
            digest.update(identity)
        else:
            digest.update(str(identity).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)
