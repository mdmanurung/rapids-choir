"""Stable hierarchical seed derivation."""

from __future__ import annotations

import hashlib
import importlib
import struct
from collections.abc import Iterable
from typing import Any

from .exceptions import GPUUnavailableError, InputResidencyError


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


def device_membership_fingerprint(indices: Any) -> tuple[int, int, int]:
    """Return a compact order-independent fingerprint derived on the GPU.

    The fingerprint is a lookup/seed key, not an equality proof. Cache callers
    must verify complete device memberships before reusing an entry.
    """

    try:
        cp = importlib.import_module("cupy")
    except Exception as error:
        raise GPUUnavailableError(f"could not import CuPy: {error}") from error
    if not isinstance(indices, cp.ndarray):
        raise InputResidencyError("membership indices must be a CuPy device array")
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("membership indices must be a non-empty vector")
    if indices.dtype.kind not in {"i", "u"}:
        raise TypeError("membership indices must have integer dtype")

    values = indices.astype(cp.uint64, copy=False)
    # SplitMix64 provides good avalanche behavior using operations supported by
    # CuPy. Sum and xor make the compact key independent of caller ordering.
    mixed = values + cp.uint64(0x9E3779B97F4A7C15)
    mixed = (mixed ^ (mixed >> cp.uint64(30))) * cp.uint64(0xBF58476D1CE4E5B9)
    mixed = (mixed ^ (mixed >> cp.uint64(27))) * cp.uint64(0x94D049BB133111EB)
    mixed = mixed ^ (mixed >> cp.uint64(31))
    xor_value = int(cp.bitwise_xor.reduce(mixed).item())
    sum_value = int(cp.sum(mixed, dtype=cp.uint64).item())
    return int(indices.size), xor_value, sum_value


def device_memberships_equal(left: Any, right: Any) -> bool:
    """Compare complete memberships on-device, independent of ordering."""

    try:
        cp = importlib.import_module("cupy")
    except Exception as error:
        raise GPUUnavailableError(f"could not import CuPy: {error}") from error
    if not isinstance(left, cp.ndarray) or not isinstance(right, cp.ndarray):
        raise InputResidencyError("memberships must be CuPy device arrays")
    if left.ndim != 1 or right.ndim != 1 or left.size != right.size:
        return False
    return bool(cp.array_equal(cp.sort(left), cp.sort(right)).item())
