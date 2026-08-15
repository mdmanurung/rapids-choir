"""Canonical content identities used by provenance and stale-artifact checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON without platform- or insertion-order-dependent whitespace."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _little_endian_bytes(values: np.ndarray[Any, Any]) -> bytes:
    array = np.ascontiguousarray(values)
    dtype = array.dtype
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        array = array.byteswap().view(dtype.newbyteorder("<"))
    return array.tobytes(order="C")


def hash_matrix(value: Any) -> str:
    """Hash matrix values canonically across dense and SciPy sparse storage.

    Explicit sparse zeros, duplicate entries, sparse format, and dense versus
    sparse representation do not affect the identity. Shape, dtype, and every
    numerical value do.
    """

    if sparse.issparse(value):
        matrix = sparse.csr_matrix(value, copy=True)
    else:
        array = np.asarray(value)
        if array.ndim != 2:
            raise ValueError("matrix identity requires a two-dimensional input")
        matrix = sparse.csr_matrix(array)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(b"rapids-choir/matrix/v1\0")
    digest.update(
        canonical_json_bytes({"shape": matrix.shape, "dtype": matrix.dtype.str})
    )
    digest.update(b"\0indptr\0")
    digest.update(_little_endian_bytes(matrix.indptr.astype(np.int64, copy=False)))
    digest.update(b"\0indices\0")
    digest.update(_little_endian_bytes(matrix.indices.astype(np.int64, copy=False)))
    digest.update(b"\0data\0")
    digest.update(_little_endian_bytes(matrix.data))
    return digest.hexdigest()


def hash_vector(value: Any) -> str:
    """Hash an aligned vector, including dtype and missing-value placement."""

    values = np.asarray(value)
    if values.ndim != 1:
        raise ValueError("vector identity requires a one-dimensional input")
    digest = hashlib.sha256()
    digest.update(b"rapids-choir/vector/v1\0")
    digest.update(
        canonical_json_bytes({"shape": values.shape, "dtype": values.dtype.str})
    )
    digest.update(b"\0")
    if values.dtype.kind in {"O", "U", "S"}:
        encoded = [None if item is None else str(item) for item in values.tolist()]
        digest.update(canonical_json_bytes(encoded))
    else:
        digest.update(_little_endian_bytes(values))
    return digest.hexdigest()


def membership_sha256(indices: Sequence[int] | np.ndarray[Any, Any]) -> str:
    canonical = np.unique(np.asarray(indices, dtype=np.int64))
    if canonical.ndim != 1 or np.any(canonical < 0):
        raise ValueError("memberships must be a one-dimensional non-negative vector")
    digest = hashlib.sha256()
    digest.update(b"rapids-choir/membership/v1\0")
    digest.update(",".join(str(int(value)) for value in canonical).encode("ascii"))
    return digest.hexdigest()


def combine_hashes(namespace: str, components: Mapping[str, str]) -> str:
    payload = {"namespace": namespace, "components": dict(sorted(components.items()))}
    return sha256_bytes(canonical_json_bytes(payload))


def anndata_input_hashes(
    adata: Any,
    *,
    layer: str,
    use_rep: str,
    nn_graph_key: str,
    snn_graph_key: str,
    feature_mask_key: str | None,
    batch_key: str | None,
) -> tuple[str, dict[str, str]]:
    """Return a complete identity for every content-bearing hierarchy input."""

    matrix = adata.X if layer == "X" else adata.layers[layer]
    components = {
        "features": hash_matrix(matrix),
        "representation": hash_matrix(adata.obsm[use_rep]),
        "nn_graph": hash_matrix(adata.obsp[nn_graph_key]),
        "snn_graph": hash_matrix(adata.obsp[snn_graph_key]),
        "feature_mask": hash_vector(
            np.ones(adata.n_vars, dtype=bool)
            if feature_mask_key is None
            else adata.var[feature_mask_key].to_numpy(dtype=bool, copy=False)
        ),
        "obs_names": hash_vector(np.asarray(adata.obs_names, dtype=str)),
        "var_names": hash_vector(np.asarray(adata.var_names, dtype=str)),
    }
    if batch_key is not None:
        components["batches"] = hash_vector(
            adata.obs[batch_key].astype("string").to_numpy(dtype=str)
        )
    return combine_hashes("rapids-choir/anndata-input/v1", components), components
