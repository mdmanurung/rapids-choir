"""Deterministic unimodal-RNA preparation for the locked AnnData scope."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from .._hashing import combine_hashes, hash_matrix, hash_vector
from ..exceptions import (
    ArtifactCollisionError,
    PreparedInputError,
    StaleArtifactError,
)

PREPARED_SCHEMA = "rapids_choir.prepare_rna/v1"
PREPARED_UNS_KEY = "rapids_choir_prepare_rna"


def _counts_matrix(value: Any) -> sparse.csr_matrix:
    matrix = sparse.csr_matrix(value, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise PreparedInputError(
            "RNA counts must be a non-empty two-dimensional matrix"
        )
    if not np.issubdtype(matrix.dtype, np.number):
        raise PreparedInputError("RNA counts must be numeric")
    if matrix.data.size and (
        not np.all(np.isfinite(matrix.data)) or np.any(matrix.data < 0)
    ):
        raise PreparedInputError("RNA counts must contain finite non-negative values")
    totals = np.asarray(matrix.sum(axis=1), dtype=np.float64).ravel()
    if np.any(totals <= 0):
        raise PreparedInputError("every cell must have a positive RNA count total")
    return matrix


def _normalize_log1p(
    counts: sparse.csr_matrix, target_sum: float
) -> sparse.csr_matrix:
    totals = np.asarray(counts.sum(axis=1), dtype=np.float64).ravel()
    normalized = sparse.diags(target_sum / totals, format="csr") @ counts
    normalized = sparse.csr_matrix(normalized, dtype=np.float64)
    normalized.data = np.log1p(normalized.data)
    normalized.eliminate_zeros()
    normalized.sort_indices()
    return normalized.astype(np.float32)


def _feature_scores(matrix: sparse.csr_matrix, flavor: str) -> np.ndarray[Any, Any]:
    n_cells = matrix.shape[0]
    means = np.asarray(matrix.mean(axis=0), dtype=np.float64).ravel()
    squares = np.asarray(matrix.power(2).mean(axis=0), dtype=np.float64).ravel()
    variances = np.maximum(squares - means**2, 0.0)
    if n_cells > 1:
        variances *= n_cells / (n_cells - 1)
    if flavor == "cell_ranger":
        scores = variances / np.maximum(means, np.finfo(np.float64).eps)
    elif flavor == "seurat_v3":
        scores = variances / np.maximum(means + means**2, np.finfo(np.float64).eps)
    elif flavor == "pearson_residuals":
        scores = variances / np.maximum(means, np.finfo(np.float64).eps)
    else:  # guarded by the public validator
        raise PreparedInputError(f"unsupported hvg_flavor: {flavor}")
    scores[~np.isfinite(scores)] = -np.inf
    return cast(np.ndarray[Any, Any], scores)


def _rank_features(scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    indices = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((indices, -scores))


def _select_hvgs(
    matrix: sparse.csr_matrix,
    *,
    flavor: str,
    n_top_genes: int,
    batch_codes: np.ndarray[Any, Any] | None,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    n_vars = matrix.shape[1]
    selected_count = min(n_top_genes, n_vars)
    mask = np.zeros(n_vars, dtype=bool)
    if batch_codes is None:
        mask[_rank_features(_feature_scores(matrix, flavor))[:selected_count]] = True
        return mask

    unique_batches = np.unique(batch_codes)
    ranks = np.full((unique_batches.size, n_vars), n_vars, dtype=np.int64)
    chosen = np.zeros((unique_batches.size, n_vars), dtype=bool)
    for batch_index, batch in enumerate(unique_batches):
        order = _rank_features(_feature_scores(matrix[batch_codes == batch], flavor))
        ranks[batch_index, order] = np.arange(n_vars, dtype=np.int64)
        chosen[batch_index, order[:selected_count]] = True
    batch_frequency = chosen.sum(axis=0)
    median_rank = np.median(ranks, axis=0)
    indices = np.arange(n_vars, dtype=np.int64)
    aggregate_order = np.lexsort((indices, median_rank, -batch_frequency))
    mask[aggregate_order[:selected_count]] = True
    return mask


def _pca(
    matrix: sparse.csr_matrix,
    feature_mask: np.ndarray[Any, np.dtype[np.bool_]],
    n_pcs: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    values = matrix[:, feature_mask].toarray().astype(np.float64, copy=False)
    values -= values.mean(axis=0, keepdims=True)
    scales = values.std(axis=0, ddof=1 if values.shape[0] > 1 else 0)
    scales[~np.isfinite(scales) | (scales == 0)] = 1.0
    values /= scales
    n_components = min(n_pcs, values.shape[0] - 1, values.shape[1])
    if n_components < 1:
        raise PreparedInputError("PCA requires at least two cells and one feature")
    left, singular_values, right = np.linalg.svd(values, full_matrices=False)
    for component in range(right.shape[0]):
        anchor = int(np.argmax(np.abs(right[component])))
        if right[component, anchor] < 0:
            right[component] *= -1
            left[:, component] *= -1
    coordinates = left[:, :n_components] * singular_values[:n_components]
    return np.asarray(coordinates, dtype=np.float32)


def _neighbor_graphs(
    representation: np.ndarray[Any, Any], n_neighbors: int
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    n_cells = representation.shape[0]
    if n_cells < 2:
        raise PreparedInputError("neighbor construction requires at least two cells")
    k = min(n_neighbors, n_cells - 1)
    distances, indices = cKDTree(np.asarray(representation, dtype=np.float64)).query(
        representation, k=k + 1, workers=1
    )
    distances = np.atleast_2d(distances)
    indices = np.atleast_2d(indices)
    neighbor_rows: list[int] = []
    neighbor_columns: list[int] = []
    for cell in range(n_cells):
        candidates = [
            (float(distance), int(index))
            for distance, index in zip(distances[cell], indices[cell], strict=True)
            if int(index) != cell
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        selected = candidates[:k]
        neighbor_rows.extend([cell] * len(selected))
        neighbor_columns.extend(index for _, index in selected)
    nn = sparse.csr_matrix(
        (
            np.ones(len(neighbor_rows), dtype=np.float32),
            (neighbor_rows, neighbor_columns),
        ),
        shape=(n_cells, n_cells),
    )
    nn.sum_duplicates()
    nn.sort_indices()

    closed = (nn + sparse.eye(n_cells, dtype=np.float32, format="csr")).astype(bool)
    intersections = (closed.astype(np.float32) @ closed.T.astype(np.float32)).tocsr()
    intersections.setdiag(0)
    intersections.eliminate_zeros()
    coo = intersections.tocoo()
    denominator = 2.0 * (k + 1) - coo.data
    weights = np.divide(
        coo.data,
        denominator,
        out=np.zeros_like(coo.data, dtype=np.float32),
        where=denominator > 0,
    )
    keep = weights > (1.0 / 15.0)
    snn = sparse.csr_matrix(
        (weights[keep], (coo.row[keep], coo.col[keep])),
        shape=(n_cells, n_cells),
        dtype=np.float32,
    )
    snn = snn.maximum(snn.T).tocsr()
    snn.sort_indices()
    return nn, snn


def _batch_codes(adata: Any, batch_key: str | None) -> np.ndarray[Any, Any] | None:
    if batch_key is None:
        return None
    if batch_key not in adata.obs:
        raise PreparedInputError(f"AnnData obs field {batch_key!r} does not exist")
    values = adata.obs[batch_key]
    if bool(values.isna().any()):
        raise PreparedInputError(f"batch field {batch_key!r} contains missing values")
    import pandas as pd  # type: ignore[import-untyped]

    codes, _ = pd.factorize(values, sort=True)
    return np.asarray(codes, dtype=np.int64)


def _collisions(adata: Any, targets: dict[str, str]) -> list[str]:
    present = []
    if targets["normalized_layer"] in adata.layers:
        present.append(f"layers[{targets['normalized_layer']!r}]")
    if targets["use_rep"] in adata.obsm:
        present.append(f"obsm[{targets['use_rep']!r}]")
    for key in ("nn_graph_key", "snn_graph_key"):
        if targets[key] in adata.obsp:
            present.append(f"obsp[{targets[key]!r}]")
    if targets["feature_mask_key"] in adata.var:
        present.append(f"var[{targets['feature_mask_key']!r}]")
    if PREPARED_UNS_KEY in adata.uns:
        present.append(f"uns[{PREPARED_UNS_KEY!r}]")
    return present


def prepare_rna(
    adata: Any,
    *,
    counts_layer: str = "counts",
    normalized_layer: str = "log1p",
    target_sum: float = 10_000.0,
    hvg_flavor: str = "seurat_v3",
    n_top_genes: int = 2_000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
    batch_key: str | None = None,
    use_rep: str = "X_pca",
    nn_graph_key: str = "choir_nn",
    snn_graph_key: str = "choir_snn",
    feature_mask_key: str = "highly_variable",
    random_state: int = 1,
    copy: bool = False,
    overwrite: bool = False,
) -> Any | None:
    """Prepare unimodal RNA with Scanpy-style ``copy`` behavior.

    ``copy=False`` mutates ``adata`` and returns ``None``. ``copy=True`` leaves
    the input untouched and returns a prepared copy.
    """

    if adata is None or not hasattr(adata, "layers"):
        raise TypeError("adata must be an AnnData-like object")
    if not bool(getattr(adata.obs_names, "is_unique", False)):
        raise PreparedInputError("AnnData obs_names must be unique")
    if not bool(getattr(adata.var_names, "is_unique", False)):
        raise PreparedInputError("AnnData var_names must be unique")
    string_values = {
        "counts_layer": counts_layer,
        "normalized_layer": normalized_layer,
        "use_rep": use_rep,
        "nn_graph_key": nn_graph_key,
        "snn_graph_key": snn_graph_key,
        "feature_mask_key": feature_mask_key,
    }
    if any(not value for value in string_values.values()):
        raise PreparedInputError("preprocessing keys must be non-empty")
    if normalized_layer == counts_layer:
        raise PreparedInputError("counts and normalized layers must be distinct")
    if nn_graph_key == snn_graph_key:
        raise PreparedInputError("NN and SNN graph keys must be distinct")
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise PreparedInputError("target_sum must be finite and positive")
    if hvg_flavor not in {"cell_ranger", "seurat_v3", "pearson_residuals"}:
        raise PreparedInputError("unsupported hvg_flavor")
    for name, value in {
        "n_top_genes": n_top_genes,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise PreparedInputError(f"{name} must be a positive integer")
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise PreparedInputError("random_state must be an integer")

    target = adata.copy() if copy else adata
    targets = {
        "normalized_layer": normalized_layer,
        "use_rep": use_rep,
        "nn_graph_key": nn_graph_key,
        "snn_graph_key": snn_graph_key,
        "feature_mask_key": feature_mask_key,
    }
    collisions = _collisions(target, targets)
    if collisions and not overwrite:
        raise ArtifactCollisionError(
            "RNA preparation would overwrite existing artifacts: "
            + ", ".join(collisions)
        )

    if counts_layer in target.layers:
        counts_source = target.layers[counts_layer]
        create_counts_layer = False
    else:
        if target.X is None:
            raise PreparedInputError(
                f"counts layer {counts_layer!r} is absent and AnnData X is empty"
            )
        counts_source = target.X
        create_counts_layer = True
    counts = _counts_matrix(counts_source)
    batches = _batch_codes(target, batch_key)
    normalized = _normalize_log1p(counts, target_sum)
    feature_mask = _select_hvgs(
        normalized,
        flavor=hvg_flavor,
        n_top_genes=n_top_genes,
        batch_codes=batches,
    )
    representation = _pca(normalized, feature_mask, n_pcs)
    nn_graph, snn_graph = _neighbor_graphs(representation, n_neighbors)

    parameters = {
        **string_values,
        "target_sum": float(target_sum),
        "hvg_flavor": hvg_flavor,
        "n_top_genes": n_top_genes,
        "n_pcs": n_pcs,
        "n_neighbors": n_neighbors,
        "batch_key": batch_key,
        "random_state": random_state,
    }
    input_hashes = {
        "counts": hash_matrix(counts),
        "obs_names": hash_vector(np.asarray(target.obs_names, dtype=str)),
        "var_names": hash_vector(np.asarray(target.var_names, dtype=str)),
    }
    if batches is not None:
        input_hashes["batches"] = hash_vector(batches)
    output_hashes = {
        "normalized": hash_matrix(normalized),
        "feature_mask": hash_vector(feature_mask),
        "representation": hash_matrix(representation),
        "nn_graph": hash_matrix(nn_graph),
        "snn_graph": hash_matrix(snn_graph),
    }
    manifest = {
        "schema": PREPARED_SCHEMA,
        "parameters": parameters,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "input_sha256": combine_hashes(
            "rapids-choir/prepare-input/v1", input_hashes
        ),
        "output_sha256": combine_hashes(
            "rapids-choir/prepare-output/v1", output_hashes
        ),
    }

    if create_counts_layer:
        target.layers[counts_layer] = counts.copy()
    target.layers[normalized_layer] = normalized
    target.var[feature_mask_key] = feature_mask
    target.obsm[use_rep] = representation
    target.obsp[nn_graph_key] = nn_graph
    target.obsp[snn_graph_key] = snn_graph
    target.uns[PREPARED_UNS_KEY] = manifest
    return target if copy else None


def validate_prepared_rna(
    adata: Any,
    *,
    layer: str,
    use_rep: str,
    nn_graph_key: str,
    snn_graph_key: str,
    feature_mask_key: str,
    batch_key: str | None,
    require_manifest: bool = False,
) -> dict[str, Any] | None:
    """Validate shapes and, when present, every preparation content hash."""

    n_obs = int(adata.n_obs)
    n_vars = int(adata.n_vars)
    if not bool(adata.obs_names.is_unique) or not bool(adata.var_names.is_unique):
        raise PreparedInputError(
            "AnnData observation and variable names must be unique"
        )
    if layer != "X" and layer not in adata.layers:
        raise PreparedInputError(f"AnnData layer {layer!r} does not exist")
    matrix = adata.X if layer == "X" else adata.layers[layer]
    if matrix is None or matrix.shape != (n_obs, n_vars):
        raise PreparedInputError("prepared feature matrix is empty or misaligned")
    if use_rep not in adata.obsm or adata.obsm[use_rep].shape[0] != n_obs:
        raise PreparedInputError(f"AnnData representation {use_rep!r} is misaligned")
    for key, role in ((nn_graph_key, "NN"), (snn_graph_key, "SNN")):
        if key not in adata.obsp or adata.obsp[key].shape != (n_obs, n_obs):
            raise PreparedInputError(
                f"AnnData {role} graph {key!r} is missing or misaligned"
            )
    if nn_graph_key == snn_graph_key:
        raise PreparedInputError("NN and SNN graph keys must be distinct")
    if feature_mask_key not in adata.var:
        raise PreparedInputError(
            f"AnnData feature mask {feature_mask_key!r} does not exist"
        )
    feature_mask = adata.var[feature_mask_key]
    if len(feature_mask) != n_vars or bool(feature_mask.isna().any()):
        raise PreparedInputError("prepared feature mask is missing or misaligned")
    if not bool(feature_mask.astype(bool).any()):
        raise PreparedInputError("prepared feature mask selects no features")
    _batch_codes(adata, batch_key)

    manifest = adata.uns.get(PREPARED_UNS_KEY)
    if manifest is None:
        if require_manifest:
            raise PreparedInputError("prepared RNA manifest is required")
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != PREPARED_SCHEMA:
        raise StaleArtifactError("prepared RNA manifest has an unsupported schema")
    parameters = manifest.get("parameters", {})
    expected_keys = {
        "normalized_layer": layer,
        "use_rep": use_rep,
        "nn_graph_key": nn_graph_key,
        "snn_graph_key": snn_graph_key,
        "feature_mask_key": feature_mask_key,
        "batch_key": batch_key,
    }
    for name, expected in expected_keys.items():
        if parameters.get(name) != expected:
            raise StaleArtifactError(
                f"prepared RNA manifest key {name!r} does not match requested input"
            )
    counts_layer = parameters.get("counts_layer")
    if not isinstance(counts_layer, str) or counts_layer not in adata.layers:
        raise StaleArtifactError("prepared counts layer is absent")
    batches = _batch_codes(adata, batch_key)
    input_hashes = {
        "counts": hash_matrix(adata.layers[counts_layer]),
        "obs_names": hash_vector(np.asarray(adata.obs_names, dtype=str)),
        "var_names": hash_vector(np.asarray(adata.var_names, dtype=str)),
    }
    if batches is not None:
        input_hashes["batches"] = hash_vector(batches)
    output_hashes = {
        "normalized": hash_matrix(matrix),
        "feature_mask": hash_vector(
            feature_mask.to_numpy(dtype=bool, copy=False)
        ),
        "representation": hash_matrix(adata.obsm[use_rep]),
        "nn_graph": hash_matrix(adata.obsp[nn_graph_key]),
        "snn_graph": hash_matrix(adata.obsp[snn_graph_key]),
    }
    if input_hashes != manifest.get("input_hashes"):
        raise StaleArtifactError("prepared RNA input content has changed")
    if output_hashes != manifest.get("output_hashes"):
        raise StaleArtifactError("prepared RNA output content has changed")
    if combine_hashes("rapids-choir/prepare-input/v1", input_hashes) != manifest.get(
        "input_sha256"
    ):
        raise StaleArtifactError("prepared RNA input aggregate hash is invalid")
    if combine_hashes("rapids-choir/prepare-output/v1", output_hashes) != manifest.get(
        "output_sha256"
    ):
        raise StaleArtifactError("prepared RNA output aggregate hash is invalid")
    return manifest
