# User guide

The public workflow has two stages. Prepare the RNA fields, then run the CHOIR
analysis on one GPU.

## AnnData contract

`prepare_rna` reads nonnegative counts from the named counts layer. It uses
`adata.X` only when that layer is absent. The function writes these fields by
default:

| Location | Key | Purpose |
|---|---|---|
| `layers` | `log1p` | Normalized RNA matrix |
| `var` | `highly_variable` | Feature mask |
| `obsm` | `X_pca` | PCA representation |
| `obsp` | `choir_nn` | Nearest-neighbor connectivity graph |
| `obsp` | `choir_snn` | Shared-neighbor clustering graph |
| `uns` | `rapids_choir_prepare_rna` | Content-bound preparation manifest |

Existing outputs cause `ArtifactCollisionError` unless `overwrite=True`.
`validate_prepared_rna` checks shapes and any stored content hashes.

## Rule sets

Use `rules="corrected"` for new analyses. Corrected rules use stable cell
identities and corrected multiplicity behavior.

Use `rules="choir_v0_3_0"` only for compatibility evaluation. Compatibility
rules preserve captured R quirks and operation order. Corrected results cannot
satisfy an R-parity gate.

## GPU behavior

GPU entry points accept devices in the form `cuda:<index>`. They import CUDA
libraries only when a GPU function runs. Importing `rapids_choir` does not
initialize CUDA.

The fixed workload preflight checks device memory. It does not reduce cells,
features, trees, iterations, bins, or streams after a failed estimate.

## Results and persistence

`choir` always returns `ChoirResult`. The result contains final labels,
comparison records, provenance, and an immutable `HierarchyTree`.

Pass `records_path` to stream comparison records to Parquet. The manifest binds
the file checksum, row count, run ID, and input checksum. Use
`rapids_choir.get.records` to read in-memory or persisted records.

## Failures

Configuration, input, graph, collision, stale-artifact, and device-memory
failures use package-specific exceptions. The package does not replace a GPU
failure with CPU execution.

See the [API reference](api.rst) for signatures and the
[architecture](architecture/architecture.md) for internal ownership rules.
