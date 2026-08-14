# Architecture

## Boundary

`rapids-choir` v0.1 is a single-GPU, unimodal RNA implementation for AnnData.
It has no CPU backend or fallback. Runtime imports are CUDA-independent until a
GPU function is called.

```text
AnnData adapter ──> validated device inputs ──> ChoirComparator
     │                       │                        │
     │                       ├─ NN connectivity      ├─ sampling/permutation
     │                       ├─ feature mask         ├─ four cuML RFs/round
     │                       └─ transfer ledger      └─ GPU float64 summaries
     │
     └─ compact provenance / categorical labels / record manifest
```

The representation-independent `ChoirComparator` accepts CuPy dense or CSR
features, CuPy device index arrays, and a GPU NN graph. AnnData resolution and
mutation live outside the comparator.

## Residency and ownership

- Host: configuration, compact typed records, tree topology, manifests,
  checksums, and provenance.
- Device: features, representations, NN/SNN graphs, feature masks, memberships,
  the current cell-to-node vector, sampled indices, predictions, and hot-path
  statistics.
- A tree node stores stable integer identity plus parent/child links and a
  membership predicate; it does not duplicate a full cell-index vector.
- The input policy `transfer_once` permits one measured ingest. Any later host
  transfer over 1 MiB must be explicitly approved and recorded.

## GPU lifecycle

CUDA modules are imported inside GPU entry points. The first call validates the
requested device, initializes one RMM pool, and records allocator state. Every
cuML fit receives contiguous device `float32` features and device `int32`
labels with `convert_dtype=False`. Statistical reductions, KDE/CDF, bootstrap
variance, and threshold calculations use device `float64`.

Fixed mode preflights the full comparison. It never reduces cells, features,
trees, iterations, bins, or streams after OOM estimation. Failure raises
`InsufficientDeviceMemoryError` containing estimated bytes, free bytes,
configured headroom, and required bytes.

## Graphs

NN and SNN graphs are separate typed inputs. NN is used only for cross-cluster
connectivity. SNN is used only for Louvain hierarchy construction. A Scanpy
neighbors metadata adapter may resolve two keys only when their semantics and
shape are explicit; otherwise it raises `GraphSemanticError`.

## Determinism and caching

Comparisons execute in canonical membership-hash order. Corrected child seeds
are derived with BLAKE2b from root seed, operation kind, canonical membership,
round, and fit role. Compatibility mode uses a distinct seed namespace and
records unavoidable ranger/cuML algorithm differences.

Comparator result caching is disabled for v0.1. Pointer identity cannot prove
that mutable matrices, graphs, memberships, masks, or batches are unchanged.
Any future cache must use complete content-derived identities covering those
inputs plus seeds, mode, and the frozen RF profile.

## Records

Every comparison yields a `ComparisonResult`, including decision, stable reason
code, metrics, connectivity, cell and feature counts, seeds, timings, transfer
ledger, and peak memory. A complete hierarchy returns `ChoirResult`.

With `records_path`, records are streamed to Parquet and the AnnData `.uns`
entry stores only a manifest containing schema version, URI, SHA-256, row count,
and run identity. `rapids_choir.get.records` accepts either the in-memory result
or an AnnData carrying this manifest.

## Evidence boundary

CPU tests establish import safety, schemas, validation, hashing, formatting,
and abstract tree invariants only. Functional claims require actual CuPy,
cuML, cuGraph, RMM, rapids-singlecell, and an L40S Slurm allocation. The tested
commit, lock checksum, job ID, GPU, driver, and profiling artifacts are part of
the evidence manifest.
