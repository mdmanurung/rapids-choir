# Architecture

## Boundary

`rapids-choir` v0.1 is a single-GPU, unimodal RNA implementation for AnnData.
The `choir` analysis has no CPU backend or fallback; deterministic preparation
and topology bookkeeping are host operations. Runtime imports are
CUDA-independent until a GPU function is called.

```text
AnnData ──> optional prepare_rna ──> prepared-input validation
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 v                                               v
       SNN Louvain resolution state                    NN ChoirComparator
                 │                                  sampling / cuML forests
                 v                                               │
       strict immutable HierarchyTree <──────────────────────────┘
                 │                         subtree and bridge decisions
                 v
       bottom-up prune / merge / singleton / final relabel
                 │
                 v
       ChoirResult + provenance + checksummed Parquet records
```

The representation-independent `ChoirComparator` accepts CuPy dense or CSR
features, CuPy device index arrays, and a GPU NN graph. AnnData resolution and
mutation live outside the comparator.

## Residency and ownership

- Host: configuration, deterministic preprocessing outputs, compact membership
  identities, hierarchy state transitions, tree topology, centroid/silhouette
  control values, typed records, manifests, checksums, and provenance.
- Device: ingested feature and NN/SNN matrices, feature masks, comparator
  memberships, sampled indices, forest predictions, and comparator hot-path
  statistics. cuGraph performs Louvain on induced SNN graphs and returns the
  compact partition vector to the host topology state machine.
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

## Compatibility and corrected behavior

Compatibility mode (`rules="choir_v0_3_0"`) follows the frozen R operation
order and records captured quirks, including sampled-position repeated errors,
the duplicated-variance FDR expression, and the full-reduction optimization
selector. Corrected mode is the default and uses stable cell identities,
corrected FDR candidates, and node-local optimization. Every trace and result
records the ruleset; corrected differences cannot satisfy an R-parity gate.

Stable hierarchy node identities are the integer interpretation of membership
SHA-256 values. They remain integers in memory but are serialized as decimal
strings, because JSON numbers cannot carry 256-bit identities losslessly across
R and Python.

## Validation-only semantic trace

Instrumentation emits versioned events for provenance, preprocessing,
resolution search and transitions, hierarchy nodes and stops, comparator
calls, multiplicity, pruning, bridges, merges, singletons, and final relabeling.
Trace artifacts are append-only canonical JSONL or typed Parquet, referenced by
schema, SHA-256, row count, run ID, and event schema. Trace generation is a
private validation facility and is excluded from public API and benchmark
timing.

## Evidence boundary

CPU tests establish import safety, schemas, validation, hashing, formatting,
mutation sensitivity, and abstract tree invariants only. Functional claims
require actual CuPy,
cuML, cuGraph, RMM, rapids-singlecell, and an L40S Slurm allocation. The tested
commit, lock checksum, job ID, GPU, driver, and profiling artifacts are part of
the evidence manifest.
