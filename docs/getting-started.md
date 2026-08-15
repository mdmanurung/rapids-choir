# Getting started

This guide runs the package from a source checkout. It creates development
output only.

:::{warning}
No calibrated release exists. An explicit random-forest profile makes a run
ineligible for parity, release, and publication evidence.
:::

## Requirements

Use Python 3.12, CUDA 12.8, RAPIDS 26.06, rapids-singlecell 0.16.1, and one
NVIDIA L40S. The analysis has no CPU fallback. The environment locks in
`workflow/envs/` define the runtime.

## Install the source tree

Install the package after you create the locked RAPIDS environment:

```bash
python -m pip install -e .
```

Run the strict runtime check on the allocated GPU:

```bash
rapids-choir doctor --device cuda:0 --strict
```

Stop if the command returns a nonzero status.

## Prepare RNA

Load an `AnnData` object with raw counts in `adata.layers["counts"]` or
`adata.X`. Observation and variable names must be unique.

```python
import rapids_choir as rc

rc.pp.prepare_rna(adata, batch_key="donor")
```

The default call mutates `adata` and returns `None`. Pass `copy=True` to return
a prepared copy.

## Run a development analysis

Pass an explicit profile until a calibrated profile ships with the package:

```python
comparison = rc.ComparisonConfig(rf_profile=rc.RFProfile())
result = rc.tl.choir(
    adata,
    batch_key="donor",
    rules="corrected",
    comparison=comparison,
)
```

`choir` returns a `ChoirResult`. With the default `inplace=True`, it also writes
labels and provenance to `adata`.

## Read comparison records

Convert the in-memory records to a data frame:

```python
records = rc.get.records(result)
```

Set `records_path` on `choir` to write checksummed Parquet records. The package
rejects stale records when the recorded `AnnData` inputs change.
