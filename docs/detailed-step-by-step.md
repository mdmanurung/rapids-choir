# Detailed step by step

This guide prepares one RNA dataset, runs CHOIR, and reads the results.

## 1. Check the input

Use one unimodal RNA `AnnData` object. Observation names and variable names
must be unique. Raw counts must be finite and nonnegative.

## 2. Prepare RNA fields

```python
import rapids_choir as choir

choir.pp.prepare_rna(
    adata,
    counts_layer="counts",
    batch_key="donor",
)
```

The default call writes these fields:

| Location | Key | Use |
|---|---|---|
| `layers` | `log1p` | Normalized RNA matrix |
| `var` | `highly_variable` | Variable-feature mask |
| `obsm` | `X_pca` | Cell representation |
| `obsp` | `choir_nn` | Nearest-neighbor graph |
| `obsp` | `choir_snn` | Shared-neighbor graph |

Existing fields cause an error. Set `overwrite=True` only when replacement is
intentional.

## 3. Run CHOIR

```python
result = choir.CHOIR(
    adata,
    key="CHOIR",
    alpha=0.05,
    p_adjust="bonferroni",
    feature_set="var",
    n_iterations=100,
    n_trees=50,
    batch_labels="donor",
    rules="corrected",
    device="cuda:0",
    rf_profile=choir.RFProfile(),
)
```

Shared scientific parameters use R CHOIR names and defaults when the Python
implementation has a direct equivalent. The main mappings are:

| R CHOIR | `rapids-choir` |
|---|---|
| `object` | An `AnnData` object |
| `key` | Keys in `adata.obs` and `adata.uns` |
| `p_adjust` | `"bonferroni"`, `"fdr"`, or `"none"` |
| `batch_labels` | A column in `adata.obs` |
| `random_seed` | The root seed for the run |
| `feature_set="var"` | The mask named by `feature_mask_key` |
| `feature_set="all"` | All variables |

AnnData requires explicit field selectors such as `layer`, `reduction`, and
the graph keys. GPU execution adds `device`, `vram_headroom`, and `rf_profile`.
The package fails if a real CUDA device is unavailable. It does not use a CPU
fallback.

Use `rules="corrected"` for new analyses. Use `rules="choir_v0_3_0"` only for
frozen-reference compatibility work. Neither rule set claims numerical
identity between ranger and cuML.

## 4. Inspect results

```python
labels = result.labels
records = choir.getRecords(result)
run_id = result.provenance["run_id"]
```

Pass `records_path="records.parquet"` to `CHOIR` to write immutable comparison
records. The package binds that file to the run and input checksums.

## 5. Compare two clusters

Select cells directly:

```python
comparison = choir.compareClusters(
    adata,
    cluster1_cells=adata.obs_names[adata.obs["cluster"] == "A"],
    cluster2_cells=adata.obs_names[adata.obs["cluster"] == "B"],
    rf_profile=choir.RFProfile(),
)
```

Or use the R-style identity arguments:

```python
comparison = choir.compareClusters(
    adata,
    ident1="A",
    ident2="B",
    group_by="cluster",
    rf_profile=choir.RFProfile(),
)
```
