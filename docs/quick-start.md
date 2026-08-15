# Quick start guide

Load an `AnnData` object with nonnegative RNA counts. Store the counts in
`adata.layers["counts"]` or `adata.X`.

```python
import rapids_choir as choir

choir.pp.prepare_rna(adata, batch_key="donor")

result = choir.CHOIR(
    adata,
    batch_labels="donor",
    rf_profile=choir.RFProfile(),
)
```

`CHOIR` follows the R CHOIR names for shared parameters. It writes final labels
to `adata.obs["CHOIR"]` and provenance to `adata.uns["CHOIR"]`. It also returns
a `ChoirResult`.

The explicit random-forest profile is for development only. It cannot support
a parity or release claim. A calibrated release will load its frozen profile
when `rf_profile` is omitted.

Read the comparison records:

```python
records = choir.getRecords(result)
```
