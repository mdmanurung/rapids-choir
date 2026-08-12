# rapids-choir

`rapids-choir` is an experimental, single-GPU port of CHOIR for unimodal RNA
`AnnData` objects. The behavioral reference is CHOIR v0.3.0 at commit
`0db5e90ef46cd4f2f3765aa812325c8c768563fa`.

This repository is under gated scientific validation. A passing CPU test suite
does **not** constitute functional validation. The supported v0.1 baseline is
Python 3.12, RAPIDS 26.06, rapids-singlecell 0.16.1, CUDA 12.8, and one L40S.

The default `rules="corrected"` repairs captured reference quirks. The optional
`rules="choir_v0_3_0"` mode supports behavioral comparison, but neither mode
claims numerical identity with ranger. Reported tail quantities are KDE-derived
reference tail probabilities, not permutation p-values and not confirmatory
FDR guarantees.

Development evidence is built by Snakemake:

```bash
snakemake --cores 1 validation/evidence/gate0.json
```

No CPU backend or fallback is provided. Importing `rapids_choir` is deliberately
CUDA-independent; GPU dependencies are loaded only when GPU functionality is
called.

See `NOTICE` for derivative-work attribution and non-affiliation terms.
