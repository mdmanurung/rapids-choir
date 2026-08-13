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

## Gate 2 comparator validation

Gate 2 is a hard stop: hierarchy work is prohibited until the sealed holdout
evidence passes. The parent Zenodo archive is retained under the configured LTS
raw root; selected extracted objects and H5AD files stay under the configured
scratch work root.

Run the stages separately with the R 4.5.1 Snakemake executable and the tracked
Slurm profile:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm gate2_development
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm gate2_calibration
```

After calibration, review and locally commit only
`validation/evidence/rf_profile.json`. Do not push. Holdout extraction fails
unless that file is tracked, committed, and unchanged. Only then run:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm gate2_holdout
```

`gate2_development` has no holdout dependency. Calibration expands the frozen
12-profile grid into typed-L40S dataset-by-repeat shards, then verifies and
aggregates exactly five same-seed repeats per pair before ranking profiles.
Holdout acceptance uses the same shard contract and only the committed winner;
it writes `validation/evidence/gate2.json`.

No CPU backend or fallback is provided. Importing `rapids_choir` is deliberately
CUDA-independent; GPU dependencies are loaded only when GPU functionality is
called.

See `NOTICE` for derivative-work attribution and non-affiliation terms.
