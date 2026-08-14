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

The safe default freezes a clean candidate and runs only Gate 0 plus the static
part of Gate 1:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --cores 1
```

Evidence is written below
`validation/evidence/candidates/candidate-<commit>/attempt-<n>/`. A dirty
tracked tree, untracked runtime file, stale candidate, existing output, or
inherited Slurm allocation used with the Slurm executor is rejected.

## Gate 2 comparator validation

Gate 2 is a hard stop: hierarchy work is prohibited until the sealed holdout
evidence passes. The parent Zenodo archive is retained under the configured LTS
raw root; selected extracted objects and H5AD files stay under the configured
scratch work root.

From a login shell outside any existing Slurm allocation, first requalify the
candidate on one real L40S. Development and calibration remain separate
targets:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm candidate_requalification
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm development_ready
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm gate2_calibration
```

After calibration, the custodian reviews the complete grid and copies the
candidate profile byte-for-byte to `validation/evidence/rf_profile.json` in a
new local commit. Do not push. The next clean candidate must have the same
runtime-tree digest as calibration. It also requires a current, detached SSH
authorization described in `workflow/authorization/README.md`; only then run:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm comparator_confirm
```

`gate2_development` has no holdout dependency. Calibration expands the frozen
12-profile grid into typed-L40S dataset-by-repeat shards, then verifies and
aggregates exactly five same-seed repeats per pair before ranking profiles.
Holdout acceptance uses the same shard contract and only the committed winner.
Every repeat and every holdout dataset must pass; a negative report is retained
but cannot create the confirmation pass seal. Hierarchy implementation remains
prohibited until that seal exists.

No CPU backend or fallback is provided. Importing `rapids_choir` is deliberately
CUDA-independent; GPU dependencies are loaded only when GPU functionality is
called.

See `NOTICE` for derivative-work attribution and non-affiliation terms.
