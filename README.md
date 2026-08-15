# rapids-choir

[![CI](https://github.com/mdmanurung/rapids-choir/actions/workflows/ci.yml/badge.svg)](https://github.com/mdmanurung/rapids-choir/actions/workflows/ci.yml)
[![Pages](https://github.com/mdmanurung/rapids-choir/actions/workflows/pages.yml/badge.svg)](https://github.com/mdmanurung/rapids-choir/actions/workflows/pages.yml)

`rapids-choir` is an experimental, single-GPU port of CHOIR for unimodal RNA
`AnnData` objects. The behavioral reference is CHOIR v0.3.0 at commit
`0db5e90ef46cd4f2f3765aa812325c8c768563fa`.

This repository is under gated scientific validation. The unimodal-RNA
preprocessing, hierarchy, pruning, final-label, record, and provenance surfaces
are implemented, but the parity claim is **not yet authorized**. A passing CPU
test suite does not constitute cross-language or GPU functional validation. The
supported v0.1 baseline is Python 3.12, RAPIDS 26.06,
rapids-singlecell 0.16.1, CUDA 12.8, and one L40S.

The default `rules="corrected"` repairs captured reference quirks. The optional
`rules="choir_v0_3_0"` mode supports behavioral comparison, but neither mode
claims numerical identity with ranger. Reported tail quantities are KDE-derived
reference tail probabilities, not permutation p-values and not confirmatory
FDR guarantees.

The public analysis surface is:

```python
import rapids_choir as rc

rc.pp.prepare_rna(adata)  # mutates and returns None; pass copy=True for a copy
result = rc.CHOIR(adata, batch_labels="donor", rules="corrected")
```

`CHOIR` retains R CHOIR names for shared scientific parameters and adds
explicit AnnData and GPU controls. It always returns a `ChoirResult`. It
rejects output collisions unless
`overwrite=True`, and optional Parquet records are content-bound to their run
ID, schema, row count, input checksum, and file checksum.

The safe workflow default freezes a clean candidate and runs only Gate 0 plus
the static part of Gate 1:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --cores 1
```

Evidence is written below
`validation/evidence/candidates/candidate-<commit>/attempt-<n>/`. A dirty
tracked tree, untracked runtime file, stale candidate, existing output, or
inherited Slurm allocation used with the Slurm executor is rejected.

## Candidate and parity gates

The previous candidate remains diagnostic evidence only. These assurance
contracts create a new candidate, so its Gate 0, static Gate 1, real-L40S Gate
1, comparator fixtures, 12-profile grid, development calibration, and all
downstream evidence must be rerun without pooling prior results.

Comparator Gate 2 remains sealed until the same clean candidate has produced
`parity/framework.READY.json`. Hierarchy code may exist under D020, but neither
the holdout nor the parity claim opens merely because implementation or local
tests pass. The parent Zenodo archive is retained under the configured LTS raw
root; selected extracted objects and H5AD files stay under the configured
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
new local commit. Do not push. The next clean candidate reruns the complete
grid and must reproduce the exact winner, ranking protocol, environment locks,
and R-vs-R bounded margins. Its runtime-tree digest is expected to differ from
the development build because the frozen profile and release materials have
been added. After `final_candidate_reproduction` passes, a current detached
SSH authorization described in `workflow/authorization/README.md` is still
required; only then run:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm final_candidate_reproduction
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/snakemake \
  --profile workflow/profiles/slurm comparator_confirm
```

`gate2_development` has no holdout dependency. Calibration expands the frozen
12-profile grid into typed-L40S dataset-by-repeat shards, then verifies and
aggregates exactly five same-seed repeats per pair before ranking profiles.
Holdout acceptance uses the same shard contract and only the committed winner.
Every repeat and every holdout dataset must pass; a negative report is retained
but cannot create the confirmation pass seal.

Full parity promotion additionally requires the 21 role-exclusive Splatter
development objects, frozen R-vs-R 97.5th-percentile margins capped at 0.03,
an exact final-candidate margin rerun, and one-time computational-parity passes
for Arigoni and Chen. Those datasets are not independent hierarchy validation.
The 21 development objects use the current calibration winner; confirmation
configs must use the separately promoted stable profile. Before either
confirmation dataset is read, its seal writes an immutable attempt marker. A
failed opening cannot be retried or replaced. The evaluator never pools
datasets or rescues a failed repeat by majority vote.
See
[`docs/validation/parity-assurance.md`](docs/validation/parity-assurance.md) and
[`validation/STATUS.md`](validation/STATUS.md) for the exact evidence boundary.

No CPU backend or fallback is provided. Importing `rapids_choir` is deliberately
CUDA-independent; GPU dependencies are loaded only when GPU functionality is
called.

The opt-in `publication_evidence` and `release_candidate` targets remain inert
while their reviewed paths in `workflow/config.yaml` are null. They assemble
local evidence only: release readiness binds the parity-ready candidate first,
then publication readiness binds that exact release record. They never push,
tag, upload, release, preprint, or submit.

See `NOTICE` for derivative-work attribution and non-affiliation terms.
