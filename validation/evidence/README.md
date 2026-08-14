# Evidence layout

The legacy root-level `gate0.json`, `gate1_static.json`, `gate1_gpu.json`, and
`gate1_gpu.FAILED.json` predate candidate-specific custody. They are retained
unaltered as historical evidence and cannot satisfy current Snakemake targets.
In particular, the restored Gate 1 pass is not current-candidate evidence.

Current attempts live under
`candidates/candidate-<full-commit>/attempt-<n>/`. Each attempt begins with an
immutable clean-runtime manifest and keeps its gate reports, calibration
chunks, Parquet records, resource reservations, and failures in that root.
Changing tracked runtime code changes the commit and therefore the evidence
root. An admissible infrastructure rerun increments the attempt and retains the
previous attempt.

`rf_profile.json` is the sole stable promotion input. It may appear only after
the reviewed candidate-specific calibration profile is copied byte-for-byte
and committed under the custodian procedure in
`validation/governance/RF_PROFILE_PROMOTION.md`. Its content binds the source
candidate, runtime digest, contracts, inputs, locks, margins, raw calibration
artifacts, and winning profile. A separate signed authorization binds that
hash before comparator holdout access.

`incidents/` retains infrastructure evidence. The global hash-chained resource
ledger lives under `governance/`; its entries remain consumed at the full
reservation until reconciled against completed Slurm accounting.
