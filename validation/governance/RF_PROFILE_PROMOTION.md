# RF profile promotion boundary

`gate2_calibration` produces a candidate-specific immutable profile at
`validation/evidence/candidates/<candidate>/<attempt>/gate2/rf_profile.candidate.json`.
The workflow never promotes this file automatically.

After all 12 development profiles and the R-vs-R variability contract pass,
the sole custodian reviews the ranking, hashes, raw failures, resource ledger,
and calibration candidate identity. If accepted, the custodian copies the
candidate file byte-for-byte to `validation/evidence/rf_profile.json` and
commits only that reviewed profile plus its intended provenance. A fresh clean
candidate reruns the complete 12-profile grid. The
`final_candidate_reproduction` target requires the same winning profile,
ranking protocol, environment locks, and R-vs-R bounded margins before any
holdout authorization. The development and final runtime-tree digests are
expected to differ because the reviewed profile, frozen margins, package data,
and release materials are added between those phases; they are never treated
as interchangeable evidence.

The signed `comparator_confirm` authorization must contain the SHA-256 of this
tracked stable profile and the final-candidate profile-reproduction seal.
Without that signature, a committed profile still cannot open the holdout. A
changed profile, final-candidate runtime digest, reproduction seal, contract,
attempt, or authorization creates a hard failure; confirmation never falls
back to the development profile or retunes it.
