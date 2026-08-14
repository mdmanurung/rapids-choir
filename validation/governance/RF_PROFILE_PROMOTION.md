# RF profile promotion boundary

`gate2_calibration` produces a candidate-specific immutable profile at
`validation/evidence/candidates/<candidate>/<attempt>/gate2/rf_profile.candidate.json`.
The workflow never promotes this file automatically.

After all 12 development profiles and the R-vs-R variability contract pass,
the sole custodian reviews the ranking, hashes, raw failures, resource ledger,
and calibration candidate identity. If accepted, the custodian copies the
candidate file byte-for-byte to `validation/evidence/rf_profile.json` and
commits only that reviewed profile plus its intended provenance. A fresh clean
candidate then verifies that the calibration and confirmation runtime-tree
digests are identical.

The signed `comparator_confirm` authorization must contain the SHA-256 of this
tracked stable profile. Without that signature, a committed profile still
cannot open the holdout. A changed profile, runtime digest, contract, attempt,
or authorization creates a hard failure; confirmation never falls back to the
development profile or retunes it.
