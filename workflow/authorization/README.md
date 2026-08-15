# Confirmatory authorization

Confirmatory targets fail closed until the sole custodian supplies all three
local files configured in `workflow/config.yaml`:

- the JSON authorization manifest;
- its detached SSH signature, created with namespace
  `rapids-choir-confirmation-v1`;
- an OpenSSH `allowed_signers` file mapping identity `mdmanurung` to the
  authorized public key.

The manifest binds one target and ordered dataset-role list to an exact clean
candidate commit, runtime-tree digest, frozen RF-profile digest, a positive
attempt number, and a bounded validity interval. These files are intentionally
absent from source control. The workflow verifies them and creates an immutable
authorization seal before any confirmatory data rule can run.

Example unsigned manifest shape (do not reuse its values):

```json
{
  "schema_version": 1,
  "authorization_id": "replace-with-unique-id",
  "authorization_kind": "confirmatory_opening",
  "candidate_commit": "full-git-sha",
  "runtime_tree_sha256": "sha256-from-frozen-candidate",
  "rf_profile_sha256": "sha256-of-validation-evidence-rf-profile-json",
  "rf_profile_reproduction_sha256": "sha256-of-final-candidate-reproduction-seal",
  "target": "comparator_confirm",
  "operator": "mdmanurung",
  "attempt": 1,
  "dataset_roles": ["comparator_holdout"],
  "issued_at": "2026-08-14T00:00:00Z",
  "expires_at": "2026-08-15T00:00:00Z",
  "permit_holdout_opening": true
}
```
## Hierarchy confirmation

Authorization manifests and detached signatures are private operator inputs and
must not be committed or packaged. The tracked `allowed_signers` path is also
operator-managed and excluded from distributions.

Comparator confirmation continues to use the v1 manifest and the
`rapids-choir-confirmation-v1` SSH signature namespace. Its manifest also
binds the final candidate's `rf_profile.REPRODUCED.json`; the development
runtime recorded inside the promoted profile is provenance, not the final
candidate runtime.

Arigoni and Chen each require a distinct v2 manifest and detached signature in
the `rapids-choir-confirmation-v2` namespace. Each manifest must bind exactly:

- the unchanged candidate commit, runtime digest, and attempt;
- its single target (`hierarchy_confirm_Arigoni` or
  `hierarchy_confirm_Chen`) and single dataset role;
- SHA-256 values named `hierarchy_margins`, `data_roles`, `rf_profile`, `h5ad`,
  `recipe`, `python_config`, and `custody_record`;
- an unexpired one-time-opening permission.

The authorization check verifies the tracked contracts without reading the
sealed dataset. Before the first raw byte is read, the sealing rule writes an
immutable attempt marker. It then opens each source artifact once, checks its
signed digest, validates that the Python config uses the signed compatibility
profile, and publishes immutable candidate-scoped working copies. A failed or
contaminated attempt retains a failure record and cannot be retried; there is
no replacement hierarchy holdout.
