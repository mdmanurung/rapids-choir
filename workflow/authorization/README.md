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
attempt number, and a bounded validity interval. These files are intentionally absent from source
control. The workflow verifies them and creates an immutable authorization
seal before any confirmatory data rule can run.

Example unsigned manifest shape (do not reuse its values):

```json
{
  "schema_version": 1,
  "authorization_id": "replace-with-unique-id",
  "authorization_kind": "confirmatory_opening",
  "candidate_commit": "full-git-sha",
  "runtime_tree_sha256": "sha256-from-frozen-candidate",
  "rf_profile_sha256": "sha256-of-validation-evidence-rf-profile-json",
  "target": "comparator_confirm",
  "operator": "mdmanurung",
  "attempt": 1,
  "dataset_roles": ["comparator_holdout"],
  "issued_at": "2026-08-14T00:00:00Z",
  "expires_at": "2026-08-15T00:00:00Z",
  "permit_holdout_opening": true
}
```
