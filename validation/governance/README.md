# Resource custody

`resource_budget.yaml` freezes the v0.1 limits at 250 L40S-hours, 2 TB of
scratch, and zero cloud spend. Before a GPU phase, Snakemake appends a
hash-chained reservation to
`validation/evidence/governance/resource_ledger.jsonl`. An unreconciled
reservation counts at its full amount, so a failed or abandoned phase cannot
silently free budget.

The 1.5-hour Gate 1 reservation covers the scheduler smoke, worst-shard
canary, and functional Gate 1 job at their 30-minute caps. The 120-hour
development-grid reservation is created only after eligible R-vs-R margins
and Gate 1 pass, so an ineligible reference stops before that budget is held.

After every Slurm phase, the sole custodian obtains completed-job elapsed time
and peak scratch use from `sacct` and reconciles the reservation with
`validation/scripts/resource_ledger.py reconcile`. The reconciliation must
name the reservation ID, exact candidate commit and attempt, and actual use;
it fails if actual use exceeds the frozen reservation. Infrastructure and
technical failures still consume their actual resources and retain their
failure classification.

The Slurm profile deliberately omits Snakemake's `rerun-incomplete` option.
An admissible, unchanged infrastructure retry is represented by incrementing
`candidate_attempt`, which creates new output roots while retaining the failed
attempt. A technical or scientific failure is never retried under a new
attempt.

The ledger is append-only under the workflow tool: each entry contains the
previous entry hash, its own canonical SHA-256, the frozen-budget hash, and
the totals after the entry. Publication packaging must verify the complete
chain and include the raw Slurm accounting used for reconciliation.
