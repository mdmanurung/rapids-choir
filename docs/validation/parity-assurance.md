# Faithful R-to-Python implementation assurance

## Scope and claim boundary

The locked reference is CHOIR v0.3.0 at commit
`0db5e90ef46cd4f2f3765aa812325c8c768563fa`. The v0.1 parity surface is
unimodal RNA, AnnData, deterministic preparation, comparator behavior,
hierarchy construction, pruning and merging, singleton handling, final labels,
records, and provenance.

Plotting, multimodal analysis, Seurat, SingleCellExperiment, ArchR, count
splitting, and externally supplied cluster-tree inference are excluded. The only
claim this framework can authorize is:

> A faithful Python reimplementation of the unimodal-RNA CHOIR v0.3.0
> algorithm in compatibility mode within prespecified fidelity margins.

It cannot authorize bitwise identity, source-code identity, drop-in R-package
equivalence, parity for excluded functionality, or an R-parity claim for
corrected mode.

## Candidate evidence chain

```text
freeze one clean candidate
        │
        ├─ Gate 0: source, fixtures, scope, 45-entry ledger, schemas
        ├─ static Gate 1: lint, typing, tests, ledger, 12 mutations, package
        └─ real-L40S Gate 1
                 │
                 v
        parity framework READY
                 │
        ┌────────┴───────────────────────────┐
        v                                    v
comparator development                21 Splatter roles
R-vs-R margins + 12 profiles          R-vs-R hierarchy margins
        └───────────────┬────────────────────┘
                        v
 final candidate: exact RF winner/margins + all 21 roles
        ┌───────────────┴────────────────────┐
        v                                    v
sealed comparator holdout             Arigoni and Chen once each
        └───────────────┬────────────────────┘
                        v
             parity promotion readiness
                        │
                        v
       guarded publication / release evidence
```

Every artifact carries or is checksum-bound to the candidate commit, attempt,
candidate-manifest digest, and a runtime-tree digest covering every tracked
repository file, including release and publication materials. Candidate drift
stops the chain. Earlier candidate results and current development results
cannot be pooled into a new candidate.

## Exhaustive ledger

`validation/contracts/parity_ledger_v1.json` covers every R function and every
meaningful branch. Each entry records its R source span and SHA-256, semantic
responsibility, Python components, classification, expected state transition,
tests, comparison rule, and evidence checksum. The verifier requires exactly
the 45 routines in the frozen source snapshot and permits only these states:

- `matched_compatibility`
- `intentional_corrected_divergence`
- `excluded_v0_1`
- `unavailable_input`

An `unavailable_input` or pending entry remains unresolved until the
candidate-specific hierarchy aggregate and frozen margins close it. No code
coverage or static complexity value can close an entry.

## Canonical cross-language records

The private trace schema is
`validation/contracts/semantic_trace_v1.schema.json`. R and Python emit the
same ordered events for provenance, preprocessing, prepared-input validation,
resolution search and transitions, hierarchy nodes and stops, comparator
calls, multiplicity, pruning, bridges, merges, singleton handling, and final
relabeling.

Each event contains operation index, parameters, seed, input/output membership
SHA-256 values, topology context, decision, reason, bounded metrics, and an
event-specific payload. JSONL and Parquet artifacts are immutable and validated
by schema, SHA-256, run ID, row count, and contiguous operation order. The
hierarchy result envelope is defined by
`validation/contracts/hierarchy_result_v1.schema.json`; 256-bit node IDs use
decimal strings on the wire.

Python validation runner (inside a real L40S allocation and a reserved phase):

```bash
PYTHONPATH=src:validation/scripts python \
  validation/scripts/run_python_semantic_trace.py \
  --root . --candidate-manifest candidate.json \
  --resource-reservation reservation.json \
  --resource-ledger validation/evidence/governance/resource_ledger.jsonl \
  --resource-phase hierarchy_development \
  --h5ad prepared.h5ad --config compatibility.json \
  --dataset-id DATASET --output-jsonl python.jsonl \
  --output-result python.result.json
```

The R4_51 reference driver sources the checksum-verified CHOIR v0.3.0 snapshot,
runs fresh uninstrumented and instrumented environments, and refuses to emit a
bundle unless their extracted states are identical. It traces actual CHOIR
boundaries and writes an RDS bundle containing exactly `schema_version`,
`dataset_id`, `run_id`, `cell_ids`, `labels`, `tree`, `provenance`, and
`events`. The canonicalizer never infers or fills missing state:

```bash
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/Rscript \
  --vanilla validation/scripts/run_reference_hierarchy.R \
  --reference-root validation/reference/choir_v0_3_0 \
  --recipe reference-recipe.rds --python-result python.result.json \
  --seed 104729 --bundle reference-boundaries.rds \
  --equivalence reference.instrumentation.json
/exports/archive/hg-funcgenom-research/mdmanurung/conda/envs/R4_51/bin/Rscript \
  --vanilla validation/scripts/run_reference_semantic_trace.R \
  --bundle reference-boundaries.rds --candidate-manifest candidate.json \
  --output reference.jsonl --result reference.result.json
```

Trace generation is validation-only. It is not public API, and its time is
excluded from benchmarks.

## Numeric and dataset acceptance

Discrete fields and deterministic states match exactly. Pure deterministic
floating-point helpers use `rtol=1e-7` and `atol=1e-10`. Stochastic margins are
the R-vs-R development 97.5th percentile using NumPy's `linear` quantile
method and can never exceed 0.03.

Comparator confirmation requires every repeat to match decision, reason, pair
kind, short-circuit order, and expected decision. Every bounded field must lie
within its own frozen development margin. Majority voting and dataset pooling
are prohibited.

The development build uses its current `CALIBRATION_PROFILE` to produce all 21
hierarchy objects and derive margins. After the reviewed comparator profile and
hierarchy margins are committed, a clean final candidate reruns the complete
12-profile grid and all 21 objects. `final_candidate_reproduction` requires the
exact RF winner, ranking protocol, environment locks, comparator R-vs-R bounded
margins, and hierarchy margin payload. It does not equate the development and
final runtime-tree digests; those must differ when the reviewed artifacts are
added.

Hierarchy development uses the 21 role-exclusive objects in
`validation/contracts/hierarchy_validation_v1.json`. The calibration and margin
design freezes one R-vs-R pair (`104729` versus `104759`) for each role, giving
exactly 21 calibration records before the 97.5th percentile is derived. The
calibration and margin tools are:

```bash
PYTHONPATH=src:. python validation/scripts/calibrate_hierarchy_pair.py ...
PYTHONPATH=src:. python validation/scripts/derive_hierarchy_margins.py ...
PYTHONPATH=src:. python validation/scripts/evaluate_hierarchy_fidelity.py ...
```

The endpoint metrics are `1 - ARI`, pairwise false-split rate, pairwise
false-merge rate, and semantic-record discordance. Each development object must
pass its role. Same-seed operation types, order, decisions, reasons, comparator
pair kind, bounded-metric keys, and provenance are exact. Backend-sensitive
membership changes are still compared event by event and enter the frozen
record-discordance and endpoint margins. After the design and margins are
frozen, Arigoni and Chen are opened once; each must pass independently. The
two datasets are computational-parity confirmation only. They are not an
independent hierarchy-validation or external scientific-validation cohort. A
v2 detached authorization binds each dataset, its custody-deviation record,
the data-role contract, the promoted RF profile, and the final-candidate margin
digest before a single source-to-sealed-working-copy opening. The seal writes
an immutable attempt marker before reading the first raw byte. Failed,
contaminated, or inconclusive openings retain their failure evidence and cannot
be retried or replaced.

The final-candidate two-seed rerun must reproduce the committed margin payload
exactly. A changed endpoint or stochastic margin produces `DRIFTED` and stops
confirmation. The
aggregate rejects missing or duplicate datasets and does not calculate a
pooled acceptance metric:

```bash
PYTHONPATH=src:. python \
  validation/scripts/aggregate_hierarchy_fidelity.py \
  --root . --candidate-manifest candidate.json \
  --design validation/contracts/hierarchy_validation_v1.json \
  --margins hierarchy-margins.FROZEN.json \
  --development-role-contract \
    validation/contracts/hierarchy_development_objects_v1.json \
  --confirmation-role-contract validation/contracts/data_roles_v2.json \
  --development SPLATTER_REPORT ... \
  --confirmation ARIGONI_REPORT --confirmation CHEN_REPORT \
  --output hierarchy-fidelity.json
```

## Negative controls and promotion

`validation/scripts/run_mutation_controls.py` makes 12 temporary source-level
patches in isolated package copies. The probes cover
tail inequality and precedence, batch eligibility, multiplicity, seed
derivation, sampling order, resolution stopping, bridge selection, singleton
relabeling, pruning order, membership/cache identity, and stale-artifact
detection. All 12 must be rejected before holdout authorization.
Only an expected pytest assertion failure counts as a killed mutation; import,
collection, timeout, or infrastructure errors do not receive credit.

`validation/scripts/assess_parity_promotion.py` is the final fail-closed claim
check. It requires same-candidate Gate 0, static and real-L40S Gate 1, the
framework seal, comparator confirmation and seal, frozen hierarchy margins,
exact final-candidate RF-profile and hierarchy-margin reproduction, the
complete 21+2 hierarchy aggregate, and zero unresolved ledger entries. A
`BLOCKED` report authorizes no claim.

## Secondary, performance, and release evidence

`validation/contracts/publication_claims_v2.json` exclusively targets a BMC
Bioinformatics Software Article. `data_roles_v2.json` prevents Arigoni/Chen or
the corrected case studies from being described as independent validation.
Stoeckius, Kang, and pancreas reports must show compatibility and corrected
mode side by side for every named construct; valid unfavorable results remain
in the aggregate and no pooled biological-validity metric exists.

The Sant core has four required accounting records, each classified
`reproduced`, `partial`, `unverifiable`, or `failed`. Incompleteness withholds
only the optional completeness claim. The performance evaluator requires six
10,000-gene workloads, one warm-up per implementation, seven randomized and
count-balanced fresh-process pairs, full input-to-durable-output timing, and
absolute host/device peak memory per implementation. The preregistered seeded
order and content-bound correctness report for every pair are mandatory. It
emits a tested-L40S claim only when every output is correct and every workload's
one-sided 95% lower confidence bound is at least 2×.

The `publication_evidence` and `release_candidate` Snakemake targets are not in
`rule all`. They consume reviewed artifact paths from `workflow/config.yaml`
and fail closed while those paths are null. The release target first binds a
clean reproducible build to the parity-ready candidate; publication evidence
then binds that exact local release candidate. Tags, registries, Zenodo, GitHub
releases, preprints, and journal submission remain separately authorized
external actions.

## Current state

The executable producers and local assurance machinery exist, but no new clean
candidate has completed this chain. Hierarchy margins remain
`DEVELOPMENT_REQUIRED`; the R4_51 environment also lacks Splatter, so the 21
development objects cannot execute until that dependency is installed and
frozen. No promoted stable RF profile or final-candidate reproduction seal
exists. Arigoni and Chen remain unopened. Gate 2, promotion, publication, and
release therefore remain sealed.
