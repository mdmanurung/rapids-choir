# Decision log

Decisions are frozen before RF calibration. Changes require a new dated entry
and invalidate downstream evidence that depends on the old contract.

| ID | Decision | Rationale |
|---|---|---|
| D001 | Behavioral reference is CHOIR v0.3.0 commit `0db5e90ef46cd4f2f3765aa812325c8c768563fa`. | Avoid moving-target parity. |
| D002 | Default rules are `corrected`; `choir_v0_3_0` is compatibility-only. | Preserve known quirks without making them scientific defaults. |
| D003 | Tail quantities are named reference tail probabilities, never formal permutation p-values. | KDE smoothing plus RF-algorithm substitution does not justify inferential guarantees. |
| D004 | Stable baseline is Python 3.12, RAPIDS 26.06, rapids-singlecell 0.16.1, CUDA 12.8. | This is the predeclared stable matrix. CUDA 13 is a separate future gate. |
| D005 | Functional hardware is one typed `L40S` via `--gres=gpu:L40S:1`. | Login GPU is unavailable and Titan Xp is below the supported baseline. |
| D006 | Hierarchy compatibility uses Louvain on SNN; Leiden is deferred. | Upstream default `algorithm=1` is Louvain. |
| D007 | NN connectivity and SNN clustering graphs are distinct required inputs. | Upstream uses distinct graph objects with different semantics. |
| D008 | Fixed mode is fail-closed on VRAM and never adapts the scientific workload. | Prevent hidden changes to comparisons after OOM. |
| D009 | cuML calibration grid is exactly `n_bins={128,256,512}`, `max_depth={16,None}`, `n_streams={1,4}` with all other RF parameters pinned. | Prevent holdout-driven tuning. |
| D010 | Calibration winner order is macro decision concordance, then accuracy-distribution error, then runtime. | Scientific agreement takes precedence over speed. |
| D011 | Holdout identities and acceptance thresholds are frozen before calibration. | Avoid evaluation leakage. |
| D012 | Gate 2 is a hard stop. No hierarchy implementation begins without complete L40S and scientific acceptance evidence. | Comparator discrepancy would invalidate tree work. |
| D013 | MuData, ATAC, count splitting, adaptive permutations, multi-GPU, WNN, plotting, and CPU fallback are post-v1. | Keep v0.1 scientifically and operationally narrow. |
| D014 | No push, tag, package publication, container publication, or release without explicit authorization. | Promotion is distinct from implementation and validation. |
| D015 | `validation/contracts/comparator_candidate_v2.yaml` supersedes D010: only profiles with eligible R-vs-R variability are ranked by macro joint decision/reason concordance, numeric accuracy-distribution error, runtime, then the frozen parameter ties. | The original D010 omitted reason fidelity, the reference-variability margin, and deterministic ties needed before one-shot confirmation. |
| D016 | Hierarchy fields in `validation/fixtures/specs/datasets.yaml` remain immutable historical Gate 2 metadata and are not publication promotion criteria. | Publication roles and criteria are versioned separately without rewriting earlier evidence. |
| D017 | The v2 comparator contract supersedes D011's decision-only holdout thresholds: every GPU repeat must exactly match the frozen R decision and reason, and every bounded record field must be within its eligible development-derived margin capped at 0.03; the historical transfer and VRAM safeguards remain binding. | Majority voting, pooled datasets, or loose static numeric thresholds could conceal a comparator-component failure. |
| D018 | Comparator result caching is disabled for v0.1. | Complete content-derived identities would have to cover matrices, graphs, memberships, batches, masks, seeds, mode, and RF profile; pointer identity cannot establish evidence-safe cache validity. |
| D019 | The evidence Slurm profile does not enable automatic incomplete-job reruns. | An admissible infrastructure retry requires a documented new `candidate_attempt`; technical and scientific failures cannot be silently resumed in the same attempt. |
