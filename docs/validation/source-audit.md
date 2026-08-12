# CHOIR v0.3.0 source audit

Status: **frozen for implementation**  
Reference: CHOIR `0.3.0`, commit
`0db5e90ef46cd4f2f3765aa812325c8c768563fa`  
Paper: Sant, Mucke, and Corces, *Nature Genetics* 57, 1309-1319
(2025), <https://doi.org/10.1038/s41588-025-02148-8>

This audit distinguishes the behavior of the frozen R code from the scientific
terminology used by `rapids-choir`. The executable source snapshot is under
`validation/reference/choir_v0_3_0`; its checksums are authoritative.

## Scope map

| R source | R functions | v0.1 disposition |
|---|---|---|
| `ComparisonUtils.R` | `.runPermutationTest`, `.runRF`, record and distance helpers | Comparator behavior ported; R-only implementations remain validation-only. |
| `compareClusters.R` | `compareClusters` | Implemented as `rapids_choir.tl.compare_clusters` and `ChoirComparator`. |
| `CHOIR.R` | `CHOIR` | Implemented only after the Gate 2 hard stop passes. |
| `buildTree.R` | `buildTree` | Root and node-local hierarchy behavior assigned to Gate 3. |
| `pruneTree.R` | `pruneTree` | Bottom-up pruning behavior assigned to Gate 3. |
| `TreeUtils.R` | dimensional reduction, resolution loops, hierarchy repair, subtree growth | Louvain/SNN portions assigned to Gate 3; Seurat, MRtree, ATAC, and multi-modal implementations are validation references only. |
| `HelperUtils.R` | object adapters, records, distance and naming helpers | AnnData adapters and records are ported; R object adapters are unsupported. |
| `ValidationUtils.R` | `.validInput` | Replaced by typed Python validation with explicit corrected/compatibility policies. |
| `buildParentTree.R`, `combineTrees.R` | atlas-scale split/combine workflow | Deferred; multi-object/atlas stitching is not v0.1 functionality. |
| `runCountSplit.R` | `runCountSplit` | Audited and explicitly excluded from v0.1. |
| `PlottingUtils.R` | palette, UMAP, plotting | Deferred; plotting is not v0.1 functionality. |
| `CHOIR-package.R`, `utils-pipe.R` | package docs and pipe export | No Python equivalent required. |

## Comparator contract recovered from source

The public R defaults are `alpha=0.05`, variable features, `n_iterations=100`,
`n_trees=50`, variance enabled, `min_accuracy=0.5`,
`min_connections=1`, `max_repeat_errors=20`, no sample cap, no minimum-read
threshold, and `random_seed=1`.

### Graph semantics

- `FindNeighbors` returns two distinct graphs. The `nn` graph is stored as
  `P*_graph_nn`; the `snn` graph is stored as `P*_graph_snn`.
- Connectivity screening uses only the NN graph. For clusters A and B, the R
  score is the sum of both directed blocks, `NN[A,B] + NN[B,A]`.
- Louvain/Leiden clustering uses only the SNN graph. The default
  `cluster_params$algorithm=1` is Seurat Louvain, so the compatible v0.1 path
  is GPU Louvain. A single ambiguous `neighbors_key` cannot silently stand in
  for both graph roles.

### Sampling and forest count

Without batch handling, R defines the per-class test size as

```text
n_sampled = min(sample_max,
                max(1, floor(downsampling_rate * min(n1, n2) / 2)))
```

Each round samples `2*n_sampled` cells from each cluster, then builds a
class-balanced half-train/half-test mask. It fits an observed-label forest on
each complementary half and repeats both fits after label permutation. One
round therefore performs four forests and yields two observed plus two null
accuracies.

The number of rounds is `round(n_iterations / 2)` using R's ties-to-even
rounding. Consequently, odd inputs can yield either one fewer or one more
accuracy than requested. Corrected mode rejects odd inputs and requires an
even `n_iterations >= 4`; compatibility mode preserves the captured behavior.

When batches are supplied, a batch is eligible within a cluster only if its
count is at least both 2 and 20% of the most prevalent batch in that cluster.
Only the intersection of eligible batch identities is used. No intersection
causes an immediate batch-dependent merge. Eligible identities repeat across
rounds, but R's per-batch accuracy summaries index the flattened two-accuracy
array with round positions rather than mapping both complementary accuracies
back to each batch. Compatibility records preserve this positional result;
corrected records align by explicit batch identity.

### Feature and connectivity short circuits

Decision order begins before any forest fit:

1. Insufficient NN connectivity means `split: min connections`.
2. No jointly eligible batch means `merge: batch-dependent`.
3. No feature surviving the comparison-local filter means
   `merge: minimum reads`.

The feature matrix is subset to the two memberships first. With no
`min_reads`, features whose combined column sum is zero are removed. The GPU
port filters sparse columns before any bounded local densification and never
densifies the full dataset.

### Accuracy, variance, and tail quantities

Ranger's reported holdout prediction error is converted to accuracy as
`1 - prediction.error`. Because classes and train/test halves are balanced,
this is overall accuracy; the Python record name is `overall_accuracy`.

For compatibility, the observed mean is compared with the upper tail of a
Gaussian KDE fit to null accuracies:

```text
accuracy_tail = 1 - KDE_CDF(null_accuracies, mean(observed_accuracies))
```

The observed accuracy variance is compared with the lower tail of a KDE fit to
bootstrap variances of the null array:

```text
variance_tail = KDE_CDF(bootstrap_var(null_accuracies), var(observed_accuracies))
```

These are smoothing-dependent reference tail probabilities, not empirical
permutation p-values. `rapids-choir` therefore never calls them p-values and
does not claim formal FDR or type-I-error control.

### Repeated-error quirk

The R implementation records `which(prediction != class)` after each fresh
downsampling operation. These are positions in the newly sampled vectors, not
stable cell identities. Aggregating them across rounds can therefore conflate
different cells that occupied the same sampled position. Corrected mode tracks
stable cell indices. Compatibility mode reproduces sampled-position behavior.

### Decision order

After the short circuits, the frozen R rule is:

1. `split` when mean accuracy meets the minimum and both required original
   tail scores are below alpha.
2. Otherwise, with repeated errors enabled, `split: repeat error` when the
   original scores are below `2*alpha`, modified scores are below alpha, and
   repeated-error counts stay under both configured and 30%-of-cluster caps.
3. Otherwise, `merge: min accuracy` when mean accuracy is below the minimum
   even though the original tail criteria pass.
4. Otherwise, `merge`.

Corrected mode retains this ordering but uses stable identities and explicit
metric names. Every outcome receives a stable Python reason code.

### Randomness

The frozen R comparator accepts `random_seed`, but `.runPermutationTest` and
`.runRF` never use it. Parent functions switch to `L'Ecuyer-CMRG` for
multi-core execution without calling `set.seed`. Seurat reductions and
clustering do receive the seed. Compatibility mode follows reference call
ordering where possible; corrected mode derives stable child seeds from the
root seed plus canonical operation identity.

## Hierarchy and pruning contract recovered from source

- The default hierarchy uses Seurat Louvain (`algorithm=1`) over the SNN
  graph. Leiden (`algorithm=4`) is permitted upstream but is not a compatibility
  path in v0.1.
- Starting resolution compares partitions at resolutions 0 and 1. If their
  cluster counts match, it searches upward by 0.5 below 4, by 1 below 10, and
  by 5 thereafter. If resolution 1 has more clusters, it searches downward by
  0.3 to 0.1 and powers of ten below 0.1.
- A tree starts at the selected resolution with gap equal to half that value.
  A jump greater than both 25% and three clusters halves the gap, at most three
  consecutive times. After five unchanged gap steps, the gap doubles.
- The silhouette root stops five accepted levels after the maximum silhouette;
  later levels are discarded. Tree growth also stops if the largest cluster
  has at most two cells, cluster count decreases, clustering fails, or an
  explicit maximum is exceeded.
- Non-hierarchical resolution partitions are reconciled by MRtree in the R
  reference. The Python topology must instead maintain strict parent-child
  invariants directly and validate them after each mutation.
- Pruning proceeds from the bottom two levels upward. Child pairs are visited
  in stored order, with singleton children automatically merged. Ambiguous
  overlapping merge groups trigger two bridge comparisons before connected
  merge components are relabeled.
- Dynamic adjustment starts at the tree top after at least one split, or after
  50 comparisons/one split for Bonferroni and 100 comparisons/five splits for
  FDR. Bonferroni uses `alpha / number_of_accuracy_scores`.
- The R FDR expression calculates accuracy and variance candidates but then
  takes `max(variance_candidate, variance_candidate)`. Compatibility mode
  preserves this duplicated-variance expression; corrected mode combines the
  accuracy and variance candidates.
- Tree labels use `P<root>_L<level>_<ordinal>`. Final categorical labels are
  integers ordered from largest to smallest cluster; ties retain deterministic
  canonical ordering in the Python port.
- In `.optimizeTree`, a node-local reduction is created but the full reduction
  is passed to centroid-distance calculation. Compatibility mode captures that
  representation-selection quirk; corrected mode uses the node-local subset.

## Count splitting audit

`runCountSplit` supports Seurat and SingleCellExperiment but rejects ArchR. It
uses `countsplit::countsplit`, defaults to suffixes `_1`/`_2`, and optionally
normalizes each split. The build and prune stages may then consume different
split matrices. Count splitting is validation-only and excluded from v0.1; the
runtime package has no hidden R or countsplit dependency.

## Frozen ambiguities and executable evidence

The following are covered by R-generated fixture records before Python
translation: odd iterations, batch eligibility and positional summaries,
sampled-position repeated errors, duplicated-variance FDR adjustment,
node-local distance selection, disconnected graphs, random-seed handling,
decision ordering, KDE/CDF, and fixed-index bootstrap variance. Dataset split,
licenses, archive checksums, seeds, and acceptance criteria are frozen in
`validation/fixtures/specs/datasets.yaml`.

Gate 0 passes only when Snakemake regenerates all fixture outputs under R
4.5.1, verifies the source snapshot, and produces a matching manifest.
