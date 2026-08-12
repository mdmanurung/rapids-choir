#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) {
  stop("Usage: generate_logic_fixtures.R OUTPUT_DIR")
}

output_dir <- normalizePath(args[[1]], mustWork = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

if (!identical(as.character(getRversion()), "4.5.1")) {
  stop("Reference fixtures require R 4.5.1; found ", getRversion())
}

required <- c("Matrix", "dplyr", "jsonlite", "spatstat.univar")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0L) {
  stop("Missing required R packages: ", paste(missing, collapse = ", "))
}

repo_root <- normalizePath(file.path(dirname(output_dir), "..", ".."))
reference_r <- file.path(repo_root, "validation", "reference", "choir_v0_3_0", "R")

source(file.path(reference_r, "HelperUtils.R"), local = .GlobalEnv)
source(file.path(reference_r, "ComparisonUtils.R"), local = .GlobalEnv)
source(file.path(reference_r, "TreeUtils.R"), local = .GlobalEnv)

write_fixture <- function(name, value) {
  jsonlite::write_json(
    value,
    path = file.path(output_dir, paste0(name, ".json")),
    auto_unbox = TRUE,
    pretty = TRUE,
    digits = 17,
    na = "null",
    null = "null"
  )
}

cells1 <- paste0("a", seq_len(8L))
cells2 <- paste0("b", seq_len(8L))
all_cells <- c(cells1, cells2)
feature_matrix <- Matrix::Matrix(
  matrix(seq_len(length(all_cells) * 4L), nrow = length(all_cells), ncol = 4L),
  sparse = TRUE
)
rownames(feature_matrix) <- all_cells
colnames(feature_matrix) <- paste0("g", seq_len(4L))

connected_nn <- Matrix::Matrix(1, nrow = length(all_cells), ncol = length(all_cells), sparse = TRUE)
dimnames(connected_nn) <- list(all_cells, all_cells)
disconnected_nn <- Matrix::Matrix(0, nrow = length(all_cells), ncol = length(all_cells), sparse = TRUE)
dimnames(disconnected_nn) <- list(all_cells, all_cells)

records_template <- function(batch = FALSE) {
  columns <- c(
    "comparison", "cluster1_size", "cluster2_size", "sample_size",
    "mean_accuracy", "var_accuracy", "mean_errors",
    "mean_permuted_accuracy", "var_permuted_accuracy",
    "percentile_accuracy", "percentile_variance"
  )
  if (batch) {
    columns <- c(
      columns,
      "batches_used", "batch_mean_accuracies", "batch_var_accuracies"
    )
  }
  columns <- c(columns, "connectivity", "decision")
  result <- as.data.frame(matrix(nrow = 0L, ncol = length(columns)))
  colnames(result) <- columns
  result
}

run_reference <- function(
    nn_matrix = connected_nn,
    cluster1_batches = NULL,
    cluster2_batches = NULL,
    n_iterations = 4L,
    random_seed = 1L,
    batch_records = FALSE) {
  .runPermutationTest(
    cluster1_name = "A",
    cluster1_cells = cells1,
    cluster1_cell_batches = cluster1_batches,
    cluster2_name = "B",
    cluster2_cells = cells2,
    cluster2_cell_batches = cluster2_batches,
    alpha = 0.05,
    n_iterations = n_iterations,
    n_trees = 5L,
    use_variance = TRUE,
    min_accuracy = 0.5,
    min_connections = 1L,
    max_repeat_errors = 0L,
    collect_all_metrics = FALSE,
    sample_max = Inf,
    downsampling_rate = 1,
    min_reads = NULL,
    input_matrix = feature_matrix,
    nn_matrix = nn_matrix,
    comparison_records = records_template(batch = batch_records),
    feature_importance_records = NULL,
    n_cores = 1L,
    random_seed = random_seed
  )
}

# R uses ties-to-even round(), so odd requested counts do not have one stable
# interpretation. The forest total follows directly from the frozen loop.
odd_n <- c(1L, 3L, 5L, 7L, 9L, 101L)
rounds <- round(odd_n / 2)
write_fixture("odd_iterations", list(
  requested_iterations = odd_n,
  split_rounds = rounds,
  observed_accuracies = 2L * rounds,
  null_accuracies = 2L * rounds,
  forest_fits = 4L * rounds
))

# Connectivity short circuit executes the frozen function and must not fit RF.
connectivity_output <- run_reference(nn_matrix = disconnected_nn)
write_fixture("disconnected_graph", list(
  decision = connectivity_output$result,
  reason = connectivity_output$comparison_records$decision[[1]],
  connectivity = connectivity_output$comparison_records$connectivity[[1]]
))

# Batch eligibility is based on identity intersection after the 2-cell/20% rule.
no_batch_output <- run_reference(
  cluster1_batches = rep("batch_a", length(cells1)),
  cluster2_batches = rep("batch_b", length(cells2)),
  batch_records = TRUE
)
write_fixture("batch_eligibility", list(
  cluster1_frequencies = list(batch_a = length(cells1)),
  cluster2_frequencies = list(batch_b = length(cells2)),
  eligible_intersection = character(),
  decision = no_batch_output$result,
  reason = no_batch_output$comparison_records$decision[[1]]
))

# Replace only the RF leaf with captured arrays; all aggregation remains the
# exact frozen R implementation. Two accuracies are emitted per round.
upstream_run_rf <- .runRF
.runRF <- function(i, ...) {
  if (i == 1L) {
    list(
      balanced_accuracy = c(0.60, 0.62),
      permutation_balanced_accuracy = c(0.48, 0.52)
    )
  } else {
    list(
      balanced_accuracy = c(0.80, 0.82),
      permutation_balanced_accuracy = c(0.49, 0.51)
    )
  }
}
batch_labels <- rep(c("batch_a", "batch_b"), each = 4L)
batch_output <- run_reference(
  cluster1_batches = batch_labels,
  cluster2_batches = batch_labels,
  n_iterations = 4L,
  batch_records = TRUE
)
batch_record <- batch_output$comparison_records[1, ]
write_fixture("batch_positional_metrics", list(
  round_batches = c("batch_a", "batch_b"),
  flattened_observed_accuracies = c(0.60, 0.62, 0.80, 0.82),
  reference_batch_mean_accuracies = batch_record$batch_mean_accuracies[[1]],
  corrected_batch_mean_accuracies = list(batch_a = 0.61, batch_b = 0.81)
))

# Stable cells can occupy different sampled positions. The frozen R code counts
# the positions returned by which(), while corrected mode counts cell identity.
sample_order_1 <- c("cell_a", "cell_b")
sample_order_2 <- c("cell_b", "cell_a")
position_errors <- c(
  which(sample_order_1 == "cell_a"),
  which(sample_order_2 == "cell_a")
)
write_fixture("repeat_error_identity", list(
  sampled_orders = list(sample_order_1, sample_order_2),
  stable_error_ids = c("cell_a", "cell_a"),
  reference_position_errors = position_errors,
  reference_position_frequencies = as.list(table(position_errors)),
  corrected_identity_frequencies = list(cell_a = 2L)
))

# The random_seed formal is unused by the comparator. This captured stub uses
# the ambient RNG to demonstrate both sides of the quirk.
.runRF <- function(i, ...) {
  list(
    balanced_accuracy = stats::runif(2L, min = 0.55, max = 0.90),
    permutation_balanced_accuracy = stats::runif(2L, min = 0.40, max = 0.60)
  )
}
random_case <- function(global_seed, parameter_seed) {
  set.seed(global_seed)
  output <- run_reference(n_iterations = 6L, random_seed = parameter_seed)
  output$comparison_records$mean_accuracy[[1]]
}
random_results <- c(
  same_global_seed_parameter_1 = random_case(2024L, 1L),
  same_global_seed_parameter_999 = random_case(2024L, 999L),
  different_global_seed_parameter_1 = random_case(2025L, 1L)
)
write_fixture("random_state", list(
  results = random_results,
  parameter_seed_changes_result = !isTRUE(all.equal(random_results[[1]], random_results[[2]])),
  ambient_seed_changes_result = !isTRUE(all.equal(random_results[[1]], random_results[[3]]))
))
.runRF <- upstream_run_rf

# Captured decision-rule oracles. Names are Python stable reason codes; R labels
# are retained explicitly for compatibility records.
reference_decision <- function(metrics, alpha = 0.05, use_variance = TRUE,
                               min_accuracy = 0.5, max_repeat_errors = 20,
                               cluster1_size = 20, cluster2_size = 20) {
  if (all(
    metrics$mean_accuracy >= min_accuracy,
    metrics$accuracy_tail < alpha,
    metrics$variance_tail < alpha || !use_variance
  )) {
    return(c(decision = "split", reason = "split"))
  }
  if (max_repeat_errors > 0) {
    if (all(
      metrics$mean_accuracy >= min_accuracy,
      metrics$accuracy_tail < alpha * 2,
      metrics$variance_tail < alpha * 2 || !use_variance,
      metrics$modified_accuracy_tail < alpha,
      metrics$modified_variance_tail < alpha || !use_variance,
      metrics$n_repeat_errors1 <= min(max_repeat_errors, 0.3 * cluster1_size),
      metrics$n_repeat_errors2 <= min(max_repeat_errors, 0.3 * cluster2_size)
    )) {
      return(c(decision = "split", reason = "split: repeat error"))
    }
    if (all(
      metrics$mean_accuracy < min_accuracy,
      metrics$accuracy_tail < alpha,
      metrics$variance_tail < alpha || !use_variance
    )) {
      return(c(decision = "merge", reason = "merge: min accuracy"))
    }
  }
  c(decision = "merge", reason = "merge")
}
base_metrics <- list(
  mean_accuracy = 0.80,
  accuracy_tail = 0.20,
  variance_tail = 0.20,
  modified_accuracy_tail = 0.20,
  modified_variance_tail = 0.20,
  n_repeat_errors1 = 1L,
  n_repeat_errors2 = 1L
)
decision_cases <- list(
  direct_split = modifyList(base_metrics, list(accuracy_tail = 0.01, variance_tail = 0.02)),
  repeat_error_split = modifyList(base_metrics, list(
    accuracy_tail = 0.06,
    variance_tail = 0.07,
    modified_accuracy_tail = 0.01,
    modified_variance_tail = 0.02
  )),
  minimum_accuracy_merge = modifyList(base_metrics, list(
    mean_accuracy = 0.40,
    accuracy_tail = 0.01,
    variance_tail = 0.02
  )),
  ordinary_merge = base_metrics
)
decision_results <- lapply(decision_cases, reference_decision)
write_fixture("decision_rules", list(inputs = decision_cases, outputs = decision_results))

# Frozen KDE/CDF and fixed-index bootstrap arithmetic.
observed <- c(0.49, 0.52, 0.50, 0.53, 0.48, 0.54, 0.51, 0.47, 0.55, 0.505)
null <- c(0.48, 0.51, 0.49, 0.52, 0.47, 0.53, 0.50, 0.46, 0.54, 0.495)
index_matrix <- matrix(c(
  1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
  1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
  10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
  2, 4, 6, 8, 10, 1, 3, 5, 7, 9,
  5, 5, 5, 1, 2, 3, 8, 9, 10, 10
), nrow = 5L, byrow = TRUE)
fixed_bootstrap_variances <- apply(index_matrix, 1L, function(index) stats::var(null[index]))
accuracy_tail <- 1 - suppressPackageStartupMessages(
  spatstat.univar::CDF(stats::density(null))(mean(observed))
)
variance_tail <- spatstat.univar::CDF(stats::density(fixed_bootstrap_variances))(
  stats::var(observed)
)
write_fixture("kde_bootstrap", list(
  observed_accuracies = observed,
  null_accuracies = null,
  observed_mean = mean(observed),
  observed_variance = stats::var(observed),
  fixed_bootstrap_indices_1_based = unname(split(index_matrix, row(index_matrix))),
  fixed_bootstrap_variances = fixed_bootstrap_variances,
  accuracy_upper_tail = accuracy_tail,
  variance_lower_tail = variance_tail,
  density_defaults = list(kernel = "gaussian", n = 512L, bw = "nrd0")
))

# Multiple-comparison arithmetic, including the duplicated variance expression.
alpha <- 0.05
old_adjusted_alpha <- 0.05
p_accuracy <- c(0.001, 0.002, 0.003, 0.9)
p_variance <- c(0.001, 0.5, 0.6, 0.7)
bh_candidate <- function(values) {
  passing <- values[stats::p.adjust(values, method = "fdr") < alpha]
  if (length(passing) == 0L) {
    old_adjusted_alpha
  } else {
    (max(which(sort(values) == max(passing))) / length(values)) * alpha
  }
}
accuracy_candidate <- bh_candidate(p_accuracy)
variance_candidate <- bh_candidate(p_variance)
reference_fdr <- min(
  old_adjusted_alpha,
  max(variance_candidate, variance_candidate, na.rm = TRUE),
  na.rm = TRUE
)
corrected_fdr <- min(
  old_adjusted_alpha,
  max(accuracy_candidate, variance_candidate, na.rm = TRUE),
  na.rm = TRUE
)
write_fixture("multiplicity", list(
  alpha = alpha,
  accuracy_tail_scores = p_accuracy,
  variance_tail_scores = p_variance,
  bh_adjusted_accuracy = stats::p.adjust(p_accuracy, method = "fdr"),
  bh_adjusted_variance = stats::p.adjust(p_variance, method = "fdr"),
  accuracy_candidate = accuracy_candidate,
  variance_candidate = variance_candidate,
  choir_v0_3_0_fdr_threshold = reference_fdr,
  corrected_fdr_threshold = corrected_fdr,
  bonferroni_threshold = alpha / length(p_accuracy)
))

# `.optimizeTree` creates reduction_parent but passes reduction. This captures
# the resulting recycled logical selection separately from the corrected subset.
full_reduction <- matrix(c(0, 1, 2, 3, 100, 200, 400, 800), ncol = 1L)
node_indices <- seq_len(4L)
child_labels <- paste0("c", seq_len(4L))
reference_distances <- .getCentroidDistance(
  reduction = full_reduction,
  clusters = child_labels
)
corrected_distances <- .getCentroidDistance(
  reduction = full_reduction[node_indices, , drop = FALSE],
  clusters = child_labels
)
write_fixture("subtree_distance_source", list(
  full_reduction = as.numeric(full_reduction),
  node_indices_1_based = node_indices,
  child_labels = child_labels,
  choir_v0_3_0_centroid_distances = unname(reference_distances),
  corrected_node_local_centroid_distances = unname(corrected_distances)
))

# Assertions bind fixtures to the exact source expressions that motivated them.
comparison_source <- paste(readLines(file.path(reference_r, "ComparisonUtils.R")), collapse = "\n")
prune_source <- paste(readLines(file.path(reference_r, "pruneTree.R")), collapse = "\n")
tree_source <- paste(readLines(file.path(reference_r, "TreeUtils.R")), collapse = "\n")
permutation_body <- paste(deparse(body(.runPermutationTest)), collapse = "\n")
assertions <- list(
  random_seed_unused_in_permutation_body = !grepl("random_seed", permutation_body, fixed = TRUE),
  repeated_errors_use_positions = grepl("which(RF1$predictions", comparison_source, fixed = TRUE),
  batch_metrics_use_flat_positions = grepl("accuracies[batch_inds]", comparison_source, fixed = TRUE),
  fdr_duplicates_variance_candidate = grepl(
    "max(p_variance_max, p_variance_max, na.rm = TRUE)",
    prune_source,
    fixed = TRUE
  ),
  subtree_optimizer_passes_full_reduction = grepl(
    ".getCentroidDistance(reduction = reduction,",
    tree_source,
    fixed = TRUE
  ),
  rounds_use_r_round = grepl(
    "seq_len(round(n_iterations/2))",
    comparison_source,
    fixed = TRUE
  )
)
if (!all(unlist(assertions))) {
  stop("One or more frozen source assertions failed")
}
write_fixture("source_assertions", assertions)

session <- list(
  r_version = R.version.string,
  platform = R.version$platform,
  reference_commit = "0db5e90ef46cd4f2f3765aa812325c8c768563fa",
  packages = as.list(vapply(required, function(package) {
    as.character(utils::packageVersion(package))
  }, character(1)))
)
write_fixture("session", session)
