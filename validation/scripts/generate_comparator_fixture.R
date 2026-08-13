#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("Usage: generate_comparator_fixture.R <ComparisonUtils.R> <output> <seed>")
}
reference_source <- args[[1L]]
output <- args[[2L]]
seed <- as.integer(args[[3L]])
source(reference_source, local = globalenv())

n_cells <- 80L
n_features <- 20L
features <- outer(seq_len(n_cells), seq_len(n_features), function(cell, feature) {
  ((cell * 17L + feature * 13L) %% 11L) / 100
})
features[41:80, ] <- features[41:80, ] + 3
rownames(features) <- paste0("cell", seq_len(n_cells))
colnames(features) <- paste0("gene", seq_len(n_features))
features <- methods::as(Matrix(features, sparse = TRUE), "dgCMatrix")
graph <- methods::as(Matrix(1, nrow = n_cells, ncol = n_cells, sparse = TRUE), "dgCMatrix")
rownames(graph) <- rownames(features)
colnames(graph) <- rownames(features)

pairs <- list(
  list(
    pair_id = "separable_cross",
    cluster1 = rownames(features)[1:40],
    cluster2 = rownames(features)[41:80]
  ),
  list(
    pair_id = "same_population_control",
    cluster1 = rownames(features)[1:20],
    cluster2 = rownames(features)[21:40]
  )
)
record_columns <- c(
  "comparison", "cluster1_size", "cluster2_size", "sample_size",
  "mean_accuracy", "var_accuracy", "mean_errors",
  "mean_permuted_accuracy", "var_permuted_accuracy",
  "percentile_accuracy", "percentile_variance",
  "n_repeat_errors1", "n_repeat_errors2",
  "mean_repeat_errors1", "mean_repeat_errors2",
  "connectivity", "time", "decision"
)
set.seed(seed)
results <- lapply(pairs, function(pair) {
  empty_records <- data.frame(matrix(ncol = length(record_columns), nrow = 0L))
  colnames(empty_records) <- record_columns
  result <- .runPermutationTest(
    cluster1_name = pair$pair_id,
    cluster1_cells = pair$cluster1,
    cluster2_name = paste0(pair$pair_id, "_other"),
    cluster2_cells = pair$cluster2,
    alpha = 0.05,
    n_iterations = 20L,
    n_trees = 20L,
    use_variance = TRUE,
    min_accuracy = 0.5,
    min_connections = 1,
    max_repeat_errors = 0L,
    collect_all_metrics = FALSE,
    sample_max = Inf,
    downsampling_rate = 1,
    min_reads = NULL,
    input_matrix = features,
    nn_matrix = graph,
    comparison_records = empty_records,
    feature_importance_records = NULL,
    comparison_start_time = NULL,
    n_cores = 1L,
    random_seed = seed
  )
  record <- result$comparison_records[1L, , drop = FALSE]
  list(
    pair_id = pair$pair_id,
    cluster1_indices_1_based = match(pair$cluster1, rownames(features)),
    cluster2_indices_1_based = match(pair$cluster2, rownames(features)),
    decision = result$result,
    reason = as.character(record$decision[[1L]]),
    mean_observed_accuracy = as.numeric(record$mean_accuracy[[1L]]),
    accuracy_reference_tail_probability = as.numeric(record$percentile_accuracy[[1L]]),
    variance_reference_tail_probability = as.numeric(record$percentile_variance[[1L]])
  )
})

payload <- list(
  schema_version = 1L,
  source = "tracked CHOIR v0.3.0 ComparisonUtils.R",
  seed = seed,
  n_iterations = 20L,
  n_trees = 20L,
  feature_shape = dim(features),
  features_row_major = as.vector(t(as.matrix(features))),
  graph = "complete_directed",
  results = results,
  environment = list(
    r_version = R.version.string,
    ranger = as.character(packageVersion("ranger"))
  )
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output, ".tmp")
write_json(
  payload,
  temporary,
  pretty = TRUE,
  auto_unbox = TRUE,
  null = "null",
  na = "null",
  digits = NA
)
if (!file.rename(temporary, output)) {
  stop("Could not atomically publish synthetic comparator fixture")
}
