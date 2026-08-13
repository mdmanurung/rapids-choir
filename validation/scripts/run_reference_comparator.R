#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(anndataR))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(jsonlite))

args <- commandArgs(trailingOnly = TRUE)
option_value <- function(name) {
  position <- which(args == name)
  if (length(position) != 1L || position == length(args)) {
    stop("Expected exactly one value for ", name)
  }
  args[[position + 1L]]
}

h5ad_path <- option_value("--h5ad")
pair_path <- option_value("--pairs")
dataset_id <- option_value("--dataset-id")
reference_source <- option_value("--reference-source")
output <- option_value("--output")

source(reference_source, local = globalenv())
adata <- read_h5ad(h5ad_path)
pair_manifest <- fromJSON(pair_path, simplifyVector = FALSE)

features <- which(as.logical(adata$var$highly_variable))
input_matrix <- adata$X[, features, drop = FALSE]
input_matrix <- methods::as(input_matrix, "dgCMatrix")
rownames(input_matrix) <- adata$obs_names
colnames(input_matrix) <- adata$var_names[features]
nn_matrix <- methods::as(adata$obsp[["choir_nn"]], "dgCMatrix")
rownames(nn_matrix) <- adata$obs_names
colnames(nn_matrix) <- adata$obs_names

record_columns <- c(
  "comparison", "cluster1_size", "cluster2_size", "sample_size",
  "mean_accuracy", "var_accuracy", "mean_errors",
  "mean_permuted_accuracy", "var_permuted_accuracy",
  "percentile_accuracy", "percentile_variance",
  "n_repeat_errors1", "n_repeat_errors2",
  "mean_repeat_errors1", "mean_repeat_errors2",
  "mean_modified_accuracy", "var_modified_accuracy",
  "percentile_modified_accuracy", "percentile_modified_variance",
  "connectivity", "time", "decision"
)

set.seed(1L)
results <- vector("list", length(pair_manifest$pairs))
for (index in seq_along(pair_manifest$pairs)) {
  pair <- pair_manifest$pairs[[index]]
  empty_records <- data.frame(matrix(ncol = length(record_columns), nrow = 0L))
  colnames(empty_records) <- record_columns
  started <- proc.time()[["elapsed"]]
  comparison <- .runPermutationTest(
    cluster1_name = pair$cluster1_label,
    cluster1_cells = unlist(pair$cluster1_cells, use.names = FALSE),
    cluster2_name = pair$cluster2_label,
    cluster2_cells = unlist(pair$cluster2_cells, use.names = FALSE),
    alpha = 0.05,
    n_iterations = 100L,
    n_trees = 50L,
    use_variance = TRUE,
    min_accuracy = 0.5,
    min_connections = 1,
    max_repeat_errors = 20L,
    collect_all_metrics = FALSE,
    sample_max = Inf,
    downsampling_rate = 1,
    min_reads = NULL,
    input_matrix = input_matrix,
    nn_matrix = nn_matrix,
    comparison_records = empty_records,
    feature_importance_records = NULL,
    comparison_start_time = NULL,
    n_cores = 1L,
    random_seed = 1L
  )
  record <- comparison$comparison_records[1L, , drop = FALSE]
  results[[index]] <- list(
    pair_id = pair$pair_id,
    pair_kind = pair$kind,
    expected_decision = pair$expected_decision,
    decision = comparison$result,
    reason = as.character(record$decision[[1L]]),
    mean_observed_accuracy = as.numeric(record$mean_accuracy[[1L]]),
    variance_observed_accuracy = as.numeric(record$var_accuracy[[1L]]),
    mean_null_accuracy = as.numeric(record$mean_permuted_accuracy[[1L]]),
    variance_null_accuracy = as.numeric(record$var_permuted_accuracy[[1L]]),
    accuracy_reference_tail_probability = as.numeric(record$percentile_accuracy[[1L]]),
    variance_reference_tail_probability = as.numeric(record$percentile_variance[[1L]]),
    runtime_seconds = proc.time()[["elapsed"]] - started
  )
}

payload <- list(
  schema_version = 1L,
  status = "PASS",
  dataset_id = dataset_id,
  seed = 1L,
  n_cores = 1L,
  h5ad_sha256 = digest::digest(
    h5ad_path, algo = "sha256", file = TRUE, serialize = FALSE
  ),
  pair_manifest_sha256 = digest::digest(
    pair_path, algo = "sha256", file = TRUE, serialize = FALSE
  ),
  reference_source = normalizePath(reference_source, mustWork = TRUE),
  reference_source_sha256 = digest::digest(
    reference_source, algo = "sha256", file = TRUE, serialize = FALSE
  ),
  environment = list(
    r_version = R.version.string,
    ranger = as.character(packageVersion("ranger")),
    Matrix = as.character(packageVersion("Matrix")),
    anndataR = as.character(packageVersion("anndataR"))
  ),
  results = results
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
  stop("Could not atomically publish R comparator evidence")
}
