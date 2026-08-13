#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(SeuratObject))

args <- commandArgs(trailingOnly = TRUE)
option_value <- function(name) {
  position <- which(args == name)
  if (length(position) != 1L || position == length(args)) {
    stop("Expected exactly one value for ", name)
  }
  args[[position + 1L]]
}

output <- option_value("--output")
dataset_positions <- which(args == "--dataset")
if (length(dataset_positions) == 0L) {
  stop("At least one --dataset record is required")
}

records <- list()
candidate_sets <- list()
for (position in dataset_positions) {
  fields <- strsplit(args[[position + 1L]], "\\t", fixed = FALSE)[[1L]]
  if (length(fields) != 4L) {
    stop("Dataset records must be ID<TAB>groups<TAB>cells<TAB>path")
  }
  dataset_id <- fields[[1L]]
  expected_groups <- as.integer(fields[[2L]])
  expected_cells <- as.integer(fields[[3L]])
  path <- fields[[4L]]
  object <- readRDS(path)
  if (!inherits(object, "Seurat")) {
    stop("Expected a Seurat object for ", dataset_id)
  }
  metadata <- object[[]]
  if (nrow(metadata) != expected_cells) {
    stop(
      "Cell-count contract failed for ", dataset_id, ": expected ",
      expected_cells, ", got ", nrow(metadata)
    )
  }
  candidates <- colnames(metadata)[vapply(metadata, function(values) {
    is.atomic(values) && !anyNA(values) &&
      length(unique(as.character(values))) == expected_groups
  }, logical(1L))]
  candidate_sets[[dataset_id]] <- candidates
  records[[dataset_id]] <- list(
    path = normalizePath(path, mustWork = TRUE),
    bytes = unname(file.info(path)$size),
    sha256 = digest::digest(path, algo = "sha256", file = TRUE, serialize = FALSE),
    expected_cells = expected_cells,
    observed_cells = nrow(metadata),
    expected_ground_truth_groups = expected_groups,
    candidate_fields = unname(candidates)
  )
}

common <- Reduce(intersect, candidate_sets)
if (length(common) != 1L) {
  stop(
    "Truth-label contract requires exactly one common metadata field; candidates: ",
    paste(common, collapse = ", ")
  )
}

payload <- list(
  schema_version = 1L,
  status = "PASS",
  truth_column = unname(common[[1L]]),
  datasets = records
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output, ".tmp")
write_json(payload, temporary, pretty = TRUE, auto_unbox = TRUE, null = "null")
if (!file.rename(temporary, output)) {
  stop("Could not atomically publish truth-label contract")
}
