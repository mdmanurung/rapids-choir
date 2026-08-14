#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(Matrix))

args <- commandArgs(trailingOnly = TRUE)
option_value <- function(name) {
  position <- which(args == name)
  if (length(position) != 1L || position == length(args)) {
    stop("Expected exactly one value for ", name)
  }
  args[[position + 1L]]
}

output <- option_value("--output")
extraction_manifest_path <- option_value("--extraction-manifest")
if (file.exists(output)) {
  stop("Immutable truth-label contract already exists: ", output)
}
extraction <- fromJSON(extraction_manifest_path, simplifyVector = FALSE)
if (!identical(extraction$status, "FROZEN") || !identical(extraction$mode, "development")) {
  stop("Development extraction manifest is not FROZEN")
}
extracted_by_id <- setNames(
  extraction$datasets,
  vapply(extraction$datasets, function(record) record$dataset_id, character(1L))
)
dataset_positions <- which(args == "--dataset")
if (length(dataset_positions) == 0L) {
  stop("At least one --dataset record is required")
}

records <- list()
candidate_sets <- list()
identifier_sets <- list()
for (position in dataset_positions) {
  fields <- strsplit(args[[position + 1L]], "\\t", fixed = FALSE)[[1L]]
  if (length(fields) != 5L) {
    stop(
      "Dataset records must be " ,
      "ID<TAB>groups<TAB>cells<TAB>counts<TAB>metadata"
    )
  }
  dataset_id <- fields[[1L]]
  expected_groups <- as.integer(fields[[2L]])
  expected_cells <- as.integer(fields[[3L]])
  counts_path <- fields[[4L]]
  metadata_path <- fields[[5L]]
  extracted <- extracted_by_id[[dataset_id]]
  if (is.null(extracted)) {
    stop("Dataset is absent from extraction manifest: ", dataset_id)
  }
  if (
    !identical(
      normalizePath(counts_path, mustWork = TRUE),
      normalizePath(extracted$files$counts$path, mustWork = TRUE)
    ) ||
      !identical(
        normalizePath(metadata_path, mustWork = TRUE),
        normalizePath(extracted$files$metadata$path, mustWork = TRUE)
      ) ||
      !identical(
        digest::digest(counts_path, algo = "sha256", file = TRUE, serialize = FALSE),
        extracted$files$counts$sha256
      ) ||
      !identical(
        digest::digest(metadata_path, algo = "sha256", file = TRUE, serialize = FALSE),
        extracted$files$metadata$sha256
      )
  ) {
    stop("Extracted input identity mismatch for ", dataset_id)
  }
  counts <- readRDS(counts_path)
  if (!(is.matrix(counts) || inherits(counts, "Matrix"))) {
    stop("Expected a matrix-like raw-count object for ", dataset_id)
  }
  if (length(dim(counts)) != 2L || ncol(counts) != expected_cells) {
    stop(
      "Count-matrix cell contract failed for ", dataset_id, ": expected ",
      expected_cells, ", got ", ncol(counts)
    )
  }
  if (
    is.null(rownames(counts)) || is.null(colnames(counts)) ||
      anyDuplicated(rownames(counts)) || anyDuplicated(colnames(counts)) ||
      any(!nzchar(rownames(counts))) || any(!nzchar(colnames(counts)))
  ) {
    stop("Count matrix requires unique, non-empty feature and cell identifiers")
  }
  count_values <- if (inherits(counts, "sparseMatrix")) counts@x else as.vector(counts)
  if (
    !is.numeric(count_values) || any(!is.finite(count_values)) ||
      any(count_values < 0) || any(count_values != floor(count_values))
  ) {
    stop("Counts must be finite, non-negative integers for ", dataset_id)
  }
  metadata <- read.csv(
    metadata_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (nrow(metadata) != expected_cells) {
    stop(
      "Cell-count contract failed for ", dataset_id, ": expected ",
      expected_cells, ", got ", nrow(metadata)
    )
  }
  candidates <- colnames(metadata)[vapply(metadata, function(values) {
    is.atomic(values) && !anyNA(values) && all(nzchar(as.character(values))) &&
      length(unique(as.character(values))) == expected_groups
  }, logical(1L))]
  identifiers <- colnames(metadata)[vapply(metadata, function(values) {
    characters <- as.character(values)
    is.atomic(values) && !anyNA(values) && all(nzchar(characters)) &&
      !anyDuplicated(characters) && setequal(characters, colnames(counts))
  }, logical(1L))]
  candidate_sets[[dataset_id]] <- candidates
  identifier_sets[[dataset_id]] <- identifiers
  records[[dataset_id]] <- list(
    counts_path = normalizePath(counts_path, mustWork = TRUE),
    counts_bytes = unname(file.info(counts_path)$size),
    counts_sha256 = digest::digest(
      counts_path,
      algo = "sha256",
      file = TRUE,
      serialize = FALSE
    ),
    metadata_path = normalizePath(metadata_path, mustWork = TRUE),
    metadata_bytes = unname(file.info(metadata_path)$size),
    metadata_sha256 = digest::digest(
      metadata_path,
      algo = "sha256",
      file = TRUE,
      serialize = FALSE
    ),
    expected_cells = expected_cells,
    observed_cells = nrow(metadata),
    observed_features = nrow(counts),
    expected_ground_truth_groups = expected_groups,
    candidate_fields = unname(candidates),
    cell_identifier_fields = unname(identifiers)
  )
}

common <- Reduce(intersect, candidate_sets)
if (length(common) != 1L) {
  stop(
    "Truth-label contract requires exactly one common metadata field; candidates: ",
    paste(common, collapse = ", ")
  )
}
common_identifiers <- Reduce(intersect, identifier_sets)
if (length(common_identifiers) != 1L) {
  stop(
    "Cell-identifier contract requires exactly one common metadata field; ",
    "candidates: ", paste(common_identifiers, collapse = ", ")
  )
}
if (identical(common[[1L]], common_identifiers[[1L]])) {
  stop("Truth-label and cell-identifier fields must be distinct")
}

payload <- list(
  schema_version = 2L,
  status = "PASS",
  candidate = extraction$candidate,
  candidate_manifest_sha256 = extraction$candidate_manifest_sha256,
  extraction_manifest = normalizePath(extraction_manifest_path, mustWork = TRUE),
  extraction_manifest_sha256 = digest::digest(
    extraction_manifest_path, algo = "sha256", file = TRUE, serialize = FALSE
  ),
  truth_column = unname(common[[1L]]),
  cell_id_column = unname(common_identifiers[[1L]]),
  datasets = records
)
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output, ".tmp")
write_json(payload, temporary, pretty = TRUE, auto_unbox = TRUE, null = "null")
if (!file.rename(temporary, output)) {
  stop("Could not atomically publish truth-label contract")
}
