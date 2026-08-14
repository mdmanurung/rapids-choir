#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(anndataR))
suppressPackageStartupMessages(library(Seurat))
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

counts_input <- option_value("--counts")
metadata_input <- option_value("--metadata")
output <- option_value("--output")
truth_contract_path <- option_value("--truth-contract")
dataset_id <- option_value("--dataset-id")
expected_cells <- as.integer(option_value("--expected-cells"))
expected_groups <- as.integer(option_value("--expected-groups"))
if (file.exists(output)) {
  stop("Immutable H5AD output already exists: ", output)
}

truth_contract <- fromJSON(truth_contract_path, simplifyVector = TRUE)
truth_contract_raw <- fromJSON(truth_contract_path, simplifyVector = FALSE)
if (!identical(truth_contract$status, "PASS")) {
  stop("Truth-label contract is not PASS")
}
truth_column <- truth_contract$truth_column
cell_id_column <- truth_contract$cell_id_column
counts <- readRDS(counts_input)
if (!(is.matrix(counts) || inherits(counts, "Matrix"))) {
  stop("Input is not a matrix-like raw-count object: ", counts_input)
}
if (length(dim(counts)) != 2L || ncol(counts) != expected_cells) {
  stop("Expected ", expected_cells, " cells, got ", ncol(counts))
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
  stop("Counts must be finite, non-negative integers")
}
metadata <- read.csv(
  metadata_input,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
if (nrow(metadata) != expected_cells) {
  stop("Expected ", expected_cells, " metadata rows, got ", nrow(metadata))
}
if (!cell_id_column %in% colnames(metadata)) {
  stop("Frozen cell identifier field is absent: ", cell_id_column)
}
cell_ids <- as.character(metadata[[cell_id_column]])
if (
  anyNA(cell_ids) || any(!nzchar(cell_ids)) || anyDuplicated(cell_ids) ||
    !setequal(cell_ids, colnames(counts))
) {
  stop("Metadata cell identifiers do not match count-matrix columns")
}
rownames(metadata) <- cell_ids
metadata <- metadata[colnames(counts), , drop = FALSE]
if (!identical(rownames(metadata), colnames(counts))) {
  stop("Could not align metadata to count-matrix columns")
}
if (!truth_column %in% colnames(metadata)) {
  stop("Frozen truth field is absent: ", truth_column)
}
truth <- as.character(metadata[[truth_column]])
if (anyNA(truth) || length(unique(truth)) != expected_groups) {
  stop("Truth-label cardinality contract failed for ", dataset_id)
}
dataset_contract <- truth_contract_raw$datasets[[dataset_id]]
if (is.null(dataset_contract)) {
  stop("Dataset is absent from truth-label contract: ", dataset_id)
}
if (
  !identical(
    digest::digest(counts_input, algo = "sha256", file = TRUE, serialize = FALSE),
    dataset_contract$counts_sha256
  ) ||
    !identical(
      digest::digest(metadata_input, algo = "sha256", file = TRUE, serialize = FALSE),
      dataset_contract$metadata_sha256
    )
) {
  stop("Preprocessing inputs differ from the truth-label contract")
}

object <- CreateSeuratObject(
  counts = counts,
  assay = "RNA",
  project = dataset_id,
  meta.data = metadata
)
DefaultAssay(object) <- "RNA"
assay <- DefaultAssay(object)
counts <- LayerData(object, assay = assay, layer = "counts")
if (nrow(counts) < 2000L) {
  stop("At least 2,000 features are required by the frozen preprocessing contract")
}

set.seed(8675309L)
object <- NormalizeData(
  object,
  assay = assay,
  normalization.method = "LogNormalize",
  scale.factor = 10000,
  verbose = FALSE
)
object <- FindVariableFeatures(
  object,
  assay = assay,
  selection.method = "vst",
  nfeatures = 2000L,
  verbose = FALSE
)
hvg <- VariableFeatures(object, assay = assay)
if (length(hvg) != 2000L) {
  stop("Seurat did not return exactly 2,000 variable features")
}
object <- ScaleData(object, assay = assay, features = hvg, verbose = FALSE)
object <- RunPCA(
  object,
  assay = assay,
  features = hvg,
  npcs = 50L,
  seed.use = 8675309L,
  verbose = FALSE
)
if (ncol(Embeddings(object, reduction = "pca")) != 50L) {
  stop("Seurat did not produce exactly 50 principal components")
}
object <- FindNeighbors(
  object,
  reduction = "pca",
  dims = seq_len(50L),
  k.param = 15L,
  graph.name = c("choir_nn", "choir_snn"),
  verbose = FALSE
)

cells <- colnames(object)
features <- rownames(object[[assay]])
normalized <- LayerData(object, assay = assay, layer = "data")[features, cells, drop = FALSE]
counts <- LayerData(object, assay = assay, layer = "counts")[features, cells, drop = FALSE]
pca <- Embeddings(object, reduction = "pca")[cells, seq_len(50L), drop = FALSE]
nn_graph <- as(object@graphs[["choir_nn"]][cells, cells], "dgCMatrix")

obs <- data.frame(
  truth_label = truth,
  row.names = cells,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
var <- data.frame(
  highly_variable = features %in% hvg,
  row.names = features,
  check.names = FALSE
)
adata <- AnnData(
  X = Matrix::t(normalized),
  obs = obs,
  var = var,
  layers = list(counts = Matrix::t(counts)),
  obsm = list(X_pca = unname(pca)),
  obsp = list(choir_nn = nn_graph),
  uns = list(
    dataset_id = dataset_id,
    candidate_commit = truth_contract_raw$candidate$commit,
    candidate_attempt = truth_contract_raw$candidate$attempt,
    runtime_tree_sha256 = truth_contract_raw$candidate$runtime_tree_sha256,
    truth_contract_sha256 = digest::digest(
      truth_contract_path, algo = "sha256", file = TRUE, serialize = FALSE
    ),
    truth_source_column = truth_column,
    cell_id_source_column = cell_id_column,
    preprocessing_contract = "LogNormalize:10000|vst:2000|PCA:50|NN:15"
  )
)

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output, ".tmp.h5ad")
if (file.exists(temporary)) {
  stop("Temporary H5AD already exists: ", temporary)
}
adata$write_h5ad(temporary)
if (!file.rename(temporary, output)) {
  stop("Could not atomically publish H5AD output")
}
