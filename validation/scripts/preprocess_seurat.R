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

input <- option_value("--input")
output <- option_value("--output")
truth_contract_path <- option_value("--truth-contract")
dataset_id <- option_value("--dataset-id")
expected_cells <- as.integer(option_value("--expected-cells"))
expected_groups <- as.integer(option_value("--expected-groups"))

truth_contract <- fromJSON(truth_contract_path, simplifyVector = TRUE)
truth_column <- truth_contract$truth_column
object <- readRDS(input)
if (!inherits(object, "Seurat")) {
  stop("Input is not a Seurat object: ", input)
}
if (ncol(object) != expected_cells) {
  stop("Expected ", expected_cells, " cells, got ", ncol(object))
}
if (!truth_column %in% colnames(object[[]])) {
  stop("Frozen truth field is absent: ", truth_column)
}
truth <- as.character(object[[truth_column, drop = TRUE]])
if (anyNA(truth) || length(unique(truth)) != expected_groups) {
  stop("Truth-label cardinality contract failed for ", dataset_id)
}

if ("RNA" %in% Assays(object)) {
  DefaultAssay(object) <- "RNA"
}
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
    truth_source_column = truth_column,
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
