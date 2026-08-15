#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(anndataR))

if (!identical(as.character(getRversion()), "4.5.1")) {
  stop("Hierarchy development objects require R 4.5.1; found ", getRversion())
}
for (package in c(
  "digest", "jsonlite", "splatter", "SingleCellExperiment",
  "SummarizedExperiment", "Matrix"
)) {
  if (!requireNamespace(package, quietly = TRUE)) {
    stop("Hierarchy development objects require R package: ", package)
  }
}

args <- commandArgs(trailingOnly = TRUE)
option_value <- function(name) {
  position <- which(args == name)
  if (length(position) != 1L || position == length(args)) {
    stop("Expected exactly one value for ", name)
  }
  args[[position + 1L]]
}
contract_path <- option_value("--contract")
dataset_id <- option_value("--dataset-id")
recipe_path <- option_value("--output-recipe")
h5ad_path <- option_value("--output-h5ad")
manifest_path <- option_value("--output-manifest")
if (any(file.exists(c(recipe_path, h5ad_path, manifest_path)))) {
  stop("Immutable hierarchy development output already exists")
}

contract <- jsonlite::fromJSON(
  contract_path,
  simplifyVector = FALSE,
  simplifyDataFrame = FALSE
)
if (!is.list(contract) || !identical(contract$schema_version, 1L) ||
    !identical(contract$generator, "splatter::splatSimulate") ||
    !identical(contract$evidence_role, "development_only") ||
    !identical(contract$promotable, FALSE)) {
  stop("Hierarchy development object contract is invalid")
}
matches <- Filter(
  function(record) identical(record$dataset_id, dataset_id),
  contract$objects
)
if (length(matches) != 1L) {
  stop("Dataset ID must select exactly one hierarchy development role")
}
record <- matches[[1L]]
if (sum(vapply(
  contract$objects,
  function(value) identical(value$role, record$role),
  logical(1)
)) != 1L) {
  stop("Hierarchy development role is not exclusive")
}

setting <- function(name) {
  if (!is.null(record[[name]])) record[[name]] else contract$defaults[[name]]
}
simulation <- splatter::splatSimulate(
  nGenes = as.integer(setting("n_genes")),
  batchCells = as.integer(unlist(setting("batch_cells"), use.names = FALSE)),
  group.prob = as.numeric(unlist(setting("group_prob"), use.names = FALSE)),
  de.prob = as.numeric(setting("de_prob")),
  de.facLoc = as.numeric(setting("de_fac_loc")),
  de.facScale = as.numeric(setting("de_fac_scale")),
  method = "groups",
  seed = as.integer(record$generator_seed),
  verbose = FALSE
)
counts <- SummarizedExperiment::assay(simulation, "counts")
library_sizes <- Matrix::colSums(counts)
if (any(!is.finite(library_sizes)) || any(library_sizes <= 0)) {
  stop("Hierarchy development simulation contains an empty library")
}
normalized <- Matrix::t(Matrix::t(counts) / library_sizes * 10000)
SummarizedExperiment::assay(simulation, "logcounts") <- log1p(normalized)
cell_ids <- paste0(dataset_id, "-cell-", seq_len(ncol(simulation)))
gene_ids <- paste0(dataset_id, "-gene-", seq_len(nrow(simulation)))
colnames(simulation) <- cell_ids
rownames(simulation) <- gene_ids
S4Vectors::colData(simulation)$rapids_choir_development_role <- record$role
S4Vectors::metadata(simulation)$rapids_choir_development <- list(
  schema_version = 1L,
  dataset_id = dataset_id,
  role = record$role,
  evidence_role = "development_only",
  promotable = FALSE,
  generator_seed = as.integer(record$generator_seed)
)

recipe <- list(
  schema_version = 1L,
  dataset_id = dataset_id,
  entrypoint = "CHOIR",
  adapter = "choir_object",
  cell_ids = cell_ids,
  arguments = list(
    object = simulation,
    key = "CHOIR",
    alpha = 0.05,
    p_adjust = if (identical(setting("multiplicity"), "bh")) {
      "fdr"
    } else {
      setting("multiplicity")
    },
    n_iterations = 100L,
    n_trees = 50L,
    min_cluster_depth = as.integer(setting("min_cluster_depth")),
    max_clusters = if (is.null(setting("max_clusters"))) {
      "auto"
    } else {
      as.integer(setting("max_clusters"))
    },
    cluster_params = list(
      algorithm = 1L,
      group.singletons = isTRUE(setting("group_singletons"))
    ),
    normalization_method = "none",
    n_cores = 1L,
    random_seed = as.integer(contract$reference_seeds[[1L]]),
    verbose = FALSE
  )
)

dir.create(dirname(recipe_path), recursive = TRUE, showWarnings = FALSE)
dir.create(dirname(h5ad_path), recursive = TRUE, showWarnings = FALSE)
recipe_temporary <- paste0(recipe_path, ".tmp.", Sys.getpid())
h5ad_temporary <- paste0(h5ad_path, ".tmp.", Sys.getpid(), ".h5ad")
on.exit({
  if (file.exists(recipe_temporary)) unlink(recipe_temporary)
  if (file.exists(h5ad_temporary)) unlink(h5ad_temporary)
}, add = TRUE)
saveRDS(recipe, recipe_temporary, version = 3)
adata <- anndataR::as_AnnData(
  simulation,
  x_mapping = "counts",
  layers_mapping = c(counts = "counts"),
  obs_mapping = TRUE,
  var_mapping = TRUE,
  uns_mapping = FALSE
)
adata$write_h5ad(h5ad_temporary)
recipe_published <- file.link(recipe_temporary, recipe_path)
h5ad_published <- FALSE
if (recipe_published) {
  h5ad_published <- file.link(h5ad_temporary, h5ad_path)
}
if (!recipe_published || !h5ad_published) {
  if (recipe_published && file.exists(recipe_path)) unlink(recipe_path)
  if (h5ad_published && file.exists(h5ad_path)) unlink(h5ad_path)
  stop("Could not atomically publish hierarchy development object")
}
unlink(recipe_temporary)
unlink(h5ad_temporary)

sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}
manifest <- list(
  schema_version = 1L,
  status = "GENERATED",
  dataset_id = dataset_id,
  role = record$role,
  required_event = record$required_event,
  evidence_role = "development_only",
  promotable = FALSE,
  role_exclusive = TRUE,
  generator = "splatter::splatSimulate",
  generator_seed = as.integer(record$generator_seed),
  reference_seeds = as.integer(unlist(
    contract$reference_seeds,
    use.names = FALSE
  )),
  r_version = as.character(getRversion()),
  packages = list(
    splatter = as.character(utils::packageVersion("splatter")),
    anndataR = as.character(utils::packageVersion("anndataR"))
  ),
  cell_count = as.integer(ncol(simulation)),
  gene_count = as.integer(nrow(simulation)),
  contract_sha256 = sha256_file(contract_path),
  artifacts = list(
    recipe = list(
      path = normalizePath(recipe_path, mustWork = TRUE),
      bytes = unname(file.info(recipe_path)$size),
      sha256 = sha256_file(recipe_path)
    ),
    h5ad = list(
      path = normalizePath(h5ad_path, mustWork = TRUE),
      bytes = unname(file.info(h5ad_path)$size),
      sha256 = sha256_file(h5ad_path)
    )
  )
)
dir.create(dirname(manifest_path), recursive = TRUE, showWarnings = FALSE)
manifest_temporary <- paste0(manifest_path, ".tmp.", Sys.getpid())
on.exit(if (file.exists(manifest_temporary)) unlink(manifest_temporary), add = TRUE)
writeLines(
  jsonlite::toJSON(
    manifest,
    auto_unbox = TRUE,
    null = "null",
    digits = 17,
    pretty = TRUE
  ),
  manifest_temporary,
  useBytes = TRUE
)
if (!file.link(manifest_temporary, manifest_path)) {
  unlink(recipe_path)
  unlink(h5ad_path)
  stop("Could not atomically publish hierarchy development manifest")
}
unlink(manifest_temporary)
