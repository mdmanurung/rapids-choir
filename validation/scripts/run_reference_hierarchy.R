#!/usr/bin/env Rscript

# Execute frozen CHOIR v0.3.0 in fresh uninstrumented and instrumented
# environments. No reference source file is modified.

suppressPackageStartupMessages(library(jsonlite))

if (!identical(as.character(getRversion()), "4.5.1")) {
  stop("Reference hierarchy execution requires R 4.5.1; found ", getRversion())
}
if (!requireNamespace("digest", quietly = TRUE)) {
  stop("Reference hierarchy execution requires the digest package")
}

args <- commandArgs(trailingOnly = TRUE)
has_flag <- function(name) {
  sum(args == name) == 1L
}
option_value <- function(name) {
  position <- which(args == name)
  if (length(position) != 1L || position == length(args)) {
    stop("Expected exactly one value for ", name)
  }
  args[[position + 1L]]
}
empty_object <- function() {
  structure(list(), names = character())
}
sha256_file <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}
atomic_json <- function(path, value) {
  if (file.exists(path)) {
    stop("Immutable JSON artifact already exists: ", path)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp.", Sys.getpid())
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  writeLines(
    toJSON(
      value,
      auto_unbox = TRUE,
      null = "null",
      na = "null",
      digits = 17,
      pretty = TRUE
    ),
    temporary,
    useBytes = TRUE
  )
  if (!file.link(temporary, path)) {
    stop("Could not atomically publish JSON artifact: ", path)
  }
  unlink(temporary)
}
atomic_rds <- function(path, value) {
  if (file.exists(path)) {
    stop("Immutable RDS artifact already exists: ", path)
  }
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp.", Sys.getpid())
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  saveRDS(value, temporary, version = 3)
  if (!file.link(temporary, path)) {
    stop("Could not atomically publish RDS artifact: ", path)
  }
  unlink(temporary)
}

verify_reference_snapshot <- function(reference_root) {
  reference_root <- normalizePath(reference_root, mustWork = TRUE)
  manifest_path <- file.path(reference_root, "SOURCE_MANIFEST.sha256")
  lines <- readLines(manifest_path, warn = FALSE)
  if (length(lines) == 0L) {
    stop("Frozen reference source manifest is empty")
  }
  records <- lapply(lines, function(line) {
    fields <- strsplit(line, "[[:space:]]+", perl = TRUE)[[1L]]
    fields <- fields[nzchar(fields)]
    if (length(fields) != 2L || !grepl("^[0-9a-f]{64}$", fields[[1L]])) {
      stop("Frozen reference source manifest contains an invalid record")
    }
    marker <- "validation/reference/choir_v0_3_0/"
    if (!startsWith(fields[[2L]], marker)) {
      stop("Frozen reference manifest path escaped its snapshot")
    }
    relative <- substring(fields[[2L]], nchar(marker) + 1L)
    path <- file.path(reference_root, relative)
    if (!file.exists(path) || !identical(sha256_file(path), fields[[1L]])) {
      stop("Frozen CHOIR source checksum mismatch: ", relative)
    }
    list(path = relative, sha256 = fields[[1L]])
  })
  list(
    manifest_sha256 = sha256_file(manifest_path),
    file_count = length(records),
    files = records
  )
}

source_reference <- function(reference_root) {
  environment <- new.env(parent = globalenv())
  files <- sort(list.files(
    file.path(reference_root, "R"),
    pattern = "\\.R$",
    full.names = TRUE
  ))
  if (length(files) != 14L) {
    stop("Frozen CHOIR source inventory must contain exactly 14 R files")
  }
  for (path in files) {
    sys.source(path, envir = environment, keep.source = TRUE)
  }
  assign(
    "%>%",
    getExportedValue("magrittr", "%>%"),
    envir = environment
  )
  environment
}

install_instrumentation <- function(environment, entrypoint) {
  assign(".rapids_choir_boundary_log", list(), envir = environment)
  recorder <- function(name) {
    records <- get(".rapids_choir_boundary_log", envir = environment)
    records[[length(records) + 1L]] <- list(
      boundary = name,
      sequence = length(records)
    )
    assign(".rapids_choir_boundary_log", records, envir = environment)
    invisible(NULL)
  }
  assign(".rapids_choir_record_boundary", recorder, envir = environment)
  boundaries <- unique(c(
    entrypoint,
    "CHOIR",
    "buildTree",
    "pruneTree",
    "compareClusters",
    ".getStartingResolution",
    ".getTree.silhouette",
    ".getTree.full",
    ".getTree.subtree",
    ".runPermutationTest",
    ".relabelSingletons",
    ".optimizeTree",
    ".getNewLabels",
    ".checkHierarchy",
    ".checkClusterLabels"
  ))
  installed <- character()
  for (name in boundaries) {
    if (!exists(name, envir = environment, inherits = FALSE) ||
        !is.function(get(name, envir = environment, inherits = FALSE))) {
      next
    }
    tracer <- substitute(
      .rapids_choir_record_boundary(boundary_name),
      list(boundary_name = name)
    )
    suppressMessages(suppressWarnings(invisible(capture.output(
      trace(name, tracer = tracer, print = FALSE, where = environment)
    ))))
    installed <- c(installed, name)
  }
  installed
}

clean_comparison_records <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  result <- as.data.frame(value, stringsAsFactors = FALSE)
  timing <- grepl("(^|_)time($|_)", names(result), ignore.case = TRUE)
  result[, !timing, drop = FALSE]
}

extract_state <- function(result, recipe, environment) {
  adapter <- recipe$adapter
  if (identical(adapter, "partition")) {
    labels <- if (is.data.frame(result)) result[[1L]] else result
    labels <- as.integer(as.factor(as.character(labels)))
    if (length(labels) != length(recipe$cell_ids)) {
      stop("Partition adapter result does not align to recipe cell IDs")
    }
    tree <- data.frame(level = labels, row.names = recipe$cell_ids)
    return(list(
      cell_ids = as.character(recipe$cell_ids),
      labels = labels,
      cluster_tree = tree,
      comparison_records = NULL
    ))
  }
  if (!identical(adapter, "choir_object")) {
    stop("Unsupported reference hierarchy result adapter: ", adapter)
  }
  key <- if (is.null(recipe$key)) "CHOIR" else as.character(recipe$key)
  alpha <- if (is.null(recipe$alpha)) 0.05 else as.numeric(recipe$alpha)
  cell_ids <- get(".getCellIDs", envir = environment)(result)
  cluster_tree <- get(".retrieveData", envir = environment)(
    result, key, "clusters", "full_tree"
  )
  labels <- get(".retrieveData", envir = environment)(
    result, key, "final_clusters", paste0("CHOIR_clusters_", alpha)
  )
  comparison_records <- get(".retrieveData", envir = environment)(
    result, key, "records", "comparison_records"
  )
  list(
    cell_ids = as.character(cell_ids),
    labels = as.integer(labels),
    cluster_tree = as.data.frame(cluster_tree, stringsAsFactors = FALSE),
    comparison_records = clean_comparison_records(comparison_records)
  )
}

state_digest <- function(state) {
  payload <- toJSON(
    state,
    dataframe = "rows",
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = 17,
    pretty = FALSE
  )
  digest::digest(charToRaw(payload), algo = "sha256", serialize = FALSE)
}

run_reference <- function(recipe, reference_root, seed, instrument = FALSE) {
  environment <- source_reference(reference_root)
  entrypoint <- as.character(recipe$entrypoint)
  if (!exists(entrypoint, envir = environment, inherits = FALSE)) {
    stop("Reference recipe entrypoint does not exist: ", entrypoint)
  }
  installed <- character()
  if (instrument) {
    installed <- install_instrumentation(environment, entrypoint)
  }
  arguments <- recipe$arguments
  if (!is.list(arguments)) {
    stop("Reference recipe arguments must be a list")
  }
  target <- get(entrypoint, envir = environment, inherits = FALSE)
  if ("random_seed" %in% names(formals(target))) {
    arguments$random_seed <- as.integer(seed)
  }
  set.seed(as.integer(seed))
  result <- do.call(target, arguments, envir = environment)
  state <- extract_state(result, recipe, environment)
  boundaries <- if (instrument) {
    get(".rapids_choir_boundary_log", envir = environment)
  } else {
    list()
  }
  list(
    state = state,
    digest = state_digest(state),
    boundaries = boundaries,
    installed_boundaries = installed
  )
}

membership_sha256 <- function(indices) {
  values <- sort(unique(as.integer(indices)))
  if (length(values) == 0L || any(values < 0L)) {
    stop("Hierarchy membership must contain non-negative indices")
  }
  payload <- c(
    charToRaw("rapids-choir/membership/v1"),
    as.raw(0L),
    charToRaw(paste(values, collapse = ","))
  )
  digest::digest(payload, algo = "sha256", serialize = FALSE)
}

hex_to_decimal <- function(value) {
  digits <- 0L
  for (character in strsplit(tolower(value), "", fixed = TRUE)[[1L]]) {
    hexadecimal <- match(character, c(as.character(0:9), letters[1:6])) - 1L
    carry <- hexadecimal
    for (index in seq_along(digits)) {
      current <- digits[[index]] * 16L + carry
      digits[[index]] <- current %% 10L
      carry <- current %/% 10L
    }
    while (carry > 0L) {
      digits <- c(digits, carry %% 10L)
      carry <- carry %/% 10L
    }
  }
  paste(rev(digits), collapse = "")
}

canonical_tree <- function(cluster_tree, cell_ids) {
  raw <- as.data.frame(cluster_tree, stringsAsFactors = FALSE)
  if (nrow(raw) != length(cell_ids) || ncol(raw) == 0L) {
    stop("Reference cluster tree does not align to cell IDs")
  }
  n_cells <- length(cell_ids)
  root_hash <- membership_sha256(seq_len(n_cells) - 1L)
  root_id <- hex_to_decimal(root_hash)
  nodes <- list(list(
    node_id = root_id,
    label = "P0_L1_1",
    level = 1L,
    ordinal = 1L,
    size = as.integer(n_cells),
    membership_sha256 = root_hash,
    parent_id = NULL,
    child_ids = character()
  ))
  node_positions <- setNames(1L, root_id)
  previous <- rep(root_id, n_cells)
  assignments <- list(previous)
  resolutions <- list(NULL)
  for (source_index in seq_len(ncol(raw))) {
    values <- as.character(raw[[source_index]])
    pair_keys <- paste(previous, values, sep = "\u001f")
    proposed <- previous
    changed <- FALSE
    pending <- list()
    level <- length(assignments) + 1L
    ordinal <- 0L
    for (parent_id in unique(previous)) {
      parent_members <- which(previous == parent_id)
      groups <- unique(pair_keys[parent_members])
      if (length(groups) <= 1L) {
        next
      }
      changed <- TRUE
      for (group in groups) {
        members_one_based <- which(pair_keys == group)
        members <- members_one_based - 1L
        membership <- membership_sha256(members)
        node_id <- hex_to_decimal(membership)
        ordinal <- ordinal + 1L
        position <- unname(node_positions[node_id])
        if (is.na(position)) {
          node_positions[[node_id]] <- length(nodes) + 1L
          nodes[[length(nodes) + 1L]] <- list(
            node_id = node_id,
            label = paste0("P0_L", level, "_", ordinal),
            level = as.integer(level),
            ordinal = as.integer(ordinal),
            size = as.integer(length(members)),
            membership_sha256 = membership,
            parent_id = parent_id,
            child_ids = character()
          )
        }
        proposed[members_one_based] <- node_id
        pending[[parent_id]] <- unique(c(pending[[parent_id]], node_id))
      }
    }
    if (!changed) {
      next
    }
    for (parent_id in names(pending)) {
      position <- unname(node_positions[parent_id])
      if (is.na(position)) {
        stop("Hierarchy parent position is missing: ", parent_id)
      }
      nodes[[position]]$child_ids <- as.character(pending[[parent_id]])
    }
    previous <- proposed
    assignments[[length(assignments) + 1L]] <- previous
    resolutions[[length(resolutions) + 1L]] <- NULL
  }
  list(
    schema = "rapids_choir.hierarchy/v1",
    cell_ids = as.character(cell_ids),
    assignments = lapply(assignments, as.character),
    nodes = nodes,
    resolutions = resolutions,
    root_id = root_id
  )
}

event <- function(
  event_type,
  phase,
  parameters = empty_object(),
  seed = NULL,
  input_membership_hashes = character(),
  output_membership_hashes = character(),
  topology_context = empty_object(),
  decision = NULL,
  reason = NULL,
  bounded_values = empty_object(),
  payload = empty_object()
) {
  list(
    event_type = event_type,
    phase = phase,
    parameters = parameters,
    seed = seed,
    input_membership_hashes = as.character(input_membership_hashes),
    output_membership_hashes = as.character(output_membership_hashes),
    topology_context = topology_context,
    decision = decision,
    reason = reason,
    bounded_values = bounded_values,
    payload = payload
  )
}

boundary_event <- function(record, seed) {
  mapping <- list(
    CHOIR = c("prepared_input_validation", "preprocessing"),
    buildTree = c("hierarchy_stop", "hierarchy"),
    pruneTree = c("prune", "pruning"),
    compareClusters = c("comparator_call", "pruning"),
    .getStartingResolution = c("resolution_search", "hierarchy"),
    .getTree.silhouette = c("hierarchy_stop", "hierarchy"),
    .getTree.full = c("hierarchy_stop", "hierarchy"),
    .getTree.subtree = c("hierarchy_stop", "hierarchy"),
    .runPermutationTest = c("comparator_call", "pruning"),
    .relabelSingletons = c("singleton", "hierarchy"),
    .optimizeTree = c("resolution_transition", "hierarchy"),
    .getNewLabels = c("merge", "pruning"),
    .checkHierarchy = c("prepared_input_validation", "preprocessing"),
    .checkClusterLabels = c("final_relabel", "finalization")
  )
  kind <- mapping[[record$boundary]]
  if (is.null(kind)) {
    kind <- c("provenance", "initialization")
  }
  event(
    kind[[1L]],
    kind[[2L]],
    parameters = list(
      boundary = record$boundary,
      boundary_sequence = as.integer(record$sequence)
    ),
    seed = as.integer(seed),
    decision = "boundary_entered"
  )
}

comparison_events <- function(records, seed) {
  if (is.null(records) || nrow(records) == 0L) {
    return(list())
  }
  bounded_fields <- c(
    "mean_accuracy",
    "var_accuracy",
    "mean_permuted_accuracy",
    "var_permuted_accuracy",
    "percentile_accuracy",
    "percentile_variance",
    "mean_modified_accuracy",
    "var_modified_accuracy",
    "percentile_modified_accuracy",
    "percentile_modified_variance"
  )
  lapply(seq_len(nrow(records)), function(index) {
    row <- records[index, , drop = FALSE]
    bounded <- empty_object()
    for (name in intersect(bounded_fields, names(row))) {
      value <- suppressWarnings(as.numeric(row[[name]][[1L]]))
      if (length(value) == 1L && is.finite(value) && value >= 0 && value <= 1) {
        bounded[[name]] <- value
      }
    }
    decision_text <- if ("decision" %in% names(row)) {
      as.character(row$decision[[1L]])
    } else {
      NA_character_
    }
    decision <- if (is.na(decision_text)) NULL else {
      if (startsWith(decision_text, "split")) "split" else "merge"
    }
    reason <- if (is.na(decision_text)) NULL else {
      gsub("[^a-z0-9]+", "_", tolower(decision_text))
    }
    comparison <- if ("comparison" %in% names(row)) {
      as.character(row$comparison[[1L]])
    } else {
      paste0("comparison-", index)
    }
    event(
      "comparator_call",
      "pruning",
      parameters = list(comparison = comparison, pair_kind = "recorded_reference"),
      seed = as.integer(seed),
      decision = decision,
      reason = reason,
      bounded_values = bounded
    )
  })
}

make_bundle <- function(recipe, context, run, seed, snapshot) {
  state <- run$state
  tree <- canonical_tree(state$cluster_tree, state$cell_ids)
  context_seed <- as.integer(context$provenance$random_state)
  if (identical(as.integer(seed), context_seed)) {
    run_id <- context$run_id
    provenance <- context$provenance
  } else {
    digest <- digest::digest(
      charToRaw(paste0(context$run_id, "\n", seed)),
      algo = "sha256",
      serialize = FALSE
    )
    run_id <- paste0("run-", substring(digest, 1L, 32L))
    provenance <- context$provenance
    provenance$run_id <- run_id
    provenance$random_state <- as.integer(seed)
    provenance$configuration$random_state <- as.integer(seed)
  }
  provenance$reference_source_manifest_sha256 <- snapshot$manifest_sha256
  provenance$instrumentation_equivalence_sha256 <- run$digest
  events <- list(event(
    "provenance",
    "initialization",
    parameters = list(
      reference = "CHOIR v0.3.0",
      source_manifest_sha256 = snapshot$manifest_sha256
    ),
    seed = as.integer(seed),
    decision = "validated"
  ))
  events <- c(
    events,
    lapply(run$boundaries, boundary_event, seed = seed),
    comparison_events(state$comparison_records, seed)
  )
  for (node in tree$nodes) {
    events[[length(events) + 1L]] <- event(
      "hierarchy_node",
      "hierarchy",
      parameters = list(
        node_id = node$node_id,
        label = node$label,
        size = node$size
      ),
      output_membership_hashes = node$membership_sha256,
      topology_context = list(
        level = node$level,
        parent_id = node$parent_id,
        child_ids = as.character(node$child_ids)
      ),
      decision = "create"
    )
  }
  events[[length(events) + 1L]] <- event(
    "final_relabel",
    "finalization",
    parameters = list(
      ordering = "reference_record_order",
      cluster_count = length(unique(state$labels))
    ),
    decision = "relabel"
  )
  list(
    schema_version = 1L,
    dataset_id = as.character(recipe$dataset_id),
    run_id = run_id,
    cell_ids = as.character(state$cell_ids),
    labels = as.integer(state$labels),
    tree = tree,
    provenance = provenance,
    events = events
  )
}

execute_driver <- function(recipe, reference_root, seed) {
  baseline <- run_reference(recipe, reference_root, seed, instrument = FALSE)
  instrumented <- run_reference(recipe, reference_root, seed, instrument = TRUE)
  equivalent <- identical(baseline$state, instrumented$state) &&
    identical(baseline$digest, instrumented$digest)
  if (!equivalent) {
    stop("CHOIR instrumentation changed the uninstrumented R result")
  }
  list(
    baseline = baseline,
    instrumented = instrumented,
    report = list(
      schema_version = 1L,
      status = "PASS",
      reference = "CHOIR v0.3.0",
      r_version = as.character(getRversion()),
      entrypoint = recipe$entrypoint,
      adapter = recipe$adapter,
      seed = as.integer(seed),
      uninstrumented_state_sha256 = baseline$digest,
      instrumented_state_sha256 = instrumented$digest,
      results_identical = TRUE,
      installed_boundaries = instrumented$installed_boundaries,
      observed_boundaries = vapply(
        instrumented$boundaries,
        function(record) record$boundary,
        character(1)
      )
    )
  )
}

reference_root <- option_value("--reference-root")
snapshot <- verify_reference_snapshot(reference_root)

if (has_flag("--self-test")) {
  recipe <- list(
    schema_version = 1L,
    dataset_id = "reference-instrumentation-self-test",
    entrypoint = ".relabelSingletons",
    adapter = "partition",
    cell_ids = paste0("cell-", 1:4),
    arguments = list(cluster_df = data.frame(
      cluster = c("A", "singleton", "singleton", "B")
    ))
  )
  run <- execute_driver(recipe, reference_root, 17L)
  if (!(".relabelSingletons" %in% run$report$observed_boundaries)) {
    stop("Reference instrumentation self-test did not observe its real boundary")
  }
  run$report$source_manifest_sha256 <- snapshot$manifest_sha256
  cat(toJSON(run$report, auto_unbox = TRUE, pretty = TRUE), "\n")
  quit(status = 0L)
}

recipe_path <- option_value("--recipe")
python_result_path <- option_value("--python-result")
bundle_path <- option_value("--bundle")
equivalence_path <- option_value("--equivalence")
seed <- as.integer(option_value("--seed"))
if (is.na(seed)) {
  stop("--seed must be an integer")
}
recipe <- readRDS(recipe_path)
required_recipe <- c(
  "schema_version", "dataset_id", "entrypoint", "adapter", "cell_ids", "arguments"
)
if (!is.list(recipe) || !identical(recipe$schema_version, 1L) ||
    !setequal(names(recipe), required_recipe) ||
    !is.character(recipe$dataset_id) || length(recipe$dataset_id) != 1L ||
    !is.character(recipe$cell_ids) || anyDuplicated(recipe$cell_ids) != 0L) {
  stop("Reference hierarchy recipe has an unsupported schema")
}
context <- fromJSON(
  python_result_path,
  simplifyVector = FALSE,
  simplifyDataFrame = FALSE
)
if (!is.list(context) ||
    !identical(context$schema_version, "rapids_choir.hierarchy-result/v1") ||
    !identical(context$dataset_id, recipe$dataset_id) ||
    !identical(as.character(unlist(context$cell_ids)), recipe$cell_ids) ||
    !grepl("^run-[0-9a-f]{32}$", context$run_id)) {
  stop("Python hierarchy result is not a valid run context for the R reference")
}
run <- execute_driver(recipe, reference_root, seed)
run$report$dataset_id <- recipe$dataset_id
run$report$source_manifest_sha256 <- snapshot$manifest_sha256
run$report$python_result_sha256 <- sha256_file(python_result_path)
bundle <- make_bundle(recipe, context, run$instrumented, seed, snapshot)
atomic_rds(bundle_path, bundle)
run$report$bundle_sha256 <- sha256_file(bundle_path)
atomic_json(equivalence_path, run$report)
