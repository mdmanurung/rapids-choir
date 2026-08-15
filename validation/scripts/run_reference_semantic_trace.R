#!/usr/bin/env Rscript

# Canonicalize an instrumented CHOIR v0.3.0 boundary bundle. The bundle is
# produced by the reference driver and must already contain semantic events;
# this script never infers or fills missing algorithm state.

suppressPackageStartupMessages(library(jsonlite))

if (!identical(as.character(getRversion()), "4.5.1")) {
  stop("Reference traces require R 4.5.1; found ", getRversion())
}

args <- commandArgs(trailingOnly = TRUE)
option_value <- function(name) {
  position <- which(args == name)
  if (length(position) != 1L || position == length(args)) {
    stop("Expected exactly one value for ", name)
  }
  args[[position + 1L]]
}

bundle_path <- option_value("--bundle")
candidate_manifest_path <- option_value("--candidate-manifest")
output_path <- option_value("--output")
result_path <- option_value("--result")
if (file.exists(output_path)) {
  stop("Immutable reference semantic trace already exists: ", output_path)
}
if (file.exists(result_path)) {
  stop("Immutable reference hierarchy result already exists: ", result_path)
}

bundle <- readRDS(bundle_path)
candidate_manifest <- fromJSON(
  candidate_manifest_path,
  simplifyVector = FALSE,
  simplifyDataFrame = FALSE
)
if (!is.list(candidate_manifest) ||
    !identical(candidate_manifest$status, "FROZEN") ||
    !grepl("^[0-9a-f]{40}$", candidate_manifest$candidate_commit) ||
    !grepl("^[0-9a-f]{64}$", candidate_manifest$runtime_tree_sha256) ||
    length(candidate_manifest$candidate_attempt) != 1L ||
    as.integer(candidate_manifest$candidate_attempt) < 1L) {
  stop("Candidate manifest is not a canonical frozen candidate")
}
bundle_fields <- c(
  "schema_version", "dataset_id", "run_id", "cell_ids", "labels",
  "tree", "provenance", "events"
)
if (!is.list(bundle) || !identical(bundle$schema_version, 1L) ||
    !setequal(names(bundle), bundle_fields)) {
  stop("Reference semantic boundary bundle has an unsupported schema")
}
if (!is.character(bundle$dataset_id) || length(bundle$dataset_id) != 1L ||
    nchar(bundle$dataset_id) == 0L) {
  stop("Reference semantic boundary bundle has no dataset ID")
}
if (!is.character(bundle$run_id) || length(bundle$run_id) != 1L ||
    nchar(bundle$run_id) == 0L) {
  stop("Reference semantic boundary bundle has no run ID")
}
if (!is.atomic(bundle$cell_ids) || length(bundle$cell_ids) == 0L ||
    anyDuplicated(as.character(bundle$cell_ids)) != 0L ||
    !is.atomic(bundle$labels) ||
    length(bundle$labels) != length(bundle$cell_ids)) {
  stop("Reference semantic boundary cell identities or labels are invalid")
}
if (!is.list(bundle$events)) {
  stop("Reference semantic boundary bundle events must be a list")
}
if (length(bundle$events) == 0L) {
  stop("Reference semantic boundary bundle events cannot be empty")
}

event_types <- c(
  "provenance", "preprocessing", "prepared_input_validation",
  "resolution_search", "resolution_transition", "hierarchy_node",
  "hierarchy_stop", "comparator_call", "multiplicity", "prune",
  "bridge", "merge", "singleton", "final_relabel"
)
required <- c(
  "event_type", "phase", "parameters", "seed",
  "input_membership_hashes", "output_membership_hashes",
  "topology_context", "decision", "reason", "bounded_values", "payload"
)

canonical_object <- function(value, field) {
  if (!is.list(value) || is.data.frame(value)) {
    stop(field, " must be an object")
  }
  if (length(value) == 0L) {
    return(structure(list(), names = character()))
  }
  if (is.null(names(value)) || any(names(value) == "")) {
    stop(field, " must have named object fields")
  }
  value
}

canonical_event <- function(event, index) {
  if (!is.list(event) || !setequal(names(event), required)) {
    stop("Event ", index - 1L, " has incomplete or unknown fields")
  }
  if (!(event$event_type %in% event_types)) {
    stop("Event ", index - 1L, " has an unsupported event_type")
  }
  if (!is.character(event$phase) || length(event$phase) != 1L ||
      nchar(event$phase) == 0L) {
    stop("Event ", index - 1L, " has an invalid phase")
  }
  membership_hashes <- c(
    unlist(event$input_membership_hashes, use.names = FALSE),
    unlist(event$output_membership_hashes, use.names = FALSE)
  )
  if (length(membership_hashes) > 0L &&
      any(!grepl("^[0-9a-f]{64}$", membership_hashes))) {
    stop("Event ", index - 1L, " has an invalid membership SHA-256")
  }
  bounded <- unlist(event$bounded_values, recursive = TRUE, use.names = TRUE)
  if (length(bounded) > 0L &&
      (any(!is.finite(bounded)) || any(bounded < 0) || any(bounded > 1))) {
    stop("Event ", index - 1L, " has a bounded value outside [0, 1]")
  }
  list(
    schema_version = "rapids_choir.semantic/v1",
    run_id = bundle$run_id,
    operation_index = index - 1L,
    event_type = event$event_type,
    phase = event$phase,
    parameters = canonical_object(event$parameters, "event parameters"),
    seed = event$seed,
    input_membership_hashes = I(as.character(
      unlist(event$input_membership_hashes, use.names = FALSE)
    )),
    output_membership_hashes = I(as.character(
      unlist(event$output_membership_hashes, use.names = FALSE)
    )),
    topology_context = canonical_object(
      event$topology_context, "event topology_context"
    ),
    decision = event$decision,
    reason = event$reason,
    bounded_values = canonical_object(
      event$bounded_values, "event bounded_values"
    ),
    payload = canonical_object(event$payload, "event payload")
  )
}

events <- lapply(seq_along(bundle$events), function(index) {
  canonical_event(bundle$events[[index]], index)
})

canonical_tree <- function(tree, cell_ids) {
  required_tree <- c(
    "schema", "cell_ids", "assignments", "nodes", "resolutions", "root_id"
  )
  if (!is.list(tree) || !setequal(names(tree), required_tree) ||
      !identical(tree$schema, "rapids_choir.hierarchy/v1") ||
      !identical(as.character(tree$cell_ids), as.character(cell_ids)) ||
      !is.list(tree$assignments) || length(tree$assignments) == 0L ||
      !is.list(tree$nodes) || length(tree$nodes) == 0L ||
      length(tree$resolutions) != length(tree$assignments)) {
    stop("Reference hierarchy tree is not canonical")
  }
  assignments <- lapply(tree$assignments, function(assignment) {
    values <- as.character(unlist(assignment, use.names = FALSE))
    if (length(values) != length(cell_ids) ||
        any(!grepl("^[0-9]+$", values))) {
      stop("Reference hierarchy assignment is invalid")
    }
    I(values)
  })
  node_fields <- c(
    "node_id", "label", "level", "ordinal", "size",
    "membership_sha256", "parent_id", "child_ids"
  )
  nodes <- lapply(tree$nodes, function(node) {
    if (!is.list(node) || !setequal(names(node), node_fields)) {
      stop("Reference hierarchy node is not canonical")
    }
    node_id <- as.character(node$node_id)
    parent_id <- if (is.null(node$parent_id)) NULL else as.character(node$parent_id)
    child_ids <- as.character(unlist(node$child_ids, use.names = FALSE))
    if (length(node_id) != 1L || !grepl("^[0-9]+$", node_id) ||
        (!is.null(parent_id) &&
         (length(parent_id) != 1L || !grepl("^[0-9]+$", parent_id))) ||
        (length(child_ids) > 0L && any(!grepl("^[0-9]+$", child_ids))) ||
        !grepl("^[0-9a-f]{64}$", node$membership_sha256)) {
      stop("Reference hierarchy node identity is invalid")
    }
    list(
      node_id = node_id,
      label = as.character(node$label),
      level = as.integer(node$level),
      ordinal = as.integer(node$ordinal),
      size = as.integer(node$size),
      membership_sha256 = as.character(node$membership_sha256),
      parent_id = parent_id,
      child_ids = I(child_ids)
    )
  })
  resolutions <- vapply(tree$resolutions, function(value) {
    if (is.null(value)) NA_real_ else as.numeric(value)
  }, numeric(1))
  root_id <- as.character(tree$root_id)
  if (length(root_id) != 1L || !grepl("^[0-9]+$", root_id)) {
    stop("Reference hierarchy root identity is invalid")
  }
  list(
    schema = "rapids_choir.hierarchy/v1",
    cell_ids = I(as.character(cell_ids)),
    assignments = assignments,
    nodes = nodes,
    resolutions = I(resolutions),
    root_id = root_id
  )
}

tree <- canonical_tree(bundle$tree, bundle$cell_ids)
required_provenance <- c(
  "schema", "run_id", "input_sha256", "input_contract", "configuration",
  "rules", "multiplicity", "random_state"
)
if (!is.list(bundle$provenance) ||
    any(!required_provenance %in% names(bundle$provenance)) ||
    !identical(bundle$provenance$schema, "rapids_choir.result/v1") ||
    !identical(bundle$provenance$run_id, bundle$run_id)) {
  stop("Reference hierarchy provenance is not canonical")
}
provenance <- bundle$provenance
for (field in intersect(
  c("input_components", "input_contract", "configuration"), names(provenance)
)) {
  provenance[[field]] <- canonical_object(
    provenance[[field]], paste0("provenance ", field)
  )
}

sha256_file <- function(path) {
  command <- system2(
    "sha256sum", shQuote(normalizePath(path, mustWork = TRUE)),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(command, "status")
  if (!is.null(status) && status != 0L) {
    stop("Could not calculate SHA-256 for ", path, ": ", paste(command, collapse = "\n"))
  }
  digest <- strsplit(command[[1L]], "[[:space:]]+")[[1L]][[1L]]
  if (!grepl("^[0-9a-f]{64}$", digest)) {
    stop("sha256sum returned an invalid digest for ", path)
  }
  digest
}

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
temporary <- paste0(output_path, ".tmp.", Sys.getpid())
connection <- file(temporary, open = "wb")
on.exit({
  if (!is.null(connection)) try(close(connection), silent = TRUE)
  if (file.exists(temporary)) unlink(temporary)
}, add = TRUE)
for (event in events) {
  line <- toJSON(
    event,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = 17,
    pretty = FALSE
  )
  writeBin(charToRaw(paste0(line, "\n")), connection)
}
close(connection)
connection <- NULL
if (!file.link(temporary, output_path)) {
  stop("Could not atomically publish the reference semantic trace")
}
unlink(temporary)

result <- list(
  schema_version = "rapids_choir.hierarchy-result/v1",
  candidate = list(
    commit = candidate_manifest$candidate_commit,
    attempt = as.integer(candidate_manifest$candidate_attempt),
    runtime_tree_sha256 = candidate_manifest$runtime_tree_sha256,
    manifest_sha256 = sha256_file(candidate_manifest_path)
  ),
  dataset_id = bundle$dataset_id,
  run_id = bundle$run_id,
  cell_ids = I(as.character(bundle$cell_ids)),
  labels = I(unname(bundle$labels)),
  tree = tree,
  provenance = provenance,
  semantic_trace = list(
    schema_version = "rapids_choir.semantic-artifact/v1",
    format = "jsonl",
    path = normalizePath(output_path, mustWork = TRUE),
    sha256 = sha256_file(output_path),
    row_count = length(events),
    run_id = bundle$run_id,
    event_schema = "rapids_choir.semantic/v1"
  )
)
dir.create(dirname(result_path), recursive = TRUE, showWarnings = FALSE)
result_temporary <- paste0(result_path, ".tmp.", Sys.getpid())
on.exit({
  if (file.exists(result_temporary)) unlink(result_temporary)
}, add = TRUE)
writeLines(
  toJSON(
    result,
    auto_unbox = TRUE,
    null = "null",
    na = "null",
    digits = 17,
    pretty = FALSE
  ),
  result_temporary,
  useBytes = TRUE
)
if (file.exists(result_path) || !file.link(result_temporary, result_path)) {
  stop("Could not atomically publish the reference hierarchy result")
}
unlink(result_temporary)
