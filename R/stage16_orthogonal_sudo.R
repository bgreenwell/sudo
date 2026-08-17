# Stage 16: orthogonal binary SUDO under GAMSEL regularization.
#
# Point confirmation uses two primary cells and two non-gating boundaries.
# If every primary point gate passes, the script automatically runs a
# checkpointed full-pipeline bootstrap on the primary cells.
#
# Run from the repository root after the full stage-16t tuning run:
#   Rscript R/stage16_orthogonal_sudo.R
#
# Smoke test:
#   SUDO_STAGE16_SMOKE=1 Rscript R/stage16_orthogonal_sudo.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16_SMOKE", unset = ""))
n_reps <- stage16_env_int("SUDO_STAGE16_REPS", if (smoke) 2L else 100L)
coverage_reps <- stage16_env_int(
  "SUDO_STAGE16_COVERAGE_REPS", if (smoke) 2L else 50L
)
bootstrap_reps <- stage16_env_int(
  "SUDO_STAGE16_BOOTSTRAPS", if (smoke) 3L else 99L
)
bootstrap_fit_timeout <- stage16_env_int(
  "SUDO_STAGE16_BOOTSTRAP_TIMEOUT", if (smoke) 60L else 600L
)
outer_folds <- stage16_env_int(
  "SUDO_STAGE16_OUTER_FOLDS", if (smoke) 2L else 5L
)
inner_folds <- stage16_env_int(
  "SUDO_STAGE16_INNER_FOLDS", if (smoke) 4L else 5L
)
cores <- stage16_env_int("SUDO_STAGE16_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int("SUDO_STAGE16_CHUNK_SIZE", if (smoke) 1L else 5L)
force_coverage <- nzchar(Sys.getenv(
  "SUDO_STAGE16_FORCE_COVERAGE", unset = ""
))

clip_env <- Sys.getenv("SUDO_STAGE16_CLIP", unset = "")
selection_file <- "R/results/stage16t_orthogonal_sudo_selection.csv"
smoke_selection_file <-
  "R/results/stage16t_orthogonal_sudo_selection_smoke.csv"
if (nzchar(clip_env)) {
  clip <- as.numeric(clip_env)
} else if (file.exists(selection_file)) {
  tuning <- read.csv(selection_file)
  clip <- tuning$clip[tuning$selected][1]
} else if (smoke && file.exists(smoke_selection_file)) {
  tuning <- read.csv(smoke_selection_file)
  clip <- tuning$clip[tuning$selected][1]
} else if (smoke) {
  clip <- 1e-6
} else {
  stop("run the full stage-16t clipping selection before confirmation")
}
stopifnot(length(clip) == 1L, is.finite(clip), clip > 0, clip < 0.5)

cells <- stage16_cells(smoke)
cell_filter <- Sys.getenv("SUDO_STAGE16_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]], , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L)
stopifnot(n_reps >= 2L, coverage_reps >= 2L, bootstrap_reps >= 2L,
          bootstrap_fit_timeout >= 1L,
          outer_folds >= 2L, inner_folds >= 4L, cores >= 1L,
          chunk_size >= 1L)

expected_cells <- stage16_cells(FALSE)$cell
full_point_run <- !smoke && n_reps >= 100L && outer_folds >= 5L &&
  inner_folds >= 5L && identical(as.character(cells$cell), expected_cells)
point_suffix <- if (full_point_run) "" else "_smoke"
point_checkpoint_file <- paste0(
  "R/results/stage16_orthogonal_sudo_point_checkpoint", point_suffix, ".csv"
)
point_checkpoint <- if (file.exists(point_checkpoint_file)) {
  read.csv(point_checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
point_checkpoint_keys <- c(
  "clip", "cell_n", "cell_p", "outer_folds", "inner_folds"
)
if (nrow(point_checkpoint) &&
    !all(point_checkpoint_keys %in% names(point_checkpoint))) {
  point_checkpoint <- data.frame()
}
if (nrow(point_checkpoint)) {
  cell_match <- match(point_checkpoint$cell, cells$cell)
  point_checkpoint <- point_checkpoint[
    !is.na(cell_match) &
      point_checkpoint$cell_n == cells$n[cell_match] &
      point_checkpoint$cell_p == cells$p[cell_match] &
      point_checkpoint$clip == clip &
      point_checkpoint$outer_folds == outer_folds &
      point_checkpoint$inner_folds == inner_folds, , drop = FALSE
  ]
}
cluster <- if (cores > 1L) stage16_cluster(cores) else NULL

for (cell_index in seq_len(nrow(cells))) {
  cell <- cells[cell_index, ]
  checkpoint_keys <- c(
    "clip", "cell_n", "cell_p", "outer_folds", "inner_folds"
  )
  completed <- if (nrow(point_checkpoint) &&
                   all(checkpoint_keys %in% names(point_checkpoint))) {
    unique(point_checkpoint$replication[
      point_checkpoint$cell == cell$cell &
        point_checkpoint$clip == clip &
        point_checkpoint$cell_n == cell$n &
        point_checkpoint$cell_p == cell$p &
        point_checkpoint$outer_folds == outer_folds &
        point_checkpoint$inner_folds == inner_folds &
        point_checkpoint$replication <= n_reps
    ])
  } else {
    integer()
  }
  remaining <- setdiff(seq_len(n_reps), completed)
  chunks <- split(remaining, ceiling(seq_along(remaining) / chunk_size))
  for (chunk in chunks) {
    worker <- function(replication) {
      seed <- 1603000L + cell_index * 10000L + replication
      result <- try(stage16_one_rep(
        seed, cell, clip, outer_folds, inner_folds,
        include_one_se = TRUE
      ), silent = TRUE)
      if (inherits(result, "try-error")) {
        data.frame(
          seed = seed, selection = c("min", "one_se"), truth = 0.75,
          direct = NA_real_, raw_label = NA_real_,
          index_projection = NA_real_, sudo = NA_real_,
          scaled = NA_real_, orthogonal = NA_real_,
          targeted_projection = NA_real_, targeted_sudo = NA_real_,
          correction = NA_real_, kappa = NA_real_, clipping_rate = NA_real_,
          max_inverse_information = NA_real_, target_failure = 1,
          direct_contrast_spread = NA_real_,
          index_rmse = NA_real_, index_correlation = NA_real_,
          oracle_sudo = NA_real_, oracle_orthogonal = NA_real_,
          sample_signature = NA_real_, fold_signature = NA_real_,
          error = as.character(result), stringsAsFactors = FALSE
        )
      } else {
        result$error <- ""
        result
      }
    }
    if (!is.null(cluster)) {
      parallel::clusterExport(
        cluster,
        c("cell", "clip", "cell_index", "outer_folds", "inner_folds",
          "worker"), envir = environment()
      )
    }
    added <- do.call(rbind, if (is.null(cluster)) {
      lapply(chunk, worker)
    } else {
      parallel::parLapplyLB(cluster, chunk, worker)
    })
    added$cell <- cell$cell
    added$primary <- cell$primary
    added$clip <- clip
    added$cell_n <- cell$n
    added$cell_p <- cell$p
    added$outer_folds <- outer_folds
    added$inner_folds <- inner_folds
    added$replication <- rep(chunk, each = 2L)
    point_checkpoint <- rbind(point_checkpoint, added)
    dir.create("R/results", showWarnings = FALSE)
    write.csv(point_checkpoint, point_checkpoint_file, row.names = FALSE)
    cat(sprintf(
      "point %-16s checkpointed %d/%d\n", cell$cell,
      length(unique(point_checkpoint$replication[
        point_checkpoint$cell == cell$cell &
          point_checkpoint$cell_n == cell$n &
          point_checkpoint$cell_p == cell$p &
          point_checkpoint$clip == clip &
          point_checkpoint$outer_folds == outer_folds &
          point_checkpoint$inner_folds == inner_folds
      ])), n_reps
    ))
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)

point_replications <- point_checkpoint[
  point_checkpoint$cell %in% cells$cell &
    point_checkpoint$clip == clip &
    point_checkpoint$cell_n == cells$n[
      match(point_checkpoint$cell, cells$cell)
    ] &
    point_checkpoint$cell_p == cells$p[
      match(point_checkpoint$cell, cells$cell)
    ] &
    point_checkpoint$outer_folds == outer_folds &
    point_checkpoint$inner_folds == inner_folds &
    point_checkpoint$replication <= n_reps, , drop = FALSE
]
point_replications <- point_replications[order(
  match(point_replications$cell, cells$cell),
  point_replications$replication, point_replications$selection
), ]
point_summary <- stage16_summarize_point(point_replications)
point_gates <- stage16_point_gates(point_summary)
point_summary$bias_ok <- point_gates$bias_ok
print(point_summary, row.names = FALSE)

oracle_rows <- point_summary[
  point_summary$primary & point_summary$selection == "min" &
    point_summary$estimator %in% c("oracle_sudo", "oracle_orthogonal"),
  , drop = FALSE
]
oracle_pass <- all(abs(oracle_rows$bias) <= 3 * oracle_rows$mc_se_bias)
contrast_pass <- all(
  is.finite(point_replications$direct_contrast_spread) &
    point_replications$direct_contrast_spread < 1e-8
)
point_pass <- point_gates$primary_pass & point_gates$improvement_pass &
  point_gates$informativeness_pass & oracle_pass & contrast_pass

write.csv(
  point_summary,
  paste0("R/results/stage16_orthogonal_sudo_point", point_suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  point_replications,
  paste0("R/results/stage16_orthogonal_sudo_point_replications", point_suffix,
         ".csv"),
  row.names = FALSE
)

if (full_point_run && !point_pass && !force_coverage) {
  stop(
    "stage 16 failed a primary point gate; diagnostics were retained and ",
    "coverage was not promoted"
  )
}
if (full_point_run && !point_pass && force_coverage) {
  cat("DIAGNOSTIC OVERRIDE: point gates failed; coverage will not promote ",
      "the method\n", sep = "")
}
if (point_pass) {
  cat("PASS: primary orthogonal-SUDO point gates and oracle harness pass\n")
} else if (full_point_run) {
  cat("POINT DIAGNOSTIC: primary promotion gates were not met\n")
} else {
  cat("SMOKE DIAGNOSTIC: statistical promotion requires the full point run\n")
}

run_coverage <- force_coverage || (full_point_run && point_pass)
if (!run_coverage) {
  cat("Coverage not run: enable SUDO_STAGE16_FORCE_COVERAGE for diagnostics, ",
      "or complete a passing full point stage\n", sep = "")
  quit(save = "no", status = 0L)
}

coverage_cells <- cells[cells$primary, , drop = FALSE]
full_coverage_run <- full_point_run && coverage_reps >= 50L &&
  bootstrap_reps >= 99L && nrow(coverage_cells) == 2L
coverage_suffix <- if (full_coverage_run) "" else "_smoke"
coverage_checkpoint_file <- paste0(
  "R/results/stage16_orthogonal_sudo_coverage_checkpoint", coverage_suffix,
  ".csv"
)
coverage_checkpoint <- if (file.exists(coverage_checkpoint_file)) {
  read.csv(coverage_checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
coverage_checkpoint_keys <- c(
  "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "bootstrap_reps"
)
if (nrow(coverage_checkpoint) &&
    !all(coverage_checkpoint_keys %in% names(coverage_checkpoint))) {
  coverage_checkpoint <- data.frame()
}
if (nrow(coverage_checkpoint) &&
    !"bootstrap_fit_timeout" %in% names(coverage_checkpoint)) {
  coverage_checkpoint$bootstrap_fit_timeout <- Inf
}
if (nrow(coverage_checkpoint)) {
  cell_match <- match(coverage_checkpoint$cell, coverage_cells$cell)
  coverage_checkpoint <- coverage_checkpoint[
    !is.na(cell_match) &
      coverage_checkpoint$cell_n == coverage_cells$n[cell_match] &
      coverage_checkpoint$cell_p == coverage_cells$p[cell_match] &
      coverage_checkpoint$clip == clip &
      coverage_checkpoint$outer_folds == outer_folds &
      coverage_checkpoint$inner_folds == inner_folds &
      coverage_checkpoint$bootstrap_reps == bootstrap_reps, , drop = FALSE
  ]
}
cluster <- if (cores > 1L) stage16_cluster(cores) else NULL
for (cell_index in seq_len(nrow(coverage_cells))) {
  cell <- coverage_cells[cell_index, ]
  coverage_keys <- c(
    "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
    "bootstrap_reps"
  )
  completed <- if (nrow(coverage_checkpoint) &&
                   all(coverage_keys %in% names(coverage_checkpoint))) {
    unique(coverage_checkpoint$replication[
      coverage_checkpoint$cell == cell$cell &
        coverage_checkpoint$clip == clip &
        coverage_checkpoint$cell_n == cell$n &
        coverage_checkpoint$cell_p == cell$p &
        coverage_checkpoint$outer_folds == outer_folds &
        coverage_checkpoint$inner_folds == inner_folds &
        coverage_checkpoint$bootstrap_reps == bootstrap_reps &
        coverage_checkpoint$replication <= coverage_reps
    ])
  } else {
    integer()
  }
  remaining <- setdiff(seq_len(coverage_reps), completed)
  chunks <- split(remaining, ceiling(seq_along(remaining) / chunk_size))
  for (chunk in chunks) {
    worker <- function(replication) {
      seed <- 1604000L + cell_index * 10000L + replication
      stage16_bootstrap_rep(
        seed, cell, clip, outer_folds, inner_folds, bootstrap_reps,
        bootstrap_fit_timeout
      )
    }
    if (!is.null(cluster)) {
      parallel::clusterExport(
        cluster,
        c("cell", "clip", "cell_index", "outer_folds", "inner_folds",
          "bootstrap_reps", "bootstrap_fit_timeout", "worker"),
        envir = environment()
      )
    }
    added <- do.call(rbind, if (is.null(cluster)) {
      lapply(chunk, worker)
    } else {
      parallel::parLapplyLB(cluster, chunk, worker)
    })
    added$cell <- cell$cell
    added$primary <- cell$primary
    added$clip <- clip
    added$cell_n <- cell$n
    added$cell_p <- cell$p
    added$outer_folds <- outer_folds
    added$inner_folds <- inner_folds
    added$bootstrap_reps <- bootstrap_reps
    added$bootstrap_fit_timeout <- bootstrap_fit_timeout
    added$replication <- rep(chunk, each = length(unique(added$estimator)))
    coverage_checkpoint <- rbind(coverage_checkpoint, added)
    write.csv(coverage_checkpoint, coverage_checkpoint_file, row.names = FALSE)
    cat(sprintf(
      "coverage %-13s checkpointed %d/%d\n", cell$cell,
      length(unique(coverage_checkpoint$replication[
        coverage_checkpoint$cell == cell$cell &
          coverage_checkpoint$cell_n == cell$n &
          coverage_checkpoint$cell_p == cell$p &
          coverage_checkpoint$clip == clip &
          coverage_checkpoint$outer_folds == outer_folds &
          coverage_checkpoint$inner_folds == inner_folds &
          coverage_checkpoint$bootstrap_reps == bootstrap_reps
      ])), coverage_reps
    ))
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)

coverage_replications <- coverage_checkpoint[
  coverage_checkpoint$cell %in% coverage_cells$cell &
    coverage_checkpoint$clip == clip &
    coverage_checkpoint$cell_n == coverage_cells$n[
      match(coverage_checkpoint$cell, coverage_cells$cell)
    ] &
    coverage_checkpoint$cell_p == coverage_cells$p[
      match(coverage_checkpoint$cell, coverage_cells$cell)
    ] &
    coverage_checkpoint$outer_folds == outer_folds &
    coverage_checkpoint$inner_folds == inner_folds &
    coverage_checkpoint$bootstrap_reps == bootstrap_reps &
    coverage_checkpoint$replication <= coverage_reps, , drop = FALSE
]
coverage_summary <- stage16_summarize_coverage(coverage_replications)
coverage_gates <- stage16_coverage_gates(coverage_summary)
coverage_summary$bias_ok <- coverage_gates$bias_ok
coverage_summary$coverage_ok <- coverage_gates$coverage_ok
coverage_summary$sd_to_se_ok <- coverage_gates$ratio_ok
print(coverage_summary, row.names = FALSE)
write.csv(
  coverage_summary,
  paste0("R/results/stage16_orthogonal_sudo_coverage", coverage_suffix,
         ".csv"),
  row.names = FALSE
)
write.csv(
  coverage_replications,
  paste0("R/results/stage16_orthogonal_sudo_coverage_replications",
         coverage_suffix, ".csv"),
  row.names = FALSE
)

if (full_coverage_run && !coverage_gates$pass) {
  stop(
    "stage 16 failed a primary coverage gate; results were retained and no ",
    "Python or manuscript promotion is authorized"
  )
}
if (coverage_gates$pass && point_pass) {
  cat("PASS: orthogonal SUDO satisfies every primary stage-16 gate\n")
} else {
  cat("DIAGNOSTIC COVERAGE COMPLETE: full promotion gates were not met\n")
}
