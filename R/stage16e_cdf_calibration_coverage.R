# Stage 16e: full-pipeline coverage for affine PIT calibration.
#
# Stage 16d fixes the affine calibration rule on fresh confirmation seeds.
# This stage reruns the entire fitting and calibration pipeline in every
# bootstrap resample for the randomized and confounded primary cells.
#
# Run from the repository root:
#   Rscript R/stage16e_cdf_calibration_coverage.R
#
# Execution smoke test:
#   SUDO_STAGE16E_SMOKE=1 Rscript R/stage16e_cdf_calibration_coverage.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16E_SMOKE", unset = ""))
coverage_reps <- stage16_env_int(
  "SUDO_STAGE16E_COVERAGE_REPS", if (smoke) 2L else 50L
)
bootstrap_reps <- stage16_env_int(
  "SUDO_STAGE16E_BOOTSTRAPS", if (smoke) 3L else 99L
)
bootstrap_fit_timeout <- stage16_env_int(
  "SUDO_STAGE16E_BOOTSTRAP_TIMEOUT", if (smoke) 60L else 600L
)
outer_folds <- stage16_env_int(
  "SUDO_STAGE16E_OUTER_FOLDS", if (smoke) 2L else 5L
)
inner_folds <- stage16_env_int(
  "SUDO_STAGE16E_INNER_FOLDS", if (smoke) 4L else 5L
)
cores <- stage16_env_int("SUDO_STAGE16E_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int(
  "SUDO_STAGE16E_CHUNK_SIZE", if (smoke) 1L else 5L
)
stopifnot(
  coverage_reps >= 2L, bootstrap_reps >= 2L,
  bootstrap_fit_timeout >= 1L, outer_folds >= 2L, inner_folds >= 4L,
  cores >= 1L, chunk_size >= 1L
)

tuning_file <- "R/results/stage16t_orthogonal_sudo_selection.csv"
confirmation_file <- "R/results/stage16d_cdf_calibration.csv"
if (!smoke && !file.exists(tuning_file)) {
  stop("stage 16e requires the full stage-16t clipping selection")
}
if (!smoke && !file.exists(confirmation_file)) {
  stop("stage 16e requires the passing stage-16d confirmation")
}
clip <- if (file.exists(tuning_file)) {
  tuning <- read.csv(tuning_file)
  tuning$clip[tuning$selected][1]
} else {
  1e-3
}
if (!smoke) {
  confirmation <- read.csv(confirmation_file)
  affine <- confirmation[confirmation$estimator == "pit_affine_tilt", ]
  if (nrow(affine) != 3L || !all(affine$bias_ok) ||
      any(affine$fail_rate >= 0.02)) {
    stop("stage 16d did not authorize affine-calibration coverage")
  }
}

cells <- stage16_cells(smoke)
cells <- cells[cells$primary, , drop = FALSE]
cell_filter <- Sys.getenv("SUDO_STAGE16E_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]],
                 , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L)

full_run <- !smoke && coverage_reps >= 50L && bootstrap_reps >= 99L &&
  outer_folds >= 5L && inner_folds >= 5L &&
  identical(as.character(cells$cell), c("randomized", "confounded"))
suffix <- if (full_run) "" else "_smoke"
checkpoint_file <- paste0(
  "R/results/stage16e_cdf_calibration_coverage_checkpoint", suffix, ".csv"
)
checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
checkpoint_keys <- c(
  "cell", "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "bootstrap_reps", "replication"
)
if (nrow(checkpoint) && !all(checkpoint_keys %in% names(checkpoint))) {
  checkpoint <- data.frame()
}
if (nrow(checkpoint)) {
  cell_match <- match(checkpoint$cell, cells$cell)
  checkpoint <- checkpoint[
    !is.na(cell_match) & checkpoint$clip == clip &
      checkpoint$cell_n == cells$n[cell_match] &
      checkpoint$cell_p == cells$p[cell_match] &
      checkpoint$outer_folds == outer_folds &
      checkpoint$inner_folds == inner_folds &
      checkpoint$bootstrap_reps == bootstrap_reps,
    , drop = FALSE
  ]
}

cluster <- if (cores > 1L) stage16_cluster(cores) else NULL
for (cell_index in seq_len(nrow(cells))) {
  cell <- cells[cell_index, ]
  completed <- if (nrow(checkpoint)) {
    unique(checkpoint$replication[
      checkpoint$cell == cell$cell &
        checkpoint$replication <= coverage_reps
    ])
  } else {
    integer()
  }
  remaining <- setdiff(seq_len(coverage_reps), completed)
  chunks <- split(remaining, ceiling(seq_along(remaining) / chunk_size))
  worker <- function(replication) {
    seed <- 1607000L + cell_index * 10000L + replication
    stage16_bootstrap_rep(
      seed, cell, clip, outer_folds, inner_folds, bootstrap_reps,
      bootstrap_fit_timeout, include_calibration = TRUE
    )
  }
  for (chunk in chunks) {
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
    estimators_per_rep <- length(unique(added$estimator))
    added$cell <- cell$cell
    added$primary <- TRUE
    added$clip <- clip
    added$cell_n <- cell$n
    added$cell_p <- cell$p
    added$outer_folds <- outer_folds
    added$inner_folds <- inner_folds
    added$bootstrap_reps <- bootstrap_reps
    added$bootstrap_fit_timeout <- bootstrap_fit_timeout
    added$replication <- rep(chunk, each = estimators_per_rep)
    checkpoint <- rbind(checkpoint, added)
    dir.create("R/results", showWarnings = FALSE)
    write.csv(checkpoint, checkpoint_file, row.names = FALSE)
    cat(sprintf(
      "affine coverage %-10s checkpointed %d/%d\n", cell$cell,
      length(unique(checkpoint$replication[checkpoint$cell == cell$cell])),
      coverage_reps
    ))
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)

cell_match <- match(checkpoint$cell, cells$cell)
replications <- checkpoint[
  !is.na(cell_match) & checkpoint$clip == clip &
    checkpoint$cell_n == cells$n[cell_match] &
    checkpoint$cell_p == cells$p[cell_match] &
    checkpoint$outer_folds == outer_folds &
    checkpoint$inner_folds == inner_folds &
    checkpoint$bootstrap_reps == bootstrap_reps &
    checkpoint$replication <= coverage_reps,
  , drop = FALSE
]
summary <- stage16_summarize_coverage(replications)
primary <- summary$estimator == "pit_affine_tilt"
summary$bias_ok <- abs(summary$bias) <= pmax(
  2 * summary$mc_se_bias, 0.03 * 0.75
)
summary$coverage_ok <- abs(summary$coverage - 0.95) <= pmax(
  2 * summary$mc_se_coverage, 0.04
)
summary$sd_to_se_ok <- abs(summary$sd_to_se - 1) <=
  2 * summary$ratio_mc_se
coverage_pass <- all(
  summary$bias_ok[primary] & summary$coverage_ok[primary] &
    summary$sd_to_se_ok[primary] &
    summary$mean_bootstrap_success[primary] >= 0.98
)
print(summary, row.names = FALSE)

dir.create("R/results", showWarnings = FALSE)
write.csv(
  replications,
  paste0("R/results/stage16e_cdf_calibration_coverage_replications",
         suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  summary,
  paste0("R/results/stage16e_cdf_calibration_coverage", suffix, ".csv"),
  row.names = FALSE
)

if (full_run && !coverage_pass) {
  stop(
    "stage 16e failed an affine-calibration coverage gate; results were ",
    "retained and no Python or manuscript promotion is authorized"
  )
}
if (coverage_pass) {
  cat("PASS: affine PIT calibration satisfies every primary coverage gate\n")
} else {
  cat("SMOKE DIAGNOSTIC: statistical promotion requires the full run\n")
}
