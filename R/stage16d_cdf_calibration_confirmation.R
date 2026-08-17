# Stage 16d: fresh-seed confirmation of affine PIT calibration.
#
# Stage 16c selects the calibration rule. This stage fixes that rule and
# evaluates it on new seeds in the randomized, confounded, and high-signal
# cells. No stage-16c replication is reused.
#
# Run from the repository root:
#   Rscript R/stage16d_cdf_calibration_confirmation.R
#
# Execution smoke test:
#   SUDO_STAGE16D_SMOKE=1 Rscript R/stage16d_cdf_calibration_confirmation.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16D_SMOKE", unset = ""))
n_reps <- stage16_env_int("SUDO_STAGE16D_REPS", if (smoke) 2L else 100L)
outer_folds <- stage16_env_int(
  "SUDO_STAGE16D_OUTER_FOLDS", if (smoke) 2L else 5L
)
inner_folds <- stage16_env_int(
  "SUDO_STAGE16D_INNER_FOLDS", if (smoke) 4L else 5L
)
cores <- stage16_env_int("SUDO_STAGE16D_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int(
  "SUDO_STAGE16D_CHUNK_SIZE", if (smoke) 1L else 5L
)
stopifnot(n_reps >= 2L, outer_folds >= 2L, inner_folds >= 4L,
          cores >= 1L, chunk_size >= 1L)

tuning_file <- "R/results/stage16t_orthogonal_sudo_selection.csv"
calibration_file <- "R/results/stage16c_cdf_calibration_selection.csv"
if (!smoke && !file.exists(tuning_file)) {
  stop("stage 16d requires the full stage-16t clipping selection")
}
if (!smoke && !file.exists(calibration_file)) {
  stop("stage 16d requires the full stage-16c calibration selection")
}
clip <- if (file.exists(tuning_file)) {
  tuning <- read.csv(tuning_file)
  tuning$clip[tuning$selected][1]
} else {
  1e-3
}
if (file.exists(calibration_file)) {
  calibration_selection <- read.csv(calibration_file)
  selected_estimator <- calibration_selection$estimator[
    calibration_selection$selected
  ][1]
  if (!smoke && selected_estimator != "pit_affine_tilt") {
    stop("stage 16c did not select affine PIT calibration")
  }
}
stopifnot(length(clip) == 1L, is.finite(clip), clip > 0, clip < 0.5)

cells <- stage16_cells(smoke)
cells <- cells[cells$cell %in%
                 c("randomized", "confounded", "high_signal"),
               , drop = FALSE]
cell_filter <- Sys.getenv("SUDO_STAGE16D_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]],
                 , drop = FALSE]
}
expected_cells <- c("randomized", "confounded", "high_signal")
full_run <- !smoke && n_reps >= 100L && outer_folds >= 5L &&
  inner_folds >= 5L && identical(as.character(cells$cell), expected_cells)
suffix <- if (full_run) "" else "_smoke"

checkpoint_file <- paste0(
  "R/results/stage16d_cdf_calibration_checkpoint", suffix, ".csv"
)
checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
checkpoint_keys <- c(
  "cell", "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "replication", "pit_affine_tilt", "pit_affine_tilt_slope"
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
      checkpoint$inner_folds == inner_folds, , drop = FALSE
  ]
}

cluster <- if (cores > 1L) stage16_cluster(cores) else NULL
for (cell_index in seq_len(nrow(cells))) {
  cell <- cells[cell_index, ]
  completed <- if (nrow(checkpoint)) {
    unique(checkpoint$replication[
      checkpoint$cell == cell$cell & checkpoint$replication <= n_reps
    ])
  } else {
    integer()
  }
  remaining <- setdiff(seq_len(n_reps), completed)
  chunks <- split(remaining, ceiling(seq_along(remaining) / chunk_size))
  worker <- function(replication) {
    seed <- 1606000L + cell_index * 10000L + replication
    stage16_one_rep(
      seed, cell, clip, outer_folds, inner_folds,
      include_one_se = FALSE, include_calibration = TRUE
    )
  }
  for (chunk in chunks) {
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
    added$clip <- clip
    added$cell_n <- cell$n
    added$cell_p <- cell$p
    added$outer_folds <- outer_folds
    added$inner_folds <- inner_folds
    added$replication <- chunk
    checkpoint <- rbind(checkpoint, added)
    dir.create("R/results", showWarnings = FALSE)
    write.csv(checkpoint, checkpoint_file, row.names = FALSE)
    cat(sprintf(
      "confirmation %-11s checkpointed %d/%d\n", cell$cell,
      length(unique(checkpoint$replication[
        checkpoint$cell == cell$cell
      ])), n_reps
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
    checkpoint$replication <= n_reps, , drop = FALSE
]
stopifnot(all(replications$direct_contrast_spread < 1e-8))

estimators <- c(
  "direct", "sudo", "scaled", "orthogonal", "targeted_projection",
  "targeted_sudo", "pit_intercept_tilt", "pit_affine_tilt",
  "oracle_sudo", "oracle_orthogonal"
)
rows <- list()
position <- 0L
for (cell_name in cells$cell) {
  block <- replications[replications$cell == cell_name, , drop = FALSE]
  for (estimator in estimators) {
    value <- block[[estimator]]
    ok <- is.finite(value)
    position <- position + 1L
    rows[[position]] <- data.frame(
      cell = cell_name, estimator = estimator, n_reps = sum(ok),
      mean_est = mean(value[ok]), bias = mean(value[ok] - block$truth[ok]),
      mc_se_bias = stats::sd(value[ok] - block$truth[ok]) / sqrt(sum(ok)),
      sd = stats::sd(value[ok]), fail_rate = mean(!ok)
    )
  }
}
point_summary <- do.call(rbind, rows)
point_summary$bias_ok <- abs(point_summary$bias) <= pmax(
  2 * point_summary$mc_se_bias, 0.03 * 0.75
)
print(point_summary, row.names = FALSE)

affine <- point_summary[
  point_summary$estimator == "pit_affine_tilt", , drop = FALSE
]
targeted <- point_summary[
  point_summary$estimator == "targeted_projection", , drop = FALSE
]
oracle <- point_summary[
  point_summary$estimator %in% c("oracle_sudo", "oracle_orthogonal"),
  , drop = FALSE
]
convergence_pass <- all(
  replications$pit_affine_tilt_converged == 1 &
    replications$pit_affine_tilt_score_error < 1e-8
)
bias_pass <- all(affine$bias_ok & affine$fail_rate < 0.02)
improvement_pass <- max(abs(affine$bias)) <= max(abs(targeted$bias))
oracle_pass <- all(abs(oracle$bias) <= 3 * oracle$mc_se_bias)
confirmation_pass <- bias_pass & improvement_pass & oracle_pass &
  convergence_pass

diagnostics <- do.call(rbind, lapply(
  split(replications, replications$cell),
  function(block) data.frame(
    cell = block$cell[1], mean_affine_intercept = mean(
      block$pit_affine_tilt_intercept
    ), mean_affine_slope = mean(block$pit_affine_tilt_slope),
    mean_affine_tilt = mean(block$pit_affine_tilt_tilt),
    max_score_error = max(block$pit_affine_tilt_score_error),
    convergence_rate = mean(block$pit_affine_tilt_converged)
  )
))
rownames(diagnostics) <- NULL
cat("\nCalibration diagnostics:\n")
print(diagnostics, row.names = FALSE)

dir.create("R/results", showWarnings = FALSE)
write.csv(
  replications,
  paste0("R/results/stage16d_cdf_calibration_replications", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  point_summary,
  paste0("R/results/stage16d_cdf_calibration", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  diagnostics,
  paste0("R/results/stage16d_cdf_calibration_diagnostics", suffix, ".csv"),
  row.names = FALSE
)

if (full_run && !confirmation_pass) {
  stop(
    "stage 16d affine PIT calibration failed confirmation; results were ",
    "retained and coverage was not promoted"
  )
}
if (full_run) {
  cat("PASS: affine PIT calibration advances to full-pipeline coverage\n")
} else {
  cat("EXECUTION SMOKE PASS: confirmation requires the full run\n")
}
