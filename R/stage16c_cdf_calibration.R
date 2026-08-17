# Stage 16c: probability-integral-transform calibration screen.
#
# This stage tests whether a low-dimensional, unpenalized calibration of
# cross-fitted GAMSEL probabilities can reverse treatment regularization
# without using inverse-information pseudo-outcomes. It uses dedicated seeds
# and keeps the high-dimensional cell as a non-gating boundary.
#
# Run from the repository root:
#   Rscript R/stage16c_cdf_calibration.R
#
# Execution smoke test:
#   SUDO_STAGE16C_SMOKE=1 Rscript R/stage16c_cdf_calibration.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16C_SMOKE", unset = ""))
screen_reps <- stage16_env_int(
  "SUDO_STAGE16C_REPS", if (smoke) 2L else 20L
)
boundary_reps <- stage16_env_int(
  "SUDO_STAGE16C_BOUNDARY_REPS", if (smoke) 2L else 10L
)
outer_folds <- stage16_env_int(
  "SUDO_STAGE16C_OUTER_FOLDS", if (smoke) 2L else 5L
)
inner_folds <- stage16_env_int(
  "SUDO_STAGE16C_INNER_FOLDS", if (smoke) 4L else 5L
)
cores <- stage16_env_int("SUDO_STAGE16C_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int(
  "SUDO_STAGE16C_CHUNK_SIZE", if (smoke) 1L else 5L
)
stopifnot(screen_reps >= 2L, boundary_reps >= 2L, outer_folds >= 2L,
          inner_folds >= 4L, cores >= 1L, chunk_size >= 1L)

clip_env <- Sys.getenv("SUDO_STAGE16C_CLIP", unset = "")
selection_file <- "R/results/stage16t_orthogonal_sudo_selection.csv"
if (nzchar(clip_env)) {
  clip <- as.numeric(clip_env)
} else if (file.exists(selection_file)) {
  tuning <- read.csv(selection_file)
  clip <- tuning$clip[tuning$selected][1]
} else if (smoke) {
  clip <- 1e-3
} else {
  stop("stage 16c requires the frozen full stage-16t clipping selection")
}
stopifnot(length(clip) == 1L, is.finite(clip), clip > 0, clip < 0.5)

cells <- stage16_cells(smoke)
cells$replications <- ifelse(
  cells$cell == "high_dimensional", boundary_reps, screen_reps
)
cell_filter <- Sys.getenv("SUDO_STAGE16C_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]],
                 , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L)

expected_cells <- stage16_cells(FALSE)$cell
full_run <- !smoke && screen_reps >= 20L && boundary_reps >= 10L &&
  outer_folds >= 5L && inner_folds >= 5L &&
  identical(as.character(cells$cell), expected_cells)
suffix <- if (full_run) "" else "_smoke"
checkpoint_file <- paste0(
  "R/results/stage16c_cdf_calibration_checkpoint", suffix, ".csv"
)
checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
checkpoint_keys <- c(
  "cell", "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "replication", "pit_intercept_tilt", "pit_affine_tilt",
  "pit_affine_tilt_slope"
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
  n_reps <- cell$replications
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
    seed <- 1605000L + cell_index * 10000L + replication
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
    added$primary <- cell$primary
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
      "calibration %-16s checkpointed %d/%d\n", cell$cell,
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
    checkpoint$replication <= cells$replications[cell_match],
  , drop = FALSE
]
stopifnot(all(replications$direct_contrast_spread < 1e-8))

estimators <- c(
  "direct", "sudo", "scaled", "orthogonal", "targeted_projection",
  "targeted_sudo", "pit_intercept_tilt", "pit_affine_tilt",
  "oracle_sudo", "oracle_orthogonal"
)
summary_rows <- list()
position <- 0L
for (cell_name in cells$cell) {
  block <- replications[replications$cell == cell_name, , drop = FALSE]
  for (estimator in estimators) {
    value <- block[[estimator]]
    ok <- is.finite(value)
    position <- position + 1L
    summary_rows[[position]] <- data.frame(
      cell = cell_name,
      gating = cell_name %in% c("randomized", "confounded", "high_signal"),
      estimator = estimator, n_reps = sum(ok),
      mean_est = mean(value[ok]), bias = mean(value[ok] - block$truth[ok]),
      mc_se_bias = stats::sd(value[ok] - block$truth[ok]) / sqrt(sum(ok)),
      sd = stats::sd(value[ok]), fail_rate = mean(!ok)
    )
  }
}
point_summary <- do.call(rbind, summary_rows)
point_summary$bias_ok <- abs(point_summary$bias) <= pmax(
  2 * point_summary$mc_se_bias, 0.03 * 0.75
)

calibration_diagnostics <- do.call(rbind, lapply(
  split(replications, replications$cell),
  function(block) data.frame(
    cell = block$cell[1],
    intercept_tilt_convergence = mean(
      block$pit_intercept_tilt_converged
    ),
    affine_tilt_convergence = mean(block$pit_affine_tilt_converged),
    mean_affine_intercept = mean(block$pit_affine_tilt_intercept),
    mean_affine_slope = mean(block$pit_affine_tilt_slope),
    mean_affine_tilt = mean(block$pit_affine_tilt_tilt),
    max_affine_score_error = max(block$pit_affine_tilt_score_error)
  )
))
rownames(calibration_diagnostics) <- NULL

candidate_names <- c(
  "targeted_projection", "pit_intercept_tilt", "pit_affine_tilt"
)
candidate_cells <- c("randomized", "confounded", "high_signal")
candidate_summary <- point_summary[
  point_summary$cell %in% candidate_cells &
    point_summary$estimator %in% candidate_names, , drop = FALSE
]
selection <- do.call(rbind, lapply(
  split(candidate_summary, candidate_summary$estimator),
  function(block) data.frame(
    estimator = block$estimator[1],
    worst_abs_bias = max(abs(block$bias)),
    mean_abs_bias = mean(abs(block$bias)),
    all_bias_gates = all(block$bias_ok),
    max_fail_rate = max(block$fail_rate)
  )
))
rownames(selection) <- NULL
complexity <- match(selection$estimator, candidate_names)
finite_selection <- is.finite(selection$worst_abs_bias)
best_value <- min(selection$worst_abs_bias[finite_selection])
eligible <- which(
  finite_selection & selection$worst_abs_bias <= best_value + 0.005
)
stopifnot(length(eligible) >= 1L)
chosen <- eligible[which.min(complexity[eligible])]
selection$selected <- seq_len(nrow(selection)) == chosen
selection$screen_pass <- selection$all_bias_gates &
  selection$max_fail_rate < 0.02

print(point_summary, row.names = FALSE)
cat("\nCalibration diagnostics:\n")
print(calibration_diagnostics, row.names = FALSE)
cat("\nCalibration selection:\n")
print(selection[order(selection$worst_abs_bias), ], row.names = FALSE)

dir.create("R/results", showWarnings = FALSE)
write.csv(
  replications,
  paste0("R/results/stage16c_cdf_calibration_replications", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  point_summary,
  paste0("R/results/stage16c_cdf_calibration", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  calibration_diagnostics,
  paste0("R/results/stage16c_cdf_calibration_diagnostics", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  selection,
  paste0("R/results/stage16c_cdf_calibration_selection", suffix, ".csv"),
  row.names = FALSE
)

if (full_run) {
  selected <- selection[selection$selected, , drop = FALSE]
  if (selected$estimator == "targeted_projection" ||
      !selected$screen_pass) {
    stop(
      "stage 16c found no calibration improvement that passes the screen; ",
      "results were retained"
    )
  }
  cat(sprintf(
    "PASS: %s advances to fresh-seed confirmation\n", selected$estimator
  ))
} else {
  cat("EXECUTION SMOKE PASS: statistical selection requires the full run\n")
}
