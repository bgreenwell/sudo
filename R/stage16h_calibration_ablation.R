# Stage 16h: rapid calibration-component ablation.
#
# This decision screen separates scale-only, tilt-only, and full affine
# calibration before surrogate completion. It uses fresh seeds and common
# samples across all arms. Ten replications per cell are sufficient for a
# directional decision; ambiguous arms can be extended by environment value.
#
# Run from the repository root:
#   Rscript R/stage16h_calibration_ablation.R
#
# Execution smoke test:
#   SUDO_STAGE16H_SMOKE=1 Rscript R/stage16h_calibration_ablation.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16H_SMOKE", unset = ""))
n_reps <- stage16_env_int("SUDO_STAGE16H_REPS", if (smoke) 2L else 10L)
outer_folds <- stage16_env_int(
  "SUDO_STAGE16H_OUTER_FOLDS", if (smoke) 2L else 5L
)
inner_folds <- stage16_env_int(
  "SUDO_STAGE16H_INNER_FOLDS", if (smoke) 4L else 5L
)
cores <- stage16_env_int("SUDO_STAGE16H_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int(
  "SUDO_STAGE16H_CHUNK_SIZE", if (smoke) 1L else 4L
)
stopifnot(
  n_reps >= 2L, outer_folds >= 2L, inner_folds >= 4L,
  cores >= 1L, chunk_size >= 1L
)

tuning_file <- "R/results/stage16t_orthogonal_sudo_selection.csv"
coverage_file <-
  "R/results/stage16g_affine_calibrated_sudo_coverage.csv"
if (!smoke && !file.exists(tuning_file)) {
  stop("stage 16h requires the full stage-16t clipping selection")
}
if (!smoke && !file.exists(coverage_file)) {
  stop("stage 16h requires the passing stage-16g coverage result")
}
clip <- if (file.exists(tuning_file)) {
  tuning <- read.csv(tuning_file)
  tuning$clip[tuning$selected][1]
} else {
  1e-3
}
if (!smoke) {
  coverage <- read.csv(coverage_file)
  target <- coverage[
    coverage$estimator == "affine_calibrated_sudo", , drop = FALSE
  ]
  if (nrow(target) != 3L ||
      !all(target$bias_ok & target$coverage_ok & target$sd_to_se_ok)) {
    stop("stage 16g did not authorize the component ablation")
  }
}

cells <- stage16_cells(smoke)
cells <- cells[cells$cell %in% c(
  "randomized", "confounded", "high_signal"
), , drop = FALSE]
cell_filter <- Sys.getenv("SUDO_STAGE16H_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]],
                 , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L)

full_run <- !smoke && n_reps >= 10L && outer_folds >= 5L &&
  inner_folds >= 5L && identical(
    as.character(cells$cell), c("randomized", "confounded", "high_signal")
  )
suffix <- if (full_run) "" else "_smoke"
checkpoint_file <- paste0(
  "R/results/stage16h_calibration_ablation_checkpoint", suffix, ".csv"
)
checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
checkpoint_keys <- c(
  "cell", "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "replication", "scale_calibrated_sudo", "tilt_calibrated_sudo",
  "affine_calibrated_sudo"
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
      checkpoint$inner_folds == inner_folds,
    , drop = FALSE
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
    seed <- 1610000L + cell_index * 10000L + replication
    stage16_one_rep(
      seed, cell, clip, outer_folds, inner_folds,
      include_one_se = FALSE, include_calibration = TRUE,
      include_calibration_ablation = TRUE
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
      "calibration ablation %-11s checkpointed %d/%d\n", cell$cell,
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
    checkpoint$replication <= n_reps,
  , drop = FALSE
]
estimators <- c(
  "sudo", "pit_affine_scale", "pit_intercept_tilt", "pit_affine_tilt",
  "scale_calibrated_sudo", "tilt_calibrated_sudo",
  "affine_calibrated_sudo"
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
      bias = mean(value[ok] - block$truth[ok]),
      mc_se_bias = stats::sd(value[ok] - block$truth[ok]) / sqrt(sum(ok)),
      sd = stats::sd(value[ok]), fail_rate = mean(!ok)
    )
  }
}
summary <- do.call(rbind, rows)
summary$bias_ok <- abs(summary$bias) <= pmax(
  2 * summary$mc_se_bias, 0.03 * 0.75
)
print(summary, row.names = FALSE)

mode_estimators <- c(
  scale = "scale_calibrated_sudo",
  tilt = "tilt_calibrated_sudo",
  full = "affine_calibrated_sudo"
)
decision <- do.call(rbind, lapply(names(mode_estimators), function(mode) {
  block <- summary[summary$estimator == mode_estimators[[mode]], ]
  data.frame(
    mode = mode,
    all_bias_gates = all(block$bias_ok & block$fail_rate < 0.02),
    worst_abs_bias = max(abs(block$bias)),
    stringsAsFactors = FALSE
  )
}))
cat("\nDecision summary:\n")
print(decision, row.names = FALSE)

dir.create("R/results", showWarnings = FALSE)
write.csv(
  replications,
  paste0("R/results/stage16h_calibration_ablation_replications", suffix,
         ".csv"),
  row.names = FALSE
)
write.csv(
  summary,
  paste0("R/results/stage16h_calibration_ablation", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  decision,
  paste0("R/results/stage16h_calibration_ablation_decision", suffix,
         ".csv"),
  row.names = FALSE
)

if (full_run) {
  cat("DECISION SCREEN COMPLETE: component attribution is ready\n")
} else {
  cat("SMOKE DIAGNOSTIC: attribution requires the rapid full screen\n")
}
