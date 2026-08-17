# Stage 16f: transfer affine score calibration into SUDO completion.
#
# The calibrated logistic coefficient from stages 16c to 16e does not itself
# use surrogate completion. This stage estimates the affine fluctuation on
# each training fold, constructs Rao-Blackwell surrogates from the calibrated
# held-out index, refits E[S | X], and then applies FWL.
#
# Run from the repository root:
#   Rscript R/stage16f_affine_calibrated_sudo.R
#
# Execution smoke test:
#   SUDO_STAGE16F_SMOKE=1 Rscript R/stage16f_affine_calibrated_sudo.R

source("R/stage16_orthogonal_support.R")
stage16_require_packages()
stage16_unit_checks()

smoke <- nzchar(Sys.getenv("SUDO_STAGE16F_SMOKE", unset = ""))
main_reps <- stage16_env_int("SUDO_STAGE16F_REPS", if (smoke) 2L else 20L)
boundary_reps <- stage16_env_int(
  "SUDO_STAGE16F_BOUNDARY_REPS", if (smoke) 2L else 10L
)
outer_folds <- stage16_env_int(
  "SUDO_STAGE16F_OUTER_FOLDS", if (smoke) 2L else 5L
)
inner_folds <- stage16_env_int(
  "SUDO_STAGE16F_INNER_FOLDS", if (smoke) 4L else 5L
)
cores <- stage16_env_int("SUDO_STAGE16F_CORES", if (smoke) 1L else 4L)
chunk_size <- stage16_env_int(
  "SUDO_STAGE16F_CHUNK_SIZE", if (smoke) 1L else 5L
)
stopifnot(
  main_reps >= 2L, boundary_reps >= 2L, outer_folds >= 2L,
  inner_folds >= 4L, cores >= 1L, chunk_size >= 1L
)

tuning_file <- "R/results/stage16t_orthogonal_sudo_selection.csv"
if (!smoke && !file.exists(tuning_file)) {
  stop("stage 16f requires the full stage-16t clipping selection")
}
clip <- if (file.exists(tuning_file)) {
  tuning <- read.csv(tuning_file)
  tuning$clip[tuning$selected][1]
} else {
  1e-3
}

cells <- stage16_cells(smoke)
cells$gating <- cells$cell %in% c(
  "randomized", "confounded", "high_signal"
)
cells$reps <- ifelse(
  cells$cell == "high_dimensional", boundary_reps, main_reps
)
cell_filter <- Sys.getenv("SUDO_STAGE16F_CELL", unset = "")
if (nzchar(cell_filter)) {
  cells <- cells[cells$cell %in% strsplit(cell_filter, ",")[[1]],
                 , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L)

full_run <- !smoke && main_reps >= 20L && boundary_reps >= 10L &&
  outer_folds >= 5L && inner_folds >= 5L &&
  identical(
    as.character(cells$cell),
    c("randomized", "confounded", "high_signal", "high_dimensional")
  )
suffix <- if (full_run) "" else "_smoke"
checkpoint_file <- paste0(
  "R/results/stage16f_affine_calibrated_sudo_checkpoint", suffix, ".csv"
)
checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else {
  data.frame()
}
checkpoint_keys <- c(
  "cell", "clip", "cell_n", "cell_p", "outer_folds", "inner_folds",
  "replication", "affine_calibrated_sudo"
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
  n_reps <- cell$reps
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
    seed <- 1608000L + cell_index * 10000L + replication
    result <- try(stage16_one_rep(
      seed, cell, clip, outer_folds, inner_folds,
      include_one_se = FALSE, include_calibration = TRUE
    ), silent = TRUE)
    if (inherits(result, "try-error")) {
      data.frame(
        seed = seed, selection = "min", truth = 0.75,
        sudo = NA_real_, orthogonal = NA_real_,
        targeted_projection = NA_real_, pit_affine_tilt = NA_real_,
        affine_calibrated_sudo = NA_real_, oracle_sudo = NA_real_,
        oracle_orthogonal = NA_real_,
        affine_calibrated_sudo_convergence_rate = NA_real_,
        affine_calibrated_sudo_max_score_error = NA_real_,
        error = as.character(result), stringsAsFactors = FALSE
      )
    } else {
      result$error <- ""
      result
    }
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
    added$gating <- cell$gating
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
      "calibrated SUDO %-16s checkpointed %d/%d\n", cell$cell,
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
    checkpoint$replication <= cells$reps[cell_match],
  , drop = FALSE
]
estimators <- c(
  "sudo", "orthogonal", "targeted_projection", "pit_affine_tilt",
  "affine_calibrated_sudo", "oracle_sudo", "oracle_orthogonal"
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
      cell = cell_name, gating = block$gating[1], estimator = estimator,
      n_reps = sum(ok), mean_est = mean(value[ok]),
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

target <- summary[
  summary$gating & summary$estimator == "affine_calibrated_sudo",
  , drop = FALSE
]
ordinary <- summary[
  summary$gating & summary$estimator == "sudo", c("cell", "bias")
]
comparison <- merge(
  target[c("cell", "bias")], ordinary, by = "cell",
  suffixes = c("_affine_sudo", "_sudo")
)
calibration_ok <- all(
  replications$affine_calibrated_sudo_convergence_rate[
    replications$gating
  ] == 1 &
    replications$affine_calibrated_sudo_max_score_error[
      replications$gating
    ] < 1e-8
)
transfer_pass <- all(target$bias_ok & target$fail_rate < 0.02) &&
  all(abs(comparison$bias_affine_sudo) <= abs(comparison$bias_sudo)) &&
  calibration_ok

dir.create("R/results", showWarnings = FALSE)
write.csv(
  replications,
  paste0("R/results/stage16f_affine_calibrated_sudo_replications",
         suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  summary,
  paste0("R/results/stage16f_affine_calibrated_sudo", suffix, ".csv"),
  row.names = FALSE
)

if (full_run && !transfer_pass) {
  stop(
    "stage 16f did not transfer affine calibration through SUDO; results ",
    "were retained and coverage was not promoted"
  )
}
if (transfer_pass) {
  cat("PASS: affine calibration transfers through SUDO completion\n")
} else {
  cat("SMOKE DIAGNOSTIC: statistical transfer requires the full run\n")
}
