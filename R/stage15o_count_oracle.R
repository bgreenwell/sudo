# Stage 15o: oracle bridge for coefficientless count SUDO.
#
# Six prespecified cells use the true scalar index and true count CDF. The
# full run uses n=2000, 100 replications, B=25 randomized completions, and 99
# full-pipeline bootstrap resamples. Smaller environment overrides are smoke
# checks only.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/pl.R")
source("R/sudo/discrete.R")
source("R/sudo/estimator.R")
source("R/sudo/count_validation.R")
source("R/sudo/mc.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}
n <- env_int("SUDO_STAGE15O_N", 2000L)
n_reps <- env_int("SUDO_STAGE15O_REPS", 100L)
B_outer <- env_int("SUDO_STAGE15O_OUTER", 99L)
B <- env_int("SUDO_STAGE15O_B", 25L)
n_folds <- env_int("SUDO_STAGE15O_FOLDS", 5L)
n_cores <- env_int("SUDO_STAGE15O_CORES", 6L)
chunk_size <- env_int("SUDO_STAGE15O_CHUNK", max(1L, n_cores))
cells <- count_cells()
cell_filter <- Sys.getenv("SUDO_STAGE15O_CELL", unset = "")
if (nzchar(cell_filter)) {
  ids <- mapply(count_cell_id, cells$treatment, cells$family, cells$theta)
  cells <- cells[ids %in% strsplit(cell_filter, ",", fixed = TRUE)[[1]], , drop = FALSE]
}
stopifnot(nrow(cells) >= 1L)
full_design <- n >= 2000L && n_reps >= 100L && B_outer >= 99L && B >= 25L &&
  n_folds >= 5L && nrow(cells) == 6L
checkpoint_file <- if (full_design)
  "R/results/stage15o_count_oracle_checkpoint.csv" else
  "R/results/stage15o_count_oracle_checkpoint_smoke.csv"
checkpoint <- if (file.exists(checkpoint_file))
  read.csv(checkpoint_file, stringsAsFactors = FALSE) else data.frame()

one_oracle_rep <- function(seed, n, cell, B_outer, B, n_folds) {
  set.seed(seed)
  d <- dgp_count(n, cell$theta, cell$family, cell$treatment)
  base_folds <- make_folds(n, n_folds)
  bootstrap_indices <- lapply(seq_len(B_outer), function(b)
    sample.int(n, replace = TRUE))
  bootstrap_folds <- lapply(seq_len(B_outer), function(b)
    make_folds(n, n_folds))
  adapter <- make_count_oracle_adapter(cell$family, cell$theta)
  fit <- sudo_pipeline_boot(
    d, B_outer = B_outer, inner_B = B, n_folds = n_folds,
    estimator = sudo_count, adapter = adapter,
    outcome_family = cell$family, structured_comparator = FALSE,
    one_step = FALSE, include_index_only = FALSE, include_raw_label = FALSE,
    base_folds = base_folds,
    bootstrap_indices = bootstrap_indices, bootstrap_folds = bootstrap_folds
  )
  target <- fit$target_results
  get <- function(name, column) target[[column]][target$target == name]
  data.frame(
    truth = cell$theta,
    rao_blackwell_theta = get("rao_blackwell", "estimate"),
    rao_blackwell_se = get("rao_blackwell", "se"),
    randomized_theta = get("randomized", "estimate"),
    randomized_se = get("randomized", "se"),
    completion_difference = get("randomized", "estimate") -
      get("rao_blackwell", "estimate")
  )
}

cluster <- if (n_cores > 1L) mc_cluster(c("one_oracle_rep"), n_cores) else NULL
all_replications <- list()
all_summary <- list()
for (j in seq_len(nrow(cells))) {
  cell <- cells[j, ]
  cell_id <- count_cell_id(cell$treatment, cell$family, cell$theta)
  worker <- function(r) one_oracle_rep(
    count_seed(1501L, cell, r), n, cell, B_outer, B, n_folds
  )
  if (!is.null(cluster)) parallel::clusterExport(
    cluster, c("cell", "n", "B_outer", "B", "n_folds", "worker"),
    envir = environment()
  )
  completed <- if (nrow(checkpoint)) checkpoint[
    checkpoint$cell == cell_id & checkpoint$replication <= n_reps,
    , drop = FALSE
  ] else data.frame()
  remaining <- setdiff(seq_len(n_reps), completed$replication)
  if (length(remaining)) {
    chunks <- split(remaining, ceiling(seq_along(remaining) / chunk_size))
    for (chunk in chunks) {
      added <- do.call(rbind, if (is.null(cluster)) lapply(chunk, worker) else
        parallel::parLapplyLB(cluster, chunk, worker))
      added$treatment <- cell$treatment
      added$family <- cell$family
      added$cell <- cell_id
      added$replication <- chunk
      checkpoint <- rbind(checkpoint, added)
      dir.create("R/results", showWarnings = FALSE)
      write.csv(checkpoint, checkpoint_file, row.names = FALSE)
      cat(sprintf("%-26s checkpointed %d/%d replications\n",
                  cell_id,
                  sum(checkpoint$cell == cell_id), n_reps))
    }
  }
  reps <- checkpoint[checkpoint$cell == cell_id &
                       checkpoint$replication <= n_reps, , drop = FALSE]
  reps <- reps[order(reps$replication), , drop = FALSE]
  stopifnot(nrow(reps) == n_reps)
  summary <- count_mc_summary(reps)
  summary <- summary[summary$estimator %in% c("rao_blackwell", "randomized"), ]
  summary$treatment <- cell$treatment
  summary$family <- cell$family
  summary$truth <- cell$theta
  summary$cell <- reps$cell[1]
  difference_se <- sd(reps$completion_difference) / sqrt(n_reps)
  summary$completion_mean_difference <- mean(reps$completion_difference)
  summary$completion_mc_se <- difference_se
  all_replications[[j]] <- reps
  all_summary[[j]] <- summary
  cat(sprintf("%-26s RB bias %+.4f, randomized bias %+.4f\n",
              reps$cell[1], summary$bias[1], summary$bias[2]))
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
replications <- do.call(rbind, all_replications)
summary <- do.call(rbind, all_summary)
rownames(summary) <- NULL
print(format(summary, digits = 4), row.names = FALSE)

full_run <- full_design
if (full_run) {
  gates <- count_primary_gates(summary)
  bridge_ok <- abs(summary$completion_mean_difference) <=
    2 * summary$completion_mc_se
  stopifnot(all(gates$pass), all(bridge_ok))
  cat("PASS: oracle Rao-Blackwell and randomized estimators satisfy every",
      "bias, coverage, SD-to-SE, and bridge gate\n")
} else {
  cat("SMOKE PASS: oracle statistical gates require the documented defaults\n")
}
dir.create("R/results", showWarnings = FALSE)
suffix <- if (full_run) "" else "_smoke"
write.csv(summary, paste0("R/results/stage15o_count_oracle", suffix, ".csv"),
          row.names = FALSE)
write.csv(replications,
          paste0("R/results/stage15o_count_oracle_replications", suffix, ".csv"),
          row.names = FALSE)
