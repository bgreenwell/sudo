# Stage 15t: dedicated-seed target-level tuning for count learners.
#
# Selection minimizes the worst absolute Rao-Blackwell SUDO bias across the
# six count cells. Predictive log score is diagnostic only. Confirmatory stage
# 15 never reuses these seeds.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/discrete.R")
source("R/sudo/count_validation.R")
source("R/sudo/mc.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}
n <- env_int("SUDO_STAGE15T_N", 1000L)
n_reps <- env_int("SUDO_STAGE15T_REPS", 20L)
n_folds <- env_int("SUDO_STAGE15T_FOLDS", 5L)
n_cores <- env_int("SUDO_STAGE15T_CORES", 6L)

xgb_grid <- expand.grid(
  max_depth = c(2L, 3L), min_child_weight = c(5, 20),
  nrounds = c(200L, 500L), stringsAsFactors = FALSE
)
xgb_grid$learner <- "xgboost"
xgb_grid$mtry <- NA_character_
xgb_grid$min.node.size <- NA_integer_
xgb_grid$config <- sprintf("xgb_d%d_c%d_r%d", xgb_grid$max_depth,
                           xgb_grid$min_child_weight, xgb_grid$nrounds)
ranger_grid <- expand.grid(
  min.node.size = c(5L, 20L), mtry = c("sqrt", "all"),
  stringsAsFactors = FALSE
)
ranger_grid$learner <- "ranger"
ranger_grid$max_depth <- NA_integer_
ranger_grid$min_child_weight <- NA_real_
ranger_grid$nrounds <- NA_integer_
ranger_grid$config <- sprintf("ranger_node%d_mtry%s",
                              ranger_grid$min.node.size, ranger_grid$mtry)
grid <- rbind(xgb_grid[, names(xgb_grid)], ranger_grid[, names(xgb_grid)])
cells <- count_cells()

config_filter <- Sys.getenv("SUDO_STAGE15T_CONFIG", unset = "")
if (nzchar(config_filter)) {
  grid <- grid[grid$config %in% strsplit(config_filter, ",", fixed = TRUE)[[1]],
               , drop = FALSE]
}
cell_filter <- Sys.getenv("SUDO_STAGE15T_CELL", unset = "")
if (nzchar(cell_filter)) {
  ids <- mapply(count_cell_id, cells$treatment, cells$family, cells$theta)
  cells <- cells[ids %in% strsplit(cell_filter, ",", fixed = TRUE)[[1]], , drop = FALSE]
}
stopifnot(nrow(grid) >= 1L, nrow(cells) >= 1L)

one_tuning_rep <- function(seed, n, cell, config, n_folds) {
  set.seed(seed)
  d <- dgp_count(n, cell$theta, cell$family, cell$treatment)
  folds <- make_folds(n, n_folds)
  sample_signature <- sum(d$Y * seq_along(d$Y))
  fold_signature <- sum(vapply(seq_along(folds), function(k)
    sum(folds[[k]]) * k, numeric(1)))
  set.seed(seed + 500000L)
  adapter <- if (config$learner == "xgboost")
    make_count_xgboost_adapter(
      cell$family, constrained = TRUE,
      max_depth = config$max_depth,
      min_child_weight = config$min_child_weight,
      nrounds = config$nrounds
    ) else make_count_ranger_adapter(
      cell$family, min.node.size = config$min.node.size, mtry = config$mtry
    )
  point <- count_rb_point(d, adapter, folds)
  quality <- count_quality(point$full, d, cell$family)
  data.frame(
    theta_hat = point$theta, truth = cell$theta,
    index_only = point$index_only, correction = point$correction,
    quality, sample_signature = sample_signature,
    fold_signature = fold_signature
  )
}

cluster <- if (n_cores > 1L) mc_cluster(c("one_tuning_rep"), n_cores) else NULL
rows <- list()
position <- 0L
for (j in seq_len(nrow(cells))) {
  cell <- cells[j, ]
  for (g in seq_len(nrow(grid))) {
    config <- grid[g, ]
    worker <- function(r) one_tuning_rep(
      count_seed(1502L, cell, r), n, cell, config, n_folds
    )
    if (!is.null(cluster)) parallel::clusterExport(
      cluster, c("cell", "config", "n", "n_folds", "worker"),
      envir = environment()
    )
    block <- do.call(rbind, if (is.null(cluster)) lapply(seq_len(n_reps), worker)
                     else parallel::parLapplyLB(cluster, seq_len(n_reps), worker))
    block$learner <- config$learner
    block$config <- config$config
    block$treatment <- cell$treatment
    block$family <- cell$family
    block$cell <- count_cell_id(cell$treatment, cell$family, cell$theta)
    block$replication <- seq_len(n_reps)
    position <- position + 1L
    rows[[position]] <- block
    cat(sprintf("%-27s %-24s bias %+.4f, log-score fraction %.3f\n",
                block$cell[1], config$config,
                mean(block$theta_hat - block$truth),
                mean(block$oracle_log_score_fraction)))
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
replications <- do.call(rbind, rows)

# Common samples and folds are a load-bearing comparison invariant.
signature_check <- aggregate(
  cbind(sample_signature, fold_signature) ~ cell + replication,
  replications, function(value) length(unique(value))
)
stopifnot(all(signature_check$sample_signature == 1L),
          all(signature_check$fold_signature == 1L))

cell_summary <- aggregate(
  cbind(error = replications$theta_hat - replications$truth,
        normalized_index_rmse = replications$normalized_index_rmse,
        index_correlation = replications$index_correlation,
        oracle_log_score_fraction = replications$oracle_log_score_fraction,
        dispersion = replications$dispersion) ~ learner + config + cell,
  replications, mean
)
names(cell_summary)[names(cell_summary) == "error"] <- "bias"
selection <- do.call(rbind, lapply(split(cell_summary, cell_summary$config),
                                   function(value) data.frame(
  learner = value$learner[1], config = value$config[1],
  worst_abs_bias = max(abs(value$bias)),
  mean_normalized_index_rmse = mean(value$normalized_index_rmse),
  mean_index_correlation = mean(value$index_correlation),
  mean_oracle_log_score_fraction = mean(value$oracle_log_score_fraction)
)))
rownames(selection) <- NULL

choose <- function(learner) {
  candidate <- selection[selection$learner == learner, , drop = FALSE]
  detail <- grid[match(candidate$config, grid$config), ]
  if (learner == "xgboost") {
    order_value <- order(candidate$worst_abs_bias, detail$max_depth,
                         detail$nrounds, -detail$min_child_weight)
  } else {
    order_value <- order(candidate$worst_abs_bias,
                         detail$mtry != "sqrt", -detail$min.node.size)
  }
  cbind(candidate[order_value[1], , drop = FALSE],
        detail[order_value[1], c("max_depth", "min_child_weight", "nrounds",
                                 "min.node.size", "mtry"), drop = FALSE])
}
selected <- do.call(rbind, lapply(unique(grid$learner), choose))
rownames(selected) <- NULL
print(selection[order(selection$learner, selection$worst_abs_bias), ],
      row.names = FALSE)
cat("\nSelected configurations:\n")
print(selected, row.names = FALSE)

full_run <- n >= 1000L && n_reps >= 20L && n_folds >= 5L &&
  nrow(cells) == 6L && nrow(grid) == 12L
if (full_run) {
  stopifnot(nrow(selected) == 2L, all(is.finite(selection$worst_abs_bias)))
  cat("PASS: full dedicated-seed tuning grid completed and configurations",
      "are frozen for confirmation\n")
} else {
  cat("SMOKE PASS: frozen configurations require the documented full grid\n")
}
dir.create("R/results", showWarnings = FALSE)
suffix <- if (full_run) "" else "_smoke"
write.csv(cell_summary,
          paste0("R/results/stage15t_count_tuning", suffix, ".csv"),
          row.names = FALSE)
write.csv(selection,
          paste0("R/results/stage15t_count_selection", suffix, ".csv"),
          row.names = FALSE)
write.csv(selected,
          paste0("R/results/stage15t_count_selected", suffix, ".csv"),
          row.names = FALSE)
