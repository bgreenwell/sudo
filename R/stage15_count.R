# Stage 15: coefficientless count learner confirmation and ablations.
#
# The full run first evaluates point bias and oracle-quality diagnostics on
# common samples for constrained XGBoost, unrestricted XGBoost, ranger, an
# incorrect Poisson CDF under negative-binomial truth, and the oracle. It then
# runs full-pipeline coverage for constrained XGBoost and only promotes a
# sensitivity arm when its prespecified point gates pass.
#
# Full defaults: n=2000, 100 replications, B=25, 99 outer resamples, 5 folds.
# A completed full stage-15t tuning file is required before confirmation.

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
n <- env_int("SUDO_STAGE15_N", 2000L)
n_reps <- env_int("SUDO_STAGE15_REPS", 100L)
B_outer <- env_int("SUDO_STAGE15_OUTER", 99L)
B <- env_int("SUDO_STAGE15_B", 25L)
n_folds <- env_int("SUDO_STAGE15_FOLDS", 5L)
n_cores <- env_int("SUDO_STAGE15_CORES", 6L)
point_chunk_size <- env_int("SUDO_STAGE15_POINT_CHUNK", max(1L, n_cores))
point_only <- identical(Sys.getenv("SUDO_STAGE15_POINT_ONLY", unset = "0"), "1")
coverage_chunk_size <- env_int("SUDO_STAGE15_COVERAGE_CHUNK",
                               max(1L, n_cores))
force_primary_coverage <- identical(
  Sys.getenv("SUDO_STAGE15_FORCE_PRIMARY_COVERAGE", unset = "0"), "1"
)
smoke_rounds <- env_int("SUDO_STAGE15_XGB_ROUNDS", 30L)
cells <- count_cells()
cell_filter <- Sys.getenv("SUDO_STAGE15_CELL", unset = "")
if (nzchar(cell_filter)) {
  ids <- mapply(count_cell_id, cells$treatment, cells$family, cells$theta)
  cells <- cells[ids %in% strsplit(cell_filter, ",", fixed = TRUE)[[1]], , drop = FALSE]
}
arm_filter <- Sys.getenv("SUDO_STAGE15_ARM", unset = "")

full_settings <- n >= 2000L && n_reps >= 100L && B_outer >= 99L && B >= 25L &&
  n_folds >= 5L && nrow(cells) == 6L && !nzchar(arm_filter)
selected_file <- "R/results/stage15t_count_selected.csv"
oracle_file <- "R/results/stage15o_count_oracle.csv"
if (full_settings &&
    (!file.exists(selected_file) || !file.exists(oracle_file))) {
  stop("full confirmation requires passed full stage-15o oracle and ",
       "stage-15t tuning result files")
}
if (full_settings) {
  oracle_summary <- read.csv(oracle_file, stringsAsFactors = FALSE)
  oracle_gates <- count_primary_gates(oracle_summary)
  oracle_bridge <- abs(oracle_summary$completion_mean_difference) <=
    2 * oracle_summary$completion_mc_se
  stopifnot(nrow(oracle_summary) == 12L, all(oracle_gates$pass),
            all(oracle_bridge))
}

if (file.exists(selected_file)) {
  selected <- read.csv(selected_file, stringsAsFactors = FALSE)
  xrow <- selected[selected$learner == "xgboost", , drop = FALSE]
  rrow <- selected[selected$learner == "ranger", , drop = FALSE]
  stopifnot(nrow(xrow) == 1L, nrow(rrow) == 1L)
  xgb_config <- list(max_depth = xrow$max_depth,
                     min_child_weight = xrow$min_child_weight,
                     nrounds = xrow$nrounds)
  ranger_config <- list(min.node.size = rrow$min.node.size, mtry = rrow$mtry)
} else {
  xgb_config <- list(max_depth = 2L, min_child_weight = 20,
                     nrounds = smoke_rounds)
  ranger_config <- list(min.node.size = 20L, mtry = "sqrt")
}

count_arms <- function(family) {
  value <- c("xgb_constrained", "xgb_unrestricted", "ranger", "oracle")
  if (family == "negbin") value <- c(value, "wrong_poisson_cdf")
  if (nzchar(arm_filter)) {
    value <- intersect(value, strsplit(arm_filter, ",", fixed = TRUE)[[1]])
  }
  value
}

make_arm_adapter <- function(arm, cell, xgb_config, ranger_config) {
  switch(
    arm,
    xgb_constrained = make_count_xgboost_adapter(
      cell$family, constrained = TRUE,
      max_depth = xgb_config$max_depth,
      min_child_weight = xgb_config$min_child_weight,
      nrounds = xgb_config$nrounds
    ),
    xgb_unrestricted = make_count_xgboost_adapter(
      cell$family, constrained = FALSE,
      max_depth = xgb_config$max_depth,
      min_child_weight = xgb_config$min_child_weight,
      nrounds = xgb_config$nrounds
    ),
    ranger = make_count_ranger_adapter(
      cell$family, min.node.size = ranger_config$min.node.size,
      mtry = ranger_config$mtry
    ),
    wrong_poisson_cdf = make_count_xgboost_adapter(
      cell$family, cdf_family = "poisson", constrained = TRUE,
      max_depth = xgb_config$max_depth,
      min_child_weight = xgb_config$min_child_weight,
      nrounds = xgb_config$nrounds
    ),
    oracle = make_count_oracle_adapter(cell$family, cell$theta),
    stop("unknown count arm: ", arm)
  )
}

one_point_rep <- function(seed, n, cell, arm, n_folds,
                          xgb_config, ranger_config) {
  set.seed(seed)
  d <- dgp_count(n, cell$theta, cell$family, cell$treatment)
  folds <- make_folds(n, n_folds)
  sample_signature <- sum(d$Y * seq_along(d$Y))
  fold_signature <- sum(vapply(seq_along(folds), function(k)
    sum(folds[[k]]) * k, numeric(1)))
  set.seed(seed + 600000L)
  adapter <- make_arm_adapter(arm, cell, xgb_config, ranger_config)
  point <- count_rb_point(d, adapter, folds)
  quality <- count_quality(point$full, d, cell$family)
  data.frame(
    truth = cell$theta, theta_hat = point$theta,
    index_only = point$index_only, correction = point$correction,
    quality, sample_signature = sample_signature,
    fold_signature = fold_signature
  )
}

cluster <- if (n_cores > 1L) mc_cluster(
  c("one_point_rep", "make_arm_adapter"), n_cores
) else NULL
dir.create("R/results", showWarnings = FALSE)
point_checkpoint_file <- if (full_settings)
  "R/results/stage15_count_point_checkpoint.csv" else
  "R/results/stage15_count_point_checkpoint_smoke.csv"
point_checkpoint <- if (file.exists(point_checkpoint_file))
  read.csv(point_checkpoint_file, stringsAsFactors = FALSE) else data.frame()
point_rows <- list()
position <- 0L
for (j in seq_len(nrow(cells))) {
  cell <- cells[j, ]
  cell_id <- count_cell_id(cell$treatment, cell$family, cell$theta)
  for (arm in count_arms(cell$family)) {
    worker <- function(r) one_point_rep(
      count_seed(1503L, cell, r), n, cell, arm, n_folds,
      xgb_config, ranger_config
    )
    if (!is.null(cluster)) parallel::clusterExport(
      cluster,
      c("cell", "arm", "n", "n_folds", "xgb_config", "ranger_config",
        "worker"), envir = environment()
    )
    completed <- if (nrow(point_checkpoint)) point_checkpoint[
      point_checkpoint$cell == cell_id & point_checkpoint$arm == arm &
        point_checkpoint$replication <= n_reps,
      , drop = FALSE
    ] else data.frame()
    remaining <- setdiff(seq_len(n_reps), completed$replication)
    if (length(remaining)) {
      chunks <- split(
        remaining, ceiling(seq_along(remaining) / point_chunk_size)
      )
      for (chunk in chunks) {
        added <- do.call(rbind, if (is.null(cluster)) lapply(chunk, worker) else
          parallel::parLapplyLB(cluster, chunk, worker))
        added$arm <- arm
        added$treatment <- cell$treatment
        added$family <- cell$family
        added$cell <- cell_id
        added$replication <- chunk
        point_checkpoint <- rbind(point_checkpoint, added)
        write.csv(point_checkpoint, point_checkpoint_file, row.names = FALSE)
        cat(sprintf("point %-26s %-20s checkpointed %d/%d\n",
                    cell_id, arm,
                    sum(point_checkpoint$cell == cell_id &
                          point_checkpoint$arm == arm), n_reps))
      }
    }
    block <- point_checkpoint[
      point_checkpoint$cell == cell_id & point_checkpoint$arm == arm &
        point_checkpoint$replication <= n_reps,
      , drop = FALSE
    ]
    block <- block[order(block$replication), , drop = FALSE]
    stopifnot(nrow(block) == n_reps)
    position <- position + 1L
    point_rows[[position]] <- block
    cat(sprintf("point %-26s %-20s bias %+.4f\n", block$cell[1], arm,
                mean(block$theta_hat - block$truth)))
  }
}
point_replications <- do.call(rbind, point_rows)

# Every arm in a cell must see exactly the same generated sample and folds.
signature_check <- aggregate(
  cbind(sample_signature, fold_signature) ~ cell + replication,
  point_replications, function(value) length(unique(value))
)
stopifnot(all(signature_check$sample_signature == 1L),
          all(signature_check$fold_signature == 1L))

point_summary <- do.call(rbind, lapply(
  split(point_replications, list(point_replications$cell,
                                 point_replications$arm), drop = TRUE),
  function(value) data.frame(
    cell = value$cell[1], treatment = value$treatment[1],
    family = value$family[1], truth = value$truth[1], arm = value$arm[1],
    n_reps = nrow(value), bias = mean(value$theta_hat - value$truth),
    mc_se_bias = sd(value$theta_hat - value$truth) / sqrt(nrow(value)),
    normalized_index_rmse = mean(value$normalized_index_rmse),
    index_correlation = mean(value$index_correlation),
    oracle_log_score_fraction = mean(value$oracle_log_score_fraction),
    dispersion = mean(value$dispersion),
    max_contrast_spread = max(value$max_contrast_spread)
  )
))
rownames(point_summary) <- NULL
point_summary$bias_ok <- abs(point_summary$bias) <=
  pmax(2 * point_summary$mc_se_bias, 0.03 * abs(point_summary$truth))
point_summary$quality_ok <- point_summary$normalized_index_rmse <= 0.25 &
  point_summary$index_correlation >= 0.95 &
  point_summary$oracle_log_score_fraction >= 0.90 &
  (point_summary$family == "poisson" |
     abs(point_summary$dispersion / 2 - 1) <= 0.20)
point_summary$quality_ok[point_summary$arm == "oracle"] <- TRUE

promotion <- do.call(rbind, lapply(split(point_summary, point_summary$arm),
                                   function(value) data.frame(
  arm = value$arm[1], all_bias_ok = isTRUE(all(value$bias_ok)),
  all_quality_ok = isTRUE(all(value$quality_ok)),
  promote_coverage = isTRUE(all(value$bias_ok)) &
    isTRUE(all(value$quality_ok))
)))
rownames(promotion) <- NULL
primary_row <- promotion$arm == "xgb_constrained"
promotion$promote_coverage[primary_row] <- promotion$all_bias_ok[primary_row]
promotion$promote_coverage[promotion$arm == "oracle"] <- FALSE
promotion$protocol_promote_coverage <- promotion$promote_coverage
if (force_primary_coverage) {
  promotion$promote_coverage[primary_row] <- TRUE
}
promotion$forced_coverage <- promotion$promote_coverage &
  !promotion$protocol_promote_coverage
forced_primary_run <- any(
  promotion$arm == "xgb_constrained" & promotion$forced_coverage
)
primary_protocol_promoted <- isTRUE(
  promotion$protocol_promote_coverage[primary_row]
)
exploratory_run <- full_settings && !primary_protocol_promoted
print(point_summary, row.names = FALSE)
cat("\nCoverage promotion decisions:\n")
print(promotion, row.names = FALSE)

suffix <- if (full_settings) "" else "_smoke"
point_suffix <- if (exploratory_run) "_exploratory" else suffix
write.csv(point_summary,
          paste0("R/results/stage15_count_point", point_suffix, ".csv"),
          row.names = FALSE)
write.csv(promotion,
          paste0("R/results/stage15_count_promotion", point_suffix, ".csv"),
          row.names = FALSE)
if (point_only) {
  if (!is.null(cluster)) parallel::stopCluster(cluster)
  cat("POINT PHASE COMPLETE: coverage was not started\n")
  quit(save = "no", status = 0L)
}
if (full_settings && !primary_protocol_promoted &&
    !force_primary_coverage) {
  if (!is.null(cluster)) parallel::stopCluster(cluster)
  stop("constrained XGBoost failed its point-screen promotion gate; ",
       "exploratory point diagnostics were retained and coverage was not ",
       "started")
}

one_coverage_rep <- function(seed, n, cell, arm, B_outer, B, n_folds,
                             xgb_config, ranger_config) {
  set.seed(seed)
  d <- dgp_count(n, cell$theta, cell$family, cell$treatment)
  base_folds <- make_folds(n, n_folds)
  bootstrap_indices <- lapply(seq_len(B_outer), function(b)
    sample.int(n, replace = TRUE))
  bootstrap_folds <- lapply(seq_len(B_outer), function(b)
    make_folds(n, n_folds))
  set.seed(seed + 700000L)
  adapter <- make_arm_adapter(arm, cell, xgb_config, ranger_config)
  fit <- sudo_pipeline_boot(
    d, B_outer = B_outer, inner_B = B, n_folds = n_folds,
    estimator = sudo_count, adapter = adapter,
    outcome_family = cell$family, base_folds = base_folds,
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
    index_only_theta = get("index_only", "estimate"),
    index_only_se = get("index_only", "se"),
    poisson_one_step_theta = get("poisson_one_step", "estimate"),
    poisson_one_step_se = get("poisson_one_step", "se"),
    structured_gam_theta = fit$structured_gam_theta,
    raw_label_theta = fit$raw_label_theta,
    outcome_correction = fit$outcome_correction,
    completion_difference = fit$completion_difference,
    dispersion = fit$dispersion,
    heldout_log_score = fit$heldout_log_score,
    calibration_intercept = fit$calibration_intercept,
    calibration_slope = fit$calibration_slope
  )
}

promoted_arms <- promotion$arm[promotion$promote_coverage]
coverage_checkpoint_file <- if (!full_settings) {
  "R/results/stage15_count_coverage_checkpoint_smoke.csv"
} else if (exploratory_run) {
  "R/results/stage15_count_coverage_checkpoint_exploratory.csv"
} else {
  "R/results/stage15_count_coverage_checkpoint.csv"
}
coverage_checkpoint <- if (file.exists(coverage_checkpoint_file))
  read.csv(coverage_checkpoint_file, stringsAsFactors = FALSE) else data.frame()
coverage_rows <- list()
position <- 0L
for (j in seq_len(nrow(cells))) {
  cell <- cells[j, ]
  cell_id <- count_cell_id(cell$treatment, cell$family, cell$theta)
  for (arm in intersect(count_arms(cell$family), promoted_arms)) {
    worker <- function(r) one_coverage_rep(
      count_seed(1504L, cell, r), n, cell, arm, B_outer, B, n_folds,
      xgb_config, ranger_config
    )
    if (!is.null(cluster)) parallel::clusterExport(
      cluster,
      c("cell", "arm", "n", "B_outer", "B", "n_folds", "xgb_config",
        "ranger_config", "one_coverage_rep", "worker"),
      envir = environment()
    )
    completed <- if (nrow(coverage_checkpoint)) coverage_checkpoint[
      coverage_checkpoint$cell == cell_id & coverage_checkpoint$arm == arm &
        coverage_checkpoint$replication <= n_reps,
      , drop = FALSE
    ] else data.frame()
    remaining <- setdiff(seq_len(n_reps), completed$replication)
    if (length(remaining)) {
      chunks <- split(
        remaining, ceiling(seq_along(remaining) / coverage_chunk_size)
      )
      for (chunk in chunks) {
        added <- do.call(rbind, if (is.null(cluster)) lapply(chunk, worker) else
          parallel::parLapplyLB(cluster, chunk, worker))
        added$arm <- arm
        added$treatment <- cell$treatment
        added$family <- cell$family
        added$cell <- cell_id
        added$replication <- chunk
        added$protocol_promoted <- promotion$protocol_promote_coverage[
          promotion$arm == arm
        ]
        added$forced_coverage <- promotion$forced_coverage[
          promotion$arm == arm
        ]
        coverage_checkpoint <- rbind(coverage_checkpoint, added)
        write.csv(coverage_checkpoint, coverage_checkpoint_file,
                  row.names = FALSE)
        cat(sprintf("coverage %-23s %-20s checkpointed %d/%d\n",
                    cell_id, arm,
                    sum(coverage_checkpoint$cell == cell_id &
                          coverage_checkpoint$arm == arm), n_reps))
      }
    }
    block <- coverage_checkpoint[
      coverage_checkpoint$cell == cell_id & coverage_checkpoint$arm == arm &
        coverage_checkpoint$replication <= n_reps,
      , drop = FALSE
    ]
    block <- block[order(block$replication), , drop = FALSE]
    stopifnot(nrow(block) == n_reps)
    position <- position + 1L
    coverage_rows[[position]] <- block
    cat(sprintf("coverage %-23s %-20s completed\n", block$cell[1], arm))
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
coverage_replications <- if (length(coverage_rows)) {
  do.call(rbind, coverage_rows)
} else {
  data.frame()
}
coverage_summary <- if (nrow(coverage_replications)) {
  do.call(rbind, lapply(
    split(coverage_replications,
          list(coverage_replications$cell, coverage_replications$arm),
          drop = TRUE),
    function(value) {
      result <- count_mc_summary(value)
      result$cell <- value$cell[1]
      result$arm <- value$arm[1]
      result$treatment <- value$treatment[1]
      result$family <- value$family[1]
      result$truth <- value$truth[1]
      result
    }
  ))
} else {
  data.frame()
}
if (nrow(coverage_summary)) rownames(coverage_summary) <- NULL
print(coverage_summary, row.names = FALSE)

coverage_suffix <- if (exploratory_run) "_exploratory" else suffix
write.csv(coverage_summary,
          paste0("R/results/stage15_count", coverage_suffix, ".csv"),
          row.names = FALSE)
write.csv(coverage_replications,
          paste0("R/results/stage15_count_replications", coverage_suffix,
                 ".csv"),
          row.names = FALSE)

if (full_settings) {
  primary <- coverage_summary[
    coverage_summary$arm == "xgb_constrained" &
      coverage_summary$estimator %in% c("rao_blackwell", "randomized"),
    , drop = FALSE
  ]
  if (!exploratory_run) stopifnot(nrow(primary) == 2L * nrow(cells))
  gates <- count_primary_gates(primary)
  if (exploratory_run) {
    cat("EXPLORATORY COVERAGE COMPLETE: constrained XGBoost was run despite",
        "failing its point-screen promotion gate; no count or bikeshare",
        "promotion is authorized\n")
  } else if (!all(gates$pass)) {
    stop("constrained XGBoost failed at least one primary count gate; ",
         "diagnostic result files were retained and no promotion is authorized")
  } else {
    cat("PASS: constrained XGBoost Rao-Blackwell and randomized SUDO satisfy",
        "every primary count gate\n")
  }
} else {
  cat("SMOKE PASS: statistical acceptance requires the documented defaults",
      "and a frozen full tuning result\n")
}
