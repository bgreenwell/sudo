# Stage 19: mboost proportional-odds ordinal SUDO confirmation.
#
# Run stage19t_mboost_ordinal_tuning.R first. The selected learner is frozen
# before this script evaluates full-pipeline coverage and the anchored-score
# DML negative controls.

source("R/stage19_mboost_support.R")
source("R/sudo/mc.R")

stage19_require_packages()
stage19_unit_checks()
smoke <- nzchar(Sys.getenv("SUDO_STAGE19_SMOKE", unset = ""))
n <- stage19_env_int("SUDO_STAGE19_N", if (smoke) 400L else 3000L)
p <- stage19_env_int("SUDO_STAGE19_P", if (smoke) 8L else 50L)
n_reps <- stage19_env_int("SUDO_STAGE19_REPS", if (smoke) 2L else 50L)
bootstrap_reps <- stage19_env_int(
  "SUDO_STAGE19_BOOTSTRAPS", if (smoke) 2L else 99L
)
n_folds <- stage19_env_int("SUDO_STAGE19_FOLDS", if (smoke) 2L else 5L)
cores <- stage19_env_int("SUDO_STAGE19_CORES", if (smoke) 1L else 4L)
chunk_size <- stage19_env_int(
  "SUDO_STAGE19_CHUNK_SIZE", if (smoke) 1L else cores
)

config_file <- paste0(
  "R/results/stage19t_mboost_ordinal_config",
  if (smoke) "_smoke" else "", ".csv"
)
if (!file.exists(config_file)) {
  stop("run stage-19t tuning before confirmation")
}
selected <- read.csv(config_file, stringsAsFactors = FALSE)
stopifnot(nrow(selected) == 1L)
if (!smoke && !isTRUE(selected$confirmatory_ready)) {
  stop("full stage-19 tuning did not promote a confirmatory configuration")
}
config <- stage19_config(
  method = selected$method, mstop = selected$mstop, nu = selected$nu,
  nuisance_mstop = selected$nuisance_mstop,
  nuisance_nu = selected$nuisance_nu, knots = selected$knots,
  df = selected$df, backfit_tolerance = selected$backfit_tolerance,
  backfit_max_iterations = selected$backfit_max_iterations
)
config_id <- paste(
  "stage19_v3_resumable_primary_boot", config$method, config$mstop,
  config$nuisance_mstop,
  config$nu, config$nuisance_nu, config$knots, config$df,
  as.character(utils::packageVersion("mboost")), sep = "_"
)
state_dir <- paste0(
  "R/results/stage19_mboost_ordinal_bootstrap_state",
  if (smoke) "_smoke" else ""
)
dir.create(state_dir, recursive = TRUE, showWarnings = FALSE)

stage19_confirmation_rep <- function(replication) {
  seed <- 194000L + replication
  set.seed(seed)
  d <- stage19_dgp(n, p, theta = 1)
  base_folds <- stage19_make_folds(n, n_folds, seed + 100000L)
  bootstrap_indices <- lapply(seq_len(bootstrap_reps), function(b) {
    sample.int(n, replace = TRUE)
  })
  bootstrap_folds <- lapply(seq_len(bootstrap_reps), function(b) {
    stage19_make_folds(n, n_folds, seed + 200000L + b)
  })
  state_file <- file.path(
    state_dir, sprintf("replication_%03d.rds", replication)
  )
  state_id <- paste(
    config_id, n, p, n_folds, bootstrap_reps, seed, sep = "_"
  )
  fit <- try(stage19_pipeline_boot_resumable(
    d, config, state_file = state_file, state_id = state_id,
    B_outer = bootstrap_reps, n_folds = n_folds,
    base_folds = base_folds, bootstrap_indices = bootstrap_indices,
    bootstrap_folds = bootstrap_folds
  ), silent = TRUE)
  empty <- function(error) data.frame(
    replication = replication, seed = seed, rao_blackwell = NA_real_,
    rao_blackwell_se = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_,
    direct_ordinal = NA_real_, dml_equal = NA_real_, dml_middle = NA_real_,
    coding_difference = NA_real_, coding_difference_se = NA_real_,
    oracle_dml_equal = NA_real_, oracle_dml_middle = NA_real_,
    oracle_coding_difference = NA_real_, oracle_latent_plr = NA_real_,
    thresholds_ordered = NA_real_, all_boot_thresholds_ordered = NA_real_,
    min_threshold_gap = NA_real_, min_boot_threshold_gap = NA_real_,
    max_contrast_spread = NA_real_, max_probability_error = NA_real_,
    backfit_iterations = NA_real_, max_condition_number = NA_real_,
    min_category_frequency = min(prop.table(table(d$Y))),
    bootstrap_success = 0, error = error, stringsAsFactors = FALSE
  )
  if (inherits(fit, "try-error")) return(empty(as.character(fit)))
  results <- fit$target_results
  get <- function(target, column) results[[column]][results$target == target]
  data.frame(
    replication = replication, seed = seed,
    rao_blackwell = fit$theta, rao_blackwell_se = fit$se,
    ci_lo = fit$ci_lo, ci_hi = fit$ci_hi,
    direct_ordinal = get("direct_ordinal", "estimate"),
    dml_equal = fit$dml_equal,
    dml_middle = fit$dml_middle,
    coding_difference = fit$dml_middle - fit$dml_equal,
    coding_difference_se = NA_real_,
    oracle_dml_equal = fit$oracle_dml_equal,
    oracle_dml_middle = fit$oracle_dml_middle,
    oracle_coding_difference = fit$oracle_dml_middle - fit$oracle_dml_equal,
    oracle_latent_plr = fit$oracle_latent_plr,
    thresholds_ordered = fit$thresholds_ordered,
    all_boot_thresholds_ordered = fit$all_boot_thresholds_ordered,
    min_threshold_gap = fit$min_threshold_gap,
    min_boot_threshold_gap = fit$min_boot_threshold_gap,
    max_contrast_spread = fit$max_contrast_spread,
    max_probability_error = fit$max_probability_error,
    backfit_iterations = fit$backfit_iterations,
    max_condition_number = if (is.null(fit$max_condition_number)) {
      NA_real_
    } else fit$max_condition_number,
    min_category_frequency = min(prop.table(table(d$Y))),
    bootstrap_success = 1, error = "", stringsAsFactors = FALSE
  )
}

full_run <- !smoke && n >= 3000L && p >= 50L && n_reps >= 50L &&
  bootstrap_reps >= 99L && n_folds >= 5L
suffix <- if (full_run) "" else "_smoke"
dir.create("R/results", showWarnings = FALSE)
checkpoint_file <- paste0(
  "R/results/stage19_mboost_ordinal_checkpoint", suffix, ".csv"
)

checkpoint <- if (file.exists(checkpoint_file)) {
  read.csv(checkpoint_file, stringsAsFactors = FALSE)
} else data.frame()
if (nrow(checkpoint)) {
  required <- c(
    "config", "n", "p", "n_folds", "bootstrap_reps", "replication"
  )
  metadata_match <- all(required %in% names(checkpoint)) &&
    all(checkpoint$config == config_id) && all(checkpoint$n == n) &&
    all(checkpoint$p == p) && all(checkpoint$n_folds == n_folds) &&
    all(checkpoint$bootstrap_reps == bootstrap_reps)
  if (!metadata_match) checkpoint <- data.frame()
}
completed <- if (nrow(checkpoint)) {
  unique(checkpoint$replication[checkpoint$replication <= n_reps])
} else integer()
remaining <- setdiff(seq_len(n_reps), completed)
chunks <- split(remaining, ceiling(seq_along(remaining) / max(1L, chunk_size)))

cluster <- if (cores > 1L) mc_cluster(n_cores = cores) else NULL
if (!is.null(cluster)) {
  parallel::clusterEvalQ(cluster, source("R/stage19_mboost_support.R"))
  parallel::clusterExport(
      cluster,
    c("n", "p", "n_folds", "bootstrap_reps", "config",
      "config_id", "state_dir", "stage19_confirmation_rep"),
    envir = environment()
  )
}
for (chunk in chunks) {
  added <- do.call(rbind, if (is.null(cluster)) {
    lapply(chunk, stage19_confirmation_rep)
  } else {
    parallel::parLapplyLB(cluster, chunk, stage19_confirmation_rep)
  })
  added$config <- config_id
  added$n <- n
  added$p <- p
  added$n_folds <- n_folds
  added$bootstrap_reps <- bootstrap_reps
  checkpoint <- rbind(checkpoint, added)
  write.csv(checkpoint, checkpoint_file, row.names = FALSE)
  cat(sprintf(
    "stage-19 confirmation checkpointed %d/%d replications\n",
    length(unique(checkpoint$replication[checkpoint$replication <= n_reps])),
    n_reps
  ))
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
replications <- checkpoint[checkpoint$replication <= n_reps, , drop = FALSE]

summarize_estimator <- function(estimator, with_se = FALSE) {
  value <- replications[[estimator]]
  ok <- is.finite(value)
  error <- value[ok] - 1
  coverage <- if (with_se) {
    mean(replications$ci_lo[ok] <= 1 & replications$ci_hi[ok] >= 1)
  } else NA_real_
  data.frame(
    estimator = estimator, n_success = sum(ok),
    bias = if (sum(ok)) mean(error) else NA_real_,
    mc_se_bias = if (sum(ok) > 1L) stats::sd(error) / sqrt(sum(ok)) else
      NA_real_,
    sd = if (sum(ok) > 1L) stats::sd(value[ok]) else NA_real_,
    rmse = if (sum(ok)) sqrt(mean(error^2)) else NA_real_,
    mean_se = if (with_se && sum(ok)) {
      mean(replications$rao_blackwell_se[ok])
    } else NA_real_,
    coverage = coverage,
    mc_se_coverage = if (with_se && sum(ok)) {
      sqrt(coverage * (1 - coverage) / sum(ok))
    } else NA_real_,
    fail_rate = mean(!ok), stringsAsFactors = FALSE
  )
}
summary <- do.call(rbind, lapply(
  c("rao_blackwell", "direct_ordinal", "dml_equal", "dml_middle",
    "oracle_dml_equal", "oracle_dml_middle", "oracle_latent_plr"),
  function(estimator) summarize_estimator(
    estimator, estimator == "rao_blackwell"
  )
))
rownames(summary) <- NULL
print(summary, row.names = FALSE)
write.csv(
  replications,
  paste0("R/results/stage19_mboost_ordinal_replications", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  summary, paste0("R/results/stage19_mboost_ordinal", suffix, ".csv"),
  row.names = FALSE
)

if (full_run) {
  row <- function(name) summary[summary$estimator == name, , drop = FALSE]
  primary <- row("rao_blackwell")
  primary_bias_ok <- abs(primary$bias) <= max(
    2 * primary$mc_se_bias, 0.03
  )
  coverage_ok <- abs(primary$coverage - 0.95) <= max(
    2 * primary$mc_se_coverage, 0.04
  )
  ratio <- primary$sd / primary$mean_se
  ratio_mc_se <- ratio * sqrt(1 / (2 * (primary$n_success - 1)))
  variance_ok <- abs(ratio - 1) <= 2 * ratio_mc_se
  positive <- summary[summary$estimator %in%
    c("direct_ordinal", "oracle_latent_plr"), , drop = FALSE]
  positive_ok <- abs(positive$bias) <= pmax(
    2 * positive$mc_se_bias, 0.03
  )
  negative <- summary[summary$estimator %in%
    c("dml_equal", "dml_middle", "oracle_dml_equal", "oracle_dml_middle"),
    , drop = FALSE]
  negative_ok <- abs(negative$bias) >= pmax(
    0.15, 4 * negative$mc_se_bias
  )
  difference_ok <- function(value) {
    value <- value[is.finite(value)]
    abs(mean(value)) >= max(0.10, 4 * stats::sd(value) / sqrt(length(value)))
  }
  structure_ok <-
    mean(replications$bootstrap_success) >= 0.98 &&
    all(replications$thresholds_ordered == 1, na.rm = TRUE) &&
    all(replications$all_boot_thresholds_ordered == 1, na.rm = TRUE) &&
    all(replications$min_threshold_gap > 0.05, na.rm = TRUE) &&
    all(replications$min_boot_threshold_gap > 0.05, na.rm = TRUE) &&
    all(replications$max_contrast_spread < 1e-8, na.rm = TRUE) &&
    all(replications$max_probability_error < 1e-6, na.rm = TRUE) &&
    all(replications$min_category_frequency > 0.05, na.rm = TRUE)
  stopifnot(
    primary_bias_ok, coverage_ok, variance_ok, all(positive_ok),
    all(negative_ok), difference_ok(replications$coding_difference),
    difference_ok(replications$oracle_coding_difference), structure_ok
  )
  cat("PASS: mboost ordinal SUDO satisfies every stage-19 gate\n")
} else {
  cat("SMOKE PASS: statistical acceptance requires the documented defaults\n")
}
