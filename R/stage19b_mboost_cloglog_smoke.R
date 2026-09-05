# Stage 19b: native mboost cloglog compatibility smoke.
#
# This is a focused binary check, not a second coverage experiment. It verifies
# the glm-scale cloglog family, the partially-linear treatment contrast, the
# randomized-PIT bounds, and Gumbel-max surrogate completion.

source("R/stage19_mboost_support.R")
source("R/sudo/mc.R")

stage19_require_packages()
stage19_unit_checks()
small <- nzchar(Sys.getenv("SUDO_STAGE19B_SMOKE", unset = ""))
n <- stage19_env_int("SUDO_STAGE19B_N", if (small) 400L else 2000L)
p <- stage19_env_int("SUDO_STAGE19B_P", if (small) 8L else 50L)
n_reps <- stage19_env_int("SUDO_STAGE19B_REPS", if (small) 2L else 10L)
n_folds <- stage19_env_int("SUDO_STAGE19B_FOLDS", if (small) 2L else 5L)
B <- stage19_env_int("SUDO_STAGE19B_B", if (small) 2L else 3L)
cores <- stage19_env_int("SUDO_STAGE19B_CORES", if (small) 1L else 4L)
nuisance_mstop <- stage19_env_int(
  "SUDO_STAGE19B_NUISANCE_MSTOP", if (small) 100L else 2000L
)
mstop_grid <- if (small) c(100L, 300L) else
  c(500L, 1000L, 2000L, 5000L, 10000L)
completion_mstop <- if (small) 100L else 2000L

stage19_binary_rep <- function(replication) {
  seed <- 195000L + replication
  set.seed(seed)
  d <- stage19_binary_dgp(n, p, theta = 1)
  folds <- stage19_make_folds(n, n_folds, seed + 100000L)
  path <- try(stage19_fit_binary_path(
    d, folds, mstop_grid, nu = 0.1, knots = 10L, df = 4
  ), silent = TRUE)
  if (inherits(path, "try-error")) {
    return(data.frame(
      replication = replication, seed = seed, mstop = mstop_grid,
      direct_cloglog = NA_real_, sudo_cloglog = NA_real_,
      sudo_internal_se = NA_real_, max_contrast_spread = NA_real_,
      max_probability_violation = NA_real_, min_jump_width = NA_real_,
      completion_finite = 0, error = as.character(path),
      stringsAsFactors = FALSE
    ))
  }
  config <- stage19_config(
    "pure", mstop = completion_mstop, nuisance_mstop = nuisance_mstop
  )
  completion <- try(stage19_binary_theta_from_full(
    d, folds, path[[as.character(completion_mstop)]], config, B
  ), silent = TRUE)
  completion_error <- if (inherits(completion, "try-error")) {
    as.character(completion)
  } else ""
  do.call(rbind, lapply(mstop_grid, function(mstop) {
    full <- path[[as.character(mstop)]]
    selected <- mstop == completion_mstop && !inherits(completion, "try-error")
    data.frame(
      replication = replication, seed = seed, mstop = mstop,
      direct_cloglog = full$direct_theta,
      sudo_cloglog = if (selected) completion$theta else NA_real_,
      sudo_internal_se = if (selected) completion$se else NA_real_,
      max_contrast_spread = full$diagnostics$max_contrast_spread,
      max_probability_violation =
        full$diagnostics$max_probability_violation,
      min_jump_width = full$diagnostics$min_jump_width,
      completion_finite = as.numeric(
        selected && all(is.finite(completion$draws))
      ),
      error = if (mstop == completion_mstop) completion_error else "",
      stringsAsFactors = FALSE
    )
  }))
}

cluster <- if (cores > 1L) mc_cluster(n_cores = cores) else NULL
if (!is.null(cluster)) {
  parallel::clusterEvalQ(cluster, source("R/stage19_mboost_support.R"))
  parallel::clusterExport(
    cluster,
    c("n", "p", "n_folds", "B", "mstop_grid", "completion_mstop",
      "nuisance_mstop", "stage19_binary_rep"), envir = environment()
  )
}
worker <- function(replication) stage19_binary_rep(replication)
replications <- do.call(rbind, if (is.null(cluster)) {
  lapply(seq_len(n_reps), worker)
} else {
  parallel::parLapplyLB(cluster, seq_len(n_reps), worker)
})
if (!is.null(cluster)) parallel::stopCluster(cluster)

blocks <- split(replications, replications$mstop)
summary <- do.call(rbind, lapply(blocks, function(block) {
  direct_ok <- is.finite(block$direct_cloglog)
  sudo_ok <- is.finite(block$sudo_cloglog)
  direct_error <- block$direct_cloglog[direct_ok] - 1
  sudo_error <- block$sudo_cloglog[sudo_ok] - 1
  data.frame(
    mstop = block$mstop[1], direct_success = sum(direct_ok),
    direct_bias = if (sum(direct_ok)) mean(direct_error) else NA_real_,
    direct_mc_se = if (sum(direct_ok) > 1L) {
      stats::sd(direct_error) / sqrt(sum(direct_ok))
    } else NA_real_,
    sudo_success = sum(sudo_ok),
    sudo_bias = if (sum(sudo_ok)) mean(sudo_error) else NA_real_,
    sudo_mc_se = if (sum(sudo_ok) > 1L) {
      stats::sd(sudo_error) / sqrt(sum(sudo_ok))
    } else NA_real_,
    max_contrast_spread = max(block$max_contrast_spread, na.rm = TRUE),
    max_probability_violation = max(
      block$max_probability_violation, na.rm = TRUE
    ),
    min_jump_width = min(block$min_jump_width, na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}))
rownames(summary) <- NULL
print(summary, row.names = FALSE)

dir.create("R/results", showWarnings = FALSE)
suffix <- if (small) "_smoke" else ""
write.csv(
  replications,
  paste0("R/results/stage19b_mboost_cloglog_replications", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  summary,
  paste0("R/results/stage19b_mboost_cloglog", suffix, ".csv"),
  row.names = FALSE
)

numerical_ok <-
  all(is.finite(replications$direct_cloglog)) &&
  all(replications$max_contrast_spread < 1e-8) &&
  all(replications$max_probability_violation < 1e-12) &&
  all(replications$min_jump_width > 0) &&
  all(replications$completion_finite[
    replications$mstop == completion_mstop
  ] == 1)
stopifnot(numerical_ok)
full_run <- !small && n >= 2000L && p >= 50L && n_reps >= 10L &&
  n_folds >= 5L && B >= 3L
if (full_run) {
  direct_ok <- with(summary, abs(direct_bias) <= pmax(2 * direct_mc_se, 0.08))
  selected <- summary[summary$mstop == completion_mstop, , drop = FALSE]
  sudo_ok <- abs(selected$sudo_bias) <= max(2 * selected$sudo_mc_se, 0.08)
  stopifnot(any(direct_ok), sudo_ok)
  cat("PASS: native mboost cloglog path and completion are compatible\n")
} else {
  cat("SMOKE PASS: statistical acceptance requires the documented defaults\n")
}
