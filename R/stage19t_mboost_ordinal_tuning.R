# Stage 19t: target-level tuning for the mboost ordinal learner.
#
# Pure PropOdds boosting is evaluated first. The unpenalized-treatment
# backfit runs on fresh seeds only if no pure configuration promotes. Smoke
# mode runs one backfit configuration regardless, to exercise both paths.

source("R/stage19_mboost_support.R")
source("R/sudo/mc.R")

stage19_require_packages()
stage19_unit_checks()
smoke <- nzchar(Sys.getenv("SUDO_STAGE19_SMOKE", unset = ""))
n <- stage19_env_int("SUDO_STAGE19_N", if (smoke) 400L else 3000L)
p <- stage19_env_int("SUDO_STAGE19_P", if (smoke) 8L else 50L)
n_reps <- stage19_env_int("SUDO_STAGE19_REPS", if (smoke) 2L else 20L)
n_folds <- stage19_env_int("SUDO_STAGE19_FOLDS", if (smoke) 2L else 5L)
cores <- stage19_env_int("SUDO_STAGE19_CORES", if (smoke) 1L else 4L)
nuisance_mstop <- stage19_env_int(
  "SUDO_STAGE19_NUISANCE_MSTOP", if (smoke) 100L else 2000L
)
pure_grid <- if (smoke) c(100L, 300L) else
  c(500L, 1000L, 2000L, 5000L, 10000L)
backfit_grid <- if (smoke) 100L else c(500L, 1000L, 2000L)

stage19_tuning_estimate <- function(d, folds, full, config, D_res) {
  X <- as.data.frame(d$X)
  surrogate <- full$index +
    stage19_logistic_jump_mean(full$lower, full$upper)
  outcome_fitter <- stage19_nuisance_fitter("outcome", config)
  nuisance <- crossfit(X, surrogate, outcome_fitter, folds)
  sudo <- fwl_theta(surrogate - nuisance, D_res)
  c(
    theta_hat = sudo$theta,
    direct = full$direct_theta,
    min_threshold_gap = full$diagnostics$min_threshold_gap,
    max_contrast_spread = full$diagnostics$max_contrast_spread,
    max_probability_error = full$diagnostics$max_probability_error,
    backfit_iterations = full$diagnostics$backfit_iterations,
    backfit_converged = full$diagnostics$backfit_converged,
    max_condition_number = if (is.null(
      full$diagnostics$max_condition_number
    )) NA_real_ else full$diagnostics$max_condition_number
  )
}

stage19_tuning_rep <- function(replication, method, mstop_grid, seed_start) {
  seed <- seed_start + replication
  set.seed(seed)
  d <- stage19_dgp(n, p, 1)
  folds <- stage19_make_folds(n, n_folds, seed + 100000L)
  X <- as.data.frame(d$X)
  common_config <- stage19_config(
    method, mstop = max(mstop_grid), nuisance_mstop = nuisance_mstop,
    backfit_tolerance = if (smoke) 1e-3 else 1e-4
  )
  treatment_fitter <- stage19_nuisance_fitter(
    "treatment", common_config
  )
  D_res <- d$D - crossfit(X, d$D, treatment_fitter, folds)
  path <- if (method == "pure") try(stage19_fit_pure_path(
    d, folds, mstop_grid, common_config$nu,
    common_config$knots, common_config$df
  ), silent = TRUE) else NULL

  rows <- lapply(mstop_grid, function(mstop) {
    config <- stage19_config(
      method, mstop = mstop, nuisance_mstop = nuisance_mstop,
      backfit_tolerance = if (smoke) 1e-3 else 1e-4
    )
    full <- if (method == "pure") {
      if (inherits(path, "try-error")) path else path[[as.character(mstop)]]
    } else {
      try(stage19_fit_backfit(d, folds, config), silent = TRUE)
    }
    estimate <- if (inherits(full, "try-error")) full else try(
      stage19_tuning_estimate(d, folds, full, config, D_res),
      silent = TRUE
    )
    if (inherits(estimate, "try-error")) {
      data.frame(
        method = method, mstop = mstop, theta_hat = NA_real_,
        direct = NA_real_, min_threshold_gap = NA_real_,
        max_contrast_spread = NA_real_, max_probability_error = NA_real_,
        backfit_iterations = NA_real_, backfit_converged = 0,
        max_condition_number = NA_real_, error = as.character(estimate),
        stringsAsFactors = FALSE
      )
    } else {
      data.frame(
        method = method, mstop = mstop,
        theta_hat = unname(estimate["theta_hat"]),
        direct = unname(estimate["direct"]),
        min_threshold_gap = unname(estimate["min_threshold_gap"]),
        max_contrast_spread = unname(estimate["max_contrast_spread"]),
        max_probability_error = unname(estimate["max_probability_error"]),
        backfit_iterations = unname(estimate["backfit_iterations"]),
        backfit_converged = unname(estimate["backfit_converged"]),
        max_condition_number = unname(estimate["max_condition_number"]),
        error = "", stringsAsFactors = FALSE
      )
    }
  })
  out <- do.call(rbind, rows)
  out$replication <- replication
  out$seed <- seed
  out$sample_signature <- sum(d$Y * seq_along(d$Y))
  out$fold_signature <- sum(vapply(seq_along(folds), function(k)
    sum(folds[[k]]) * k, numeric(1)))
  out
}

stage19_run_arm <- function(method, mstop_grid, seed_start, cluster) {
  worker <- function(replication) stage19_tuning_rep(
    replication, method, mstop_grid, seed_start
  )
  if (!is.null(cluster)) {
    parallel::clusterExport(
      cluster, c("method", "mstop_grid", "seed_start", "worker"),
      envir = environment()
    )
  }
  out <- do.call(rbind, if (is.null(cluster)) {
    lapply(seq_len(n_reps), worker)
  } else {
    parallel::parLapplyLB(cluster, seq_len(n_reps), worker)
  })
  cat(sprintf("%s completed %d tuning replications\n", method, n_reps))
  out
}

stage19_summarize <- function(replications) {
  blocks <- split(
    replications, list(replications$method, replications$mstop), drop = TRUE
  )
  out <- do.call(rbind, lapply(blocks, function(block) {
    ok <- is.finite(block$theta_hat)
    error <- block$theta_hat[ok] - 1
    finite_max <- function(value) {
      value <- value[is.finite(value)]
      if (length(value)) max(value) else NA_real_
    }
    finite_min <- function(value) {
      value <- value[is.finite(value)]
      if (length(value)) min(value) else NA_real_
    }
    data.frame(
      method = block$method[1], mstop = block$mstop[1],
      n_success = sum(ok), bias = if (sum(ok)) mean(error) else NA_real_,
      mc_se_bias = if (sum(ok) > 1L) sd(error) / sqrt(sum(ok)) else NA_real_,
      sd = if (sum(ok) > 1L) sd(block$theta_hat[ok]) else NA_real_,
      direct_bias = if (sum(ok)) mean(block$direct[ok]) - 1 else NA_real_,
      fail_rate = mean(!ok),
      min_threshold_gap = finite_min(block$min_threshold_gap[ok]),
      max_contrast_spread = finite_max(block$max_contrast_spread[ok]),
      max_probability_error = finite_max(block$max_probability_error[ok]),
      max_backfit_iterations = finite_max(block$backfit_iterations[ok]),
      max_condition_number = finite_max(block$max_condition_number[ok]),
      stringsAsFactors = FALSE
    )
  }))
  rownames(out) <- NULL
  out$bias_limit <- pmax(2 * out$mc_se_bias, 0.03)
  out$numerically_eligible <-
    out$fail_rate < 0.02 & out$min_threshold_gap > 0.05 &
    out$max_contrast_spread < 1e-8 &
    out$max_probability_error < 1e-6
  out$bias_ok <- abs(out$bias) <= out$bias_limit
  out$promotion_pass <- out$numerically_eligible & out$bias_ok
  out
}

cluster <- if (cores > 1L) mc_cluster(n_cores = cores) else NULL
if (!is.null(cluster)) {
  parallel::clusterEvalQ(cluster, source("R/stage19_mboost_support.R"))
  parallel::clusterExport(
    cluster,
    c("n", "p", "n_reps", "n_folds", "nuisance_mstop", "smoke",
      "stage19_tuning_estimate", "stage19_tuning_rep"),
    envir = environment()
  )
}

pure_replications <- stage19_run_arm(
  "pure", pure_grid, 192000L, cluster
)
pure_summary <- stage19_summarize(pure_replications)
pure_pass <- any(pure_summary$promotion_pass, na.rm = TRUE)
run_backfit <- smoke || !pure_pass
backfit_replications <- if (run_backfit) stage19_run_arm(
  "backfit", backfit_grid, 193000L, cluster
) else NULL
if (!is.null(cluster)) parallel::stopCluster(cluster)
backfit_summary <- if (is.null(backfit_replications)) NULL else
  stage19_summarize(backfit_replications)

replications <- if (is.null(backfit_replications)) pure_replications else
  rbind(pure_replications, backfit_replications)
summary <- if (is.null(backfit_summary)) pure_summary else
  rbind(pure_summary, backfit_summary)
eligible <- summary[summary$promotion_pass, , drop = FALSE]
if (nrow(eligible)) {
  selected_method <- if (any(eligible$method == "pure")) "pure" else "backfit"
  eligible <- eligible[eligible$method == selected_method, , drop = FALSE]
  selected <- eligible[order(abs(eligible$bias), eligible$mstop), , drop = FALSE][1, ]
} else {
  selected <- summary[order(
    !summary$numerically_eligible, abs(summary$bias), summary$mstop
  ), , drop = FALSE][1, ]
}
selected$nu <- 0.1
selected$nuisance_mstop <- nuisance_mstop
selected$nuisance_nu <- 0.1
selected$knots <- 10L
selected$df <- 4
selected$backfit_tolerance <- if (smoke) 1e-3 else 1e-4
selected$backfit_max_iterations <- 10L
selected$screen_n <- n
selected$screen_p <- p
selected$screen_reps <- n_reps
selected$screen_folds <- n_folds
selected$mboost_version <- as.character(utils::packageVersion("mboost"))
full_run <- !smoke && n >= 3000L && p >= 50L && n_reps >= 20L &&
  n_folds >= 5L && identical(pure_grid, c(500L, 1000L, 2000L, 5000L,
                                           10000L))
selected$confirmatory_ready <- full_run && isTRUE(selected$promotion_pass)

print(summary, row.names = FALSE)
cat("\nSelected configuration:\n")
print(selected, row.names = FALSE)

dir.create("R/results", showWarnings = FALSE)
suffix <- if (full_run) "" else "_smoke"
write.csv(
  replications,
  paste0("R/results/stage19t_mboost_ordinal_replications", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  summary,
  paste0("R/results/stage19t_mboost_ordinal_tuning", suffix, ".csv"),
  row.names = FALSE
)
write.csv(
  selected,
  paste0("R/results/stage19t_mboost_ordinal_config", suffix, ".csv"),
  row.names = FALSE
)

if (full_run) {
  if (!isTRUE(selected$promotion_pass)) {
    stop("neither pure nor backfit mboost passed target-level promotion")
  }
  cat("PASS: stage-19 mboost configuration is frozen for confirmation\n")
} else {
  cat("SMOKE PASS: statistical promotion requires the documented defaults\n")
}
