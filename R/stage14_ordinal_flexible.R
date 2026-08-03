# Stage 14: flexible proportional-odds imputation for ordinal outcomes.
#
# mboost::PropOdds supplies eta(D,X)=theta*D+f(X), global ordered thresholds,
# and held-out category probabilities. Tuning is target-level and uses seeds
# disjoint from confirmation. Predictive risk is reported only as a
# diagnostic. The frozen default was selected from the target-level smoke
# screen and must be reselected with SUDO_STAGE14_TUNE=1 before a new full
# confirmatory run changes the configuration.
#
# Full confirmation: J in {3,5}, n=2000, 100 replications, 99 outer
# resamples, B=25. Smoke example:
#   SUDO_STAGE14_N=500 SUDO_STAGE14_REPS=2 SUDO_STAGE14_OUTER=2 \
#   SUDO_STAGE14_B=3 SUDO_STAGE14_MSTOP=300 Rscript R/stage14_ordinal_flexible.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/discrete.R")
source("R/sudo/estimator.R")
source("R/sudo/mc.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}
n <- env_int("SUDO_STAGE14_N", 2000L)
n_reps <- env_int("SUDO_STAGE14_REPS", 100L)
B_outer <- env_int("SUDO_STAGE14_OUTER", 99L)
B <- env_int("SUDO_STAGE14_B", 25L)
n_folds <- env_int("SUDO_STAGE14_FOLDS", 5L)
n_cores <- env_int("SUDO_STAGE14_CORES", 6L)
config_file <- "R/results/stage14t_ordinal_config.csv"
config <- if (file.exists(config_file)) read.csv(config_file) else NULL
config_ready <- !is.null(config) && nrow(config) == 1L &&
  isTRUE(config$confirmatory_ready[1])
mstop_default <- if (config_ready) config$mstop[1] else 2000L
nu_default <- if (config_ready) config$nu[1] else 0.5
mstop <- env_int("SUDO_STAGE14_MSTOP", mstop_default)
nu <- as.numeric(Sys.getenv("SUDO_STAGE14_NU", unset = as.character(nu_default)))
theta <- 1
oracle_n <- env_int("SUDO_STAGE14_ORACLE_N", 2000L)
oracle_mstop <- env_int("SUDO_STAGE14_ORACLE_MSTOP", 2000L)
oracle_nu <- as.numeric(Sys.getenv("SUDO_STAGE14_ORACLE_NU", unset = "0.5"))
j_value <- Sys.getenv("SUDO_STAGE14_J", unset = "")
j_grid <- if (nzchar(j_value)) as.integer(j_value) else c(3L, 5L)

dgp14 <- function(n, J) {
  X <- matrix(rnorm(n * 5), n, 5,
              dimnames = list(NULL, paste0("X", seq_len(5))))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  eta <- theta * D + g
  cuts <- if (J == 3L) c(0, 2) else c(-1, 0.5, 1.75, 3)
  Y <- 1L + findInterval(eta + rlogis(n), cuts)
  list(X = X, D = D, Y = Y, eta = eta, cuts = cuts)
}

oracle_diagnostic <- function(J) {
  set.seed(14000L + J)
  d <- dgp14(oracle_n, J)
  fit <- fit_ordinal_propodds_adapter(
    d, make_folds(length(d$Y), 5L), mstop = oracle_mstop, nu = oracle_nu
  )
  shift <- mean(fit$index - d$eta)
  data.frame(
    J = J, direct_theta = fit$direct_theta,
    treatment_error = fit$direct_theta - theta,
    centered_index_rmse = sqrt(mean((fit$index - d$eta - shift)^2)),
    index_correlation = cor(fit$index, d$eta),
    thresholds_ordered = fit$diagnostics$thresholds_ordered,
    min_threshold_gap = fit$diagnostics$min_threshold_gap
  )
}
oracle <- do.call(rbind, lapply(j_grid, oracle_diagnostic))
print(oracle, row.names = FALSE)
stopifnot(all(oracle$thresholds_ordered == 1),
          all(abs(oracle$treatment_error) < if (oracle_n >= 2000L) 0.25 else 0.8),
          all(oracle$centered_index_rmse < 1.2),
          all(oracle$index_correlation > 0.75))
cat("PASS: PropOdds smoke fits are accurate enough to the oracle index and",
    "keep every threshold vector ordered\n")
dir.create("R/results", showWarnings = FALSE)
oracle_file <- "R/results/stage14_ordinal_oracle_smoke.csv"
if (Sys.getenv("SUDO_STAGE14_APPEND_ORACLE", unset = "") == "1" &&
    file.exists(oracle_file)) {
  previous <- read.csv(oracle_file)
  oracle <- rbind(previous[!previous$J %in% oracle$J, ], oracle)
}
if (Sys.getenv("SUDO_STAGE14_SKIP_ORACLE_WRITE", unset = "") != "1") {
  write.csv(oracle, oracle_file, row.names = FALSE)
}
if (Sys.getenv("SUDO_STAGE14_SKIP_CONFIRM", unset = "") == "1") {
  cat("SKIP: confirmatory Monte Carlo was not requested\n")
  quit(save = "no", status = 0)
}

one_rep <- function(seed, n, J, B_outer, B, mstop, nu, n_folds) {
  set.seed(seed)
  d <- dgp14(n, J)
  adapter <- make_ordinal_propodds_adapter(mstop, nu)
  fit <- sudo_pipeline_boot(
    d, B_outer = B_outer, inner_B = B, n_folds = n_folds,
    estimator = sudo_ordinal, adapter = adapter
  )
  data.frame(theta = fit$theta, se = fit$se,
             direct = fit$direct_theta,
             thresholds_ordered = fit$all_boot_thresholds_ordered,
             min_threshold_gap = fit$min_boot_threshold_gap)
}

rows <- list()
cluster <- if (n_cores > 1L) mc_cluster(
  c("dgp14", "one_rep", "theta"), n_cores = n_cores
) else NULL
for (J in j_grid) {
  worker <- function(r) {
    one_rep(140000L + J * 10000L + r, n, J, B_outer, B, mstop, nu,
            n_folds)
  }
  if (!is.null(cluster)) {
    parallel::clusterExport(
      cluster,
      c("J", "n", "B_outer", "B", "mstop", "nu", "n_folds", "worker"),
      envir = environment()
    )
  }
  replications <- do.call(rbind, if (is.null(cluster))
    lapply(seq_len(n_reps), worker) else
    parallel::parLapplyLB(cluster, seq_len(n_reps), worker))
  stopifnot(all(replications$thresholds_ordered == 1),
            all(replications$min_threshold_gap > 0))
  for (estimator in c("sudo", "direct")) {
    values <- if (estimator == "sudo") replications$theta else
      replications$direct
    sd_value <- sd(values)
    is_sudo <- estimator == "sudo"
    coverage <- if (is_sudo)
      mean(abs(values - theta) <= 1.96 * replications$se) else NA_real_
    rows[[length(rows) + 1L]] <- data.frame(
      J = J, estimator = estimator, n_reps = n_reps,
      bias = mean(values) - theta, mc_se_bias = sd_value / sqrt(n_reps),
      sd = sd_value,
      mean_se = if (is_sudo) mean(replications$se) else NA_real_,
      coverage = coverage,
      mc_se_coverage = if (is_sudo) sqrt(coverage * (1 - coverage) / n_reps)
      else NA_real_,
      all_thresholds_ordered = all(replications$thresholds_ordered == 1),
      min_threshold_gap = min(replications$min_threshold_gap)
    )
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
summary <- do.call(rbind, rows)
print(format(summary, digits = 4), row.names = FALSE)
full_run <- n >= 2000L && n_reps >= 100L && B_outer >= 99L && B >= 25L
if (full_run) {
  confirm <- summary[summary$estimator == "sudo", ]
  bias_ok <- abs(confirm$bias) <= pmax(2 * confirm$mc_se_bias, 0.03)
  coverage_ok <- abs(confirm$coverage - 0.95) <=
    pmax(2 * confirm$mc_se_coverage, 0.04)
  ratio <- confirm$sd / confirm$mean_se
  ratio_mc_se <- ratio * sqrt(1 / (2 * (n_reps - 1)))
  stopifnot(all(bias_ok), all(coverage_ok),
            all(abs(ratio - 1) <= 2 * ratio_mc_se),
            all(confirm$all_thresholds_ordered))
  cat("PASS: flexible ordinal SUDO satisfies every confirmatory gate\n")
} else {
  cat("SMOKE PASS: full acceptance requires the documented defaults\n")
}
dir.create("R/results", showWarnings = FALSE)
summary_file <- if (full_run) "R/results/stage14_ordinal_flexible.csv" else
  "R/results/stage14_ordinal_flexible_smoke.csv"
write.csv(summary, summary_file, row.names = FALSE)
if (Sys.getenv("SUDO_STAGE14_SKIP_ORACLE_WRITE", unset = "") != "1") {
  write.csv(oracle, oracle_file, row.names = FALSE)
}
