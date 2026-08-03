# Stage 13: flexible non-logit binary validation.
#
# Correct-link SUDO uses a partially-linear cloglog GAM and a normal
# full-pipeline bootstrap interval. The direct GAM coefficient and a stable
# cloglog orthogonal score are reported on the same samples. A logit SUDO arm
# is a link-sensitivity analysis. It is not presented as a weak comparator.
#
# Full confirmation: n=2000, 100 replications, 99 outer resamples, B=25.
# Smoke example:
#   SUDO_STAGE13_N=500 SUDO_STAGE13_REPS=2 SUDO_STAGE13_OUTER=3 \
#   SUDO_STAGE13_B=3 Rscript R/stage13_binary_cloglog.R

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
n <- env_int("SUDO_STAGE13_N", 2000L)
n_reps <- env_int("SUDO_STAGE13_REPS", 100L)
B_outer <- env_int("SUDO_STAGE13_OUTER", 99L)
B <- env_int("SUDO_STAGE13_B", 25L)
n_folds <- env_int("SUDO_STAGE13_FOLDS", 5L)
n_cores <- env_int("SUDO_STAGE13_CORES", 6L)
theta_value <- Sys.getenv("SUDO_STAGE13_THETA", unset = "")
theta_grid <- if (nzchar(theta_value)) as.numeric(theta_value) else c(1, 2.5)
oracle_n <- env_int("SUDO_STAGE13_ORACLE_N", 2000L)

dgp13 <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5,
              dimnames = list(NULL, paste0("X", seq_len(5))))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  m <- plogis(X[, 4] + cos(X[, 5]))
  D <- rbinom(n, 1, m)
  eta <- theta * D + g
  Y <- rbinom(n, 1, -expm1(-exp(pmin(eta, 30))))
  list(X = X, D = D, Y = Y, eta = eta, m = m)
}

cloglog_score <- function(g, D, Y, m) {
  score <- function(theta) {
    eta1 <- g + theta
    eta0 <- g
    variance_weight <- function(eta) {
      u <- exp(pmin(eta, 30))
      u^2 * exp(-u) / pmax(-expm1(-u), 1e-300)
    }
    w1 <- variance_weight(eta1)
    w0 <- variance_weight(eta0)
    a <- m * w1 / pmax(m * w1 + (1 - m) * w0, 1e-300)
    eta_d <- ifelse(D == 1, eta1, eta0)
    u <- exp(pmin(eta_d, 30))
    p <- -expm1(-u)
    score_weight <- u / pmax(p, 1e-300)
    sum((D - a) * score_weight * (Y - p))
  }
  tryCatch(uniroot(score, c(-4, 8))$root, error = function(e) NA_real_)
}

one_rep <- function(seed, n, theta, B_outer, B, n_folds) {
  set.seed(seed)
  d <- dgp13(n, theta)
  folds <- make_folds(n, n_folds)
  correct_adapter <- make_binary_pl_gam_adapter("cloglog")
  fitted <- correct_adapter(d, folds)
  m_hat <- crossfit(as.data.frame(d$X), d$D, fit_gam_binomial, folds)
  score <- cloglog_score(fitted$index_control, d$D, d$Y, m_hat)
  sudo <- sudo_pipeline_boot(
    d, B_outer = B_outer, inner_B = B, n_folds = n_folds,
    estimator = sudo_discrete, adapter = correct_adapter,
    refit_S_nuisance = FALSE
  )
  wrong <- sudo_discrete(
    d, make_binary_pl_gam_adapter("logit"), B = B, n_folds = n_folds,
    folds = folds, refit_S_nuisance = FALSE
  )
  data.frame(
    sudo = sudo$theta, sudo_se = sudo$se,
    direct = fitted$direct_theta, score = score,
    logit_sensitivity = wrong$theta
  )
}

# Oracle smoke check on an evaluation seed not used below.
set.seed(13001)
oracle <- dgp13(oracle_n, 1)
oracle_fit <- make_binary_pl_gam_adapter("cloglog")(
  oracle, make_folds(length(oracle$Y), min(5L, oracle_n))
)
oracle_rmse <- sqrt(mean((oracle_fit$index - oracle$eta)^2))
oracle_alpha_error <- oracle_fit$direct_theta - 1
cat(sprintf("oracle smoke: index RMSE %.3f, treatment error %+.3f\n",
            oracle_rmse, oracle_alpha_error))
stopifnot(is.finite(oracle_rmse), oracle_rmse < 1.5,
          abs(oracle_alpha_error) < if (oracle_n >= 2000L) 0.25 else 0.75)

rows <- list()
cluster <- if (n_cores > 1L) mc_cluster(
  c("dgp13", "cloglog_score", "one_rep"), n_cores = n_cores
) else NULL
for (theta in theta_grid) {
  worker <- function(r) {
    one_rep(130000L + as.integer(theta * 10000) + r, n, theta, B_outer, B,
            n_folds)
  }
  if (!is.null(cluster)) {
    parallel::clusterExport(
      cluster, c("theta", "n", "B_outer", "B", "n_folds", "worker"),
      envir = environment()
    )
  }
  replications <- do.call(rbind, if (is.null(cluster))
    lapply(seq_len(n_reps), worker) else
    parallel::parLapplyLB(cluster, seq_len(n_reps), worker))
  for (estimator in c("sudo", "direct", "score", "logit_sensitivity")) {
    values <- replications[[estimator]]
    sd_value <- sd(values)
    is_sudo <- estimator == "sudo"
    coverage <- if (is_sudo) mean(abs(values - theta) <= 1.96 * replications$sudo_se)
    else NA_real_
    rows[[length(rows) + 1L]] <- data.frame(
      theta = theta, estimator = estimator, n_reps = n_reps,
      bias = mean(values) - theta, mc_se_bias = sd_value / sqrt(n_reps),
      sd = sd_value,
      mean_se = if (is_sudo) mean(replications$sudo_se) else NA_real_,
      coverage = coverage,
      mc_se_coverage = if (is_sudo) sqrt(coverage * (1 - coverage) / n_reps)
      else NA_real_
    )
  }
}
if (!is.null(cluster)) parallel::stopCluster(cluster)
summary <- do.call(rbind, rows)
summary$oracle_index_rmse <- oracle_rmse
summary$oracle_alpha_error <- oracle_alpha_error
print(format(summary, digits = 4), row.names = FALSE)

full_run <- n >= 2000L && n_reps >= 100L && B_outer >= 99L && B >= 25L
if (full_run) {
  confirm <- summary[summary$estimator == "sudo", ]
  bias_ok <- abs(confirm$bias) <=
    pmax(2 * confirm$mc_se_bias, 0.03 * abs(confirm$theta))
  coverage_ok <- abs(confirm$coverage - 0.95) <=
    pmax(2 * confirm$mc_se_coverage, 0.04)
  ratio <- confirm$sd / confirm$mean_se
  ratio_mc_se <- ratio * sqrt(1 / (2 * (n_reps - 1)))
  variance_ok <- abs(ratio - 1) <= 2 * ratio_mc_se
  stopifnot(all(bias_ok), all(coverage_ok), all(variance_ok))
  cat("PASS: correct-link SUDO satisfies bias, coverage, and SE gates\n")
} else {
  cat("SMOKE PASS: full acceptance requires the documented defaults\n")
}
dir.create("R/results", showWarnings = FALSE)
summary_file <- if (full_run) "R/results/stage13_binary_cloglog.csv" else
  "R/results/stage13_binary_cloglog_smoke.csv"
write.csv(summary, summary_file, row.names = FALSE)
write.csv(data.frame(n = oracle_n, theta = 1,
                     index_rmse = oracle_rmse,
                     treatment_error = oracle_alpha_error),
          "R/results/stage13_binary_oracle_smoke.csv", row.names = FALSE)
