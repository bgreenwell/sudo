# Stage 5r: corrected ordinal SUDO with per-draw outcome-nuisance refitting.
#
# R/theory_ordinal_nuisance_ladder.R shows that reusing one plug-in estimate
# of E[S | X] across ordinal completions inflates the within-imputation
# sandwich variance. This stage validates the correction at the paper's full
# setting: B = 25 proper draws and 200 Monte Carlo replications.
#
# The full cumulative-link model is fit once and its thresholds and
# coefficients are drawn jointly. The treatment nuisance is cross-fitted
# once. The outcome nuisance is cross-fitted separately on every completed
# surrogate outcome.
#
# Run from the repository root:
#   Rscript R/stage5r_ordinal_refit.R
#
# Optional smoke-test overrides:
#   SUDO_STAGE5R_N=500 SUDO_STAGE5R_REPS=10 SUDO_STAGE5R_B=3 Rscript ...

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages({
  library(ordinal)
  library(splines)
})

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

dgp5r <- function(n, theta0, cuts = c(0, 2)) {
  X <- matrix(rnorm(n * 5), n, 5,
              dimnames = list(NULL, paste0("X", seq_len(5))))
  g0 <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  U <- theta0 * D + g0 + rlogis(n)
  list(X = X, D = D, Y = 1L + findInterval(U, cuts))
}

sudo_ordinal_refit <- function(d, B = 25L, n_folds = 5L, df_ns = 4L) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  n_cat <- max(d$Y)
  folds <- make_folds(n, n_folds)

  basis <- do.call(cbind, lapply(X, function(x) ns(x, df = df_ns)))
  colnames(basis) <- paste0("B", seq_len(ncol(basis)))
  model_matrix <- cbind(D = d$D, basis)
  dat <- data.frame(Y = factor(d$Y), model_matrix)

  full_fit <- clm(Y ~ ., data = dat)
  par_hat <- c(full_fit$alpha, full_fit$beta)
  covariance <- vcov(full_fit)[names(par_hat), names(par_hat), drop = FALSE]
  draw_surrogate <- function() {
    par <- MASS::mvrnorm(1, par_hat, covariance)
    index <- as.numeric(model_matrix %*% par[n_cat:length(par)])
    complete_surrogate(
      d$Y, index, c(-Inf, par[seq_len(n_cat - 1L)], Inf), "logit"
    )
  }

  D_residual <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)
  draws <- sapply(seq_len(B), function(draw_id) {
    surrogate <- draw_surrogate()
    ell_draw <- crossfit(X, surrogate, fit_gam, folds)
    fit <- fwl_theta(surrogate - ell_draw, D_residual)
    c(theta = fit$theta, var = fit$var)
  })
  pooled <- pool_rubin(
    draws["theta", ], draws["var", ], n_obs = n
  )
  list(
    theta = pooled$theta,
    se = pooled$se,
    ci_lo = pooled$ci_lo,
    ci_hi = pooled$ci_hi,
    W = pooled$W,
    B_between = pooled$B_between,
    df = pooled$df
  )
}

n <- env_int("SUDO_STAGE5R_N", 2000L)
n_reps <- env_int("SUDO_STAGE5R_REPS", 200L)
B <- env_int("SUDO_STAGE5R_B", 25L)
n_cores <- env_int("SUDO_STAGE5R_CORES", 6L)
theta0 <- 1

cluster <- mc_cluster(
  c("dgp5r", "sudo_ordinal_refit"),
  n_cores = n_cores
)
invisible(parallel::clusterEvalQ(
  cluster,
  suppressPackageStartupMessages({
    library(ordinal)
    library(splines)
  })
))
make_dgp <- function(n, theta0) {
  force(n)
  force(theta0)
  function() dgp5r(n, theta0)
}
make_estimator <- function(B) {
  force(B)
  function(d) sudo_ordinal_refit(d, B = B)
}

cat("Stage 5r: ordinal per-draw outcome-nuisance refit\n")
cat("n =", n, "reps =", n_reps, "B =", B, "cores =", n_cores, "\n\n")
result <- run_mc_par(
  cluster, n_reps, make_dgp(n, theta0), make_estimator(B),
  theta_true = theta0, seed = 15000
)
parallel::stopCluster(cluster)

summary <- summarize_mc(result)
summary$mean_T <- mean(result$se^2)
summary$V_emp <- var(result$est)
summary$T_over_V <- summary$mean_T / summary$V_emp
summary$mc_se_T_over_V <- summary$T_over_V *
  sqrt(2 / (n_reps - 1))
summary$sd_over_mean_se <- summary$sd / summary$mean_se
summary$mean_W <- mean(result$W)
summary$mean_B_between <- mean(result$B_between)
summary$mean_df <- mean(result$df)
print(format(summary, digits = 4), row.names = FALSE)

if (n_reps >= 200L && B >= 25L && n >= 2000L) {
  # Bias must be statistically indistinguishable from zero or below 2.5% of
  # theta, matching the original stage-5 approximation-bias allowance.
  stopifnot(
    abs(summary$bias) <
      max(2 * summary$mc_se_bias, 0.025 * abs(theta0)) + 1e-9
  )
  # At 200 replications, the Monte Carlo SE of 0.95 coverage is 0.015. The
  # two-SE band is therefore approximately [0.92, 0.98].
  stopifnot(summary$coverage >= 0.92, summary$coverage <= 0.98)
  # Direct variance calibration check against the Monte Carlo uncertainty of
  # an empirical variance. A fixed 20% cutoff is inappropriate here because
  # the ratio's MC SE is about 10% at 200 replications.
  stopifnot(
    abs(summary$T_over_V - 1) <= 2 * summary$mc_se_T_over_V
  )
}

output <- cbind(
  stage = "5r",
  estimator = sprintf("ordinal_refit_n%d_B%d", n, B),
  summary
)
dir.create("R/results", showWarnings = FALSE)
write.csv(output, "R/results/stage5r_summary.csv", row.names = FALSE)
cat("\nwrote R/results/stage5r_summary.csv\n")
if (n_reps >= 200L && B >= 25L && n >= 2000L) {
  cat("PASS: bias, coverage, and variance calibration satisfy the",
      "full stage-5r acceptance criteria\n")
} else {
  cat("SMOKE PASS: use n >= 2000, B >= 25, and at least 200",
      "replications for acceptance checks\n")
}
