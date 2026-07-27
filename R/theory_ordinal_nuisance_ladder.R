# Theory diagnostic: isolate the stage-5 ordinal variance inflation.
#
# All arms share the same data, cumulative-link fit, parameter draws,
# surrogate draws, folds, and plug-in nuisance fits. They differ only in
# which partialling nuisance is supplied to FWL:
#
#   oracle_both  true m0(X) and true ell0(X)
#   gam_m        cross-fitted m, true ell0
#   gam_l        true m0, cross-fitted plug-in ell
#   gam_both     cross-fitted m and plug-in ell, the stage-5 estimator
#   gam_per_draw cross-fitted m and ell refit on every surrogate draw
#
# The true outcome nuisance is ell0(X) = theta0 m0(X) + g0(X). A proper
# parameter draw moves that nuisance by O_p(n^-1/2), but exact residualization
# by R0 = D - m0(X) annihilates any X-only displacement in the target moment.
# The arm remains useful as the oracle benchmark for the within-draw sandwich.
#
# Each arm reports the empirical sampling variance, mean Rubin variance,
# within and between components, and coverage. The paired design identifies
# the first nuisance substitution that reproduces stage 5's large T/V ratio.
#
# Run from the repository root:
#   Rscript R/theory_ordinal_nuisance_ladder.R
#
# Optional smoke-test overrides:
#   SUDO_LADDER_N=1000 SUDO_LADDER_REPS=20 SUDO_LADDER_B=5 Rscript ...

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

dgp_ordinal_ladder <- function(n, theta0, cuts = c(0, 2)) {
  X <- matrix(rnorm(n * 5), n, 5,
              dimnames = list(NULL, paste0("X", seq_len(5))))
  g0 <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  m0 <- plogis(X[, 4] + cos(X[, 5]))
  D <- rbinom(n, 1, m0)
  U <- theta0 * D + g0 + rlogis(n)
  list(
    X = X, D = D, Y = 1L + findInterval(U, cuts),
    g0 = g0, m0 = m0, cuts = cuts, theta0 = theta0
  )
}

fit_ordinal_ladder <- function(d, B = 15L, n_folds = 5L, df_ns = 4L,
                               refit_per_draw = TRUE) {
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

  draw_surrogate <- function(perturb = TRUE) {
    par <- if (perturb) MASS::mvrnorm(1, par_hat, covariance) else par_hat
    index <- as.numeric(
      model_matrix %*% par[n_cat:length(par)]
    )
    complete_surrogate(
      d$Y, index, c(-Inf, par[seq_len(n_cat - 1L)], Inf), "logit"
    )
  }

  R_oracle <- d$D - d$m0
  ell_oracle <- d$theta0 * d$m0 + d$g0
  R_gam <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)
  surrogate_plugin <- draw_surrogate(FALSE)
  ell_gam <- crossfit(X, surrogate_plugin, fit_gam, folds)

  arm_names <- c("oracle_both", "gam_m", "gam_l", "gam_both")
  if (refit_per_draw) arm_names <- c(arm_names, "gam_per_draw")
  theta_draw <- matrix(NA_real_, nrow = length(arm_names), ncol = B,
                       dimnames = list(arm_names, NULL))
  variance_draw <- theta_draw

  store_fit <- function(arm, draw_id, outcome_residual, treatment_residual) {
    fit <- fwl_theta(outcome_residual, treatment_residual)
    theta_draw[arm, draw_id] <<- fit$theta
    variance_draw[arm, draw_id] <<- fit$var
  }

  for (b in seq_len(B)) {
    surrogate <- draw_surrogate(TRUE)
    store_fit("oracle_both", b, surrogate - ell_oracle, R_oracle)
    store_fit("gam_m", b, surrogate - ell_oracle, R_gam)
    store_fit("gam_l", b, surrogate - ell_gam, R_oracle)
    store_fit("gam_both", b, surrogate - ell_gam, R_gam)
    if (refit_per_draw) {
      ell_draw <- crossfit(X, surrogate, fit_gam, folds)
      store_fit("gam_per_draw", b, surrogate - ell_draw, R_gam)
    }
  }

  unlist(lapply(arm_names, function(arm) {
    pooled <- pool_rubin(
      theta_draw[arm, ], variance_draw[arm, ], n_obs = n
    )
    values <- c(
      theta = pooled$theta,
      T = pooled$T,
      W = pooled$W,
      B_between = pooled$B_between,
      ci_lo = pooled$ci_lo,
      ci_hi = pooled$ci_hi
    )
    names(values) <- paste(arm, names(values), sep = ".")
    values
  }), use.names = TRUE)
}

summarize_ladder <- function(result, theta0, n, B) {
  arm_names <- sub("\\..*$", "",
                   colnames(result)[seq(1, ncol(result), by = 6)])
  rows <- lapply(seq_along(arm_names), function(arm_id) {
    start <- (arm_id - 1L) * 6L
    theta <- result[, start + 1L]
    T_var <- result[, start + 2L]
    W <- result[, start + 3L]
    B_between <- result[, start + 4L]
    ci_lo <- result[, start + 5L]
    ci_hi <- result[, start + 6L]
    empirical_variance <- var(theta)
    mean_T <- mean(T_var)
    data.frame(
      arm = arm_names[arm_id],
      n = n,
      B = B,
      reps = nrow(result),
      bias = mean(theta) - theta0,
      mc_se_bias = sd(theta) / sqrt(nrow(result)),
      sd = sd(theta),
      mean_se = mean(sqrt(T_var)),
      mean_T = mean_T,
      V_emp = empirical_variance,
      T_over_V = mean_T / empirical_variance,
      ratio_se = mean_T / empirical_variance *
        sqrt(2 / (nrow(result) - 1)),
      mean_W = mean(W),
      mean_B_between = mean(B_between),
      coverage = mean(ci_lo <= theta0 & ci_hi >= theta0)
    )
  })
  do.call(rbind, rows)
}

n <- env_int("SUDO_LADDER_N", 2000L)
n_reps <- env_int("SUDO_LADDER_REPS", 200L)
B <- env_int("SUDO_LADDER_B", 15L)
theta0 <- 1

cluster <- mc_cluster(
  c("dgp_ordinal_ladder", "fit_ordinal_ladder"),
  n_cores = env_int("SUDO_LADDER_CORES", 2L)
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
  function() dgp_ordinal_ladder(n, theta0)
}
make_estimator <- function(B) {
  force(B)
  function(d) {
    values <- fit_ordinal_ladder(d, B = B)
    # Supply the standard MC contract through the oracle arm and carry every
    # paired arm as an extra scalar diagnostic.
    c(
      list(
        theta = unname(values["oracle_both.theta"]),
        se = sqrt(unname(values["oracle_both.T"])),
        ci_lo = unname(values["oracle_both.ci_lo"]),
        ci_hi = unname(values["oracle_both.ci_hi"])
      ),
      as.list(values)
    )
  }
}

cat("Ordinal nuisance-isolation ladder\n")
cat("n =", n, "reps =", n_reps, "B =", B, "\n\n")
raw <- run_mc_par(
  cluster, n_reps, make_dgp(n, theta0), make_estimator(B),
  theta_true = theta0, seed = 12000
)
parallel::stopCluster(cluster)

# run_mc_par reserves est/se/covered fields for its standard contract. This
# diagnostic returns a wider named vector, which is preserved after them.
value_columns <- setdiff(
  names(raw),
  c("rep", "est", "se", "ci_lo", "ci_hi", "covered", "error")
)
if (!length(value_columns)) {
  stop("paired arm diagnostics were not returned by run_mc_par()")
}
result <- as.matrix(raw[, value_columns, drop = FALSE])
storage.mode(result) <- "double"
summary <- summarize_ladder(result, theta0, n, B)
print(format(summary, digits = 4), row.names = FALSE)

if (n_reps >= 100L) {
  current_ratio <- summary$T_over_V[summary$arm == "gam_both"]
  refit_ratio <- summary$T_over_V[summary$arm == "gam_per_draw"]
  current_T <- summary$mean_T[summary$arm == "gam_both"]
  refit_T <- summary$mean_T[summary$arm == "gam_per_draw"]
  current_V <- summary$V_emp[summary$arm == "gam_both"]
  refit_V <- summary$V_emp[summary$arm == "gam_per_draw"]
  stopifnot(length(current_ratio) == 1L, is.finite(current_ratio),
            length(refit_ratio) == 1L, is.finite(refit_ratio))
  stopifnot(all(abs(summary$bias) <
                  pmax(3 * summary$mc_se_bias, 0.04 * abs(theta0))))
  # Reusing the plug-in outcome nuisance must reproduce the large stage-5
  # variance inflation. Refitting it per draw must remove at least 30% of T
  # without materially changing the estimator's empirical variance.
  stopifnot(current_ratio > 1.25)
  stopifnot(current_T / refit_T > 1.30)
  stopifnot(current_V / refit_V > 0.85, current_V / refit_V < 1.15)
  stopifnot(refit_ratio > 0.75, refit_ratio < 1.25)
}

dir.create("R/results", showWarnings = FALSE)
write.csv(summary, "R/results/theory_ordinal_nuisance_ladder.csv",
          row.names = FALSE)
cat("\nwrote R/results/theory_ordinal_nuisance_ladder.csv\n")
if (n_reps >= 100L) {
  cat("PASS: all arms satisfy the bias tolerance; the fixed outcome",
      "nuisance inflates T without changing empirical variance, and",
      "per-draw refitting restores T/V near one\n")
} else {
  cat("SMOKE PASS: rerun with SUDO_LADDER_REPS >= 100 for",
      "acceptance checks\n")
}
