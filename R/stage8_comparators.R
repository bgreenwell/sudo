# Stage 8: what does SUDO buy over reading theta off the cumulative-link fit?
#
# A referee will ask why one should complete the outcome and run FWL when
# `clm` already estimates the latent-scale coefficient, usually with a
# smaller standard error. This stage answers that empirically by putting four
# estimators on common data, folds, and seeds:
#
#   clm_direct  read theta straight off clm(Y ~ D + X)
#   clm_spline  read theta straight off clm(Y ~ D + ns(X_j, 3))
#   sudo        ordinal SUDO using the SAME clm as its imputation model,
#               proper parameter draws, outcome nuisance refit per draw
#   sudo_rb     Rao-Blackwellised SUDO: the completion is the conditional
#               mean E[S | Y, D, X] rather than a draw from it
#
# Two questions are under test.
#
# 1. Does truncation self-correction help? Proposition A3 says a treatment-
#    direction imputation error delta_D reaches theta damped by cbar_1 < 1.
#    So when the cumulative-link model is misspecified, direct read-off
#    should carry bias delta_D while SUDO carries roughly cbar_1 * delta_D.
#    If that shows up, it is the concrete answer to "why not just use clm".
#
# 2. Why randomise the completion at all? S_RB = E[S | Y, D, X] has the same
#    conditional mean as a draw by the tower property, so FWL on it still
#    targets theta_0, but with no surrogate Monte Carlo noise. If sudo_rb
#    matches sudo on bias with lower variance, the u/B term and the fixed-B
#    centring problem are avoidable and the method simplifies.
#
# Design D1 has every model correctly specified, so it measures the
# efficiency price of completion. Design D2 gives g_0 nonlinear structure the
# linear clm cannot represent, so it measures whether damping and flexible
# partialling buy back the bias that direct read-off suffers.
#
# This is a comparison, not a validation: the script asserts computational
# invariants and a correctly-specified sanity check, and does NOT assert that
# SUDO wins.
#
# Run from the repository root:
#   Rscript R/stage8_comparators.R
#
# Optional smoke-test overrides:
#   SUDO_STAGE8_N=400 SUDO_STAGE8_REPS=4 SUDO_STAGE8_B=3 Rscript R/stage8_comparators.R

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

N_OBS <- env_int("SUDO_STAGE8_N", 2000)
N_REPS <- env_int("SUDO_STAGE8_REPS", 200)
B_DRAWS <- env_int("SUDO_STAGE8_B", 25)
THETA0 <- 1.0
CUTS <- c(0, 2)

# ---- data-generating processes ---------------------------------------------
# D1: linear g_0, so clm(Y ~ D + X) is correctly specified.
# D2: nonlinear g_0 the linear clm cannot represent, which shifts its
#     pseudo-true D coefficient and creates the delta_D the theory predicts.

dgp_d1 <- function(n, theta0) {
  X <- matrix(rnorm(n * 3), n, 3, dimnames = list(NULL, paste0("X", 1:3)))
  g0 <- X[, 1] + 0.5 * X[, 2] - 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(0.8 * X[, 1]))
  U <- theta0 * D + g0 + rlogis(n)
  list(X = X, D = D, Y = 1L + findInterval(U, CUTS))
}

dgp_d2 <- function(n, theta0) {
  X <- matrix(rnorm(n * 3), n, 3, dimnames = list(NULL, paste0("X", 1:3)))
  g0 <- X[, 1]^2 + sin(2 * X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(0.8 * X[, 1]))
  U <- theta0 * D + g0 + rlogis(n)
  list(X = X, D = D, Y = 1L + findInterval(U, CUTS))
}

# ---- Rao-Blackwellised completion ------------------------------------------
# For a logistic latent error truncated to (a, b], the partial mean is
# minus the binary entropy at the corresponding probability (paper appendix),
# so E[xi | a < xi <= b] = {H(F(a)) - H(F(b))} / {F(b) - F(a)}.
# This is the exact conditional mean of the surrogate given the observed
# category, i.e. the B -> infinity limit of averaging draws.

binary_entropy <- function(p) {
  out <- numeric(length(p))
  ok <- p > 0 & p < 1
  out[ok] <- -p[ok] * log(p[ok]) - (1 - p[ok]) * log1p(-p[ok])
  out
}

rb_completion <- function(y, index, cutpoints) {
  lo <- cutpoints[y] - index
  hi <- cutpoints[y + 1L] - index
  f_lo <- plogis(lo)
  f_hi <- plogis(hi)
  denom <- f_hi - f_lo
  mu <- numeric(length(y))
  ok <- denom > 1e-10
  mu[ok] <- (binary_entropy(f_lo[ok]) - binary_entropy(f_hi[ok])) / denom[ok]
  # Degenerate cell: fall back to the interval midpoint, clipped to a finite
  # range so a single observation cannot dominate the regression.
  if (any(!ok)) {
    mid <- pmin(pmax((pmax(lo[!ok], -30) + pmin(hi[!ok], 30)) / 2, -30), 30)
    mu[!ok] <- mid
  }
  index + mu
}

# ---- estimators -------------------------------------------------------------
# All four share the same fitted cumulative-link model for a given rhs, so
# differences are attributable to the completion and partialling steps rather
# than to the outcome model.

fit_full_model <- function(d, rhs) {
  dat <- data.frame(Y = factor(d$Y), D = d$D, d$X)
  fit <- clm(as.formula(paste("Y ~ D +", rhs)), data = dat)
  alpha <- fit$alpha
  beta <- fit$beta
  par_hat <- c(alpha, beta)
  n_cat <- length(alpha) + 1L
  list(
    fit = fit,
    par_hat = par_hat,
    covariance = vcov(fit)[names(par_hat), names(par_hat), drop = FALSE],
    design = model.matrix(fit)$X[, names(beta), drop = FALSE],
    beta_idx = seq(n_cat, length(par_hat)),
    n_cat = n_cat
  )
}

direct_estimator <- function(rhs) {
  force(rhs)
  function(d) {
    fm <- fit_full_model(d, rhs)
    co <- summary(fm$fit)$coefficients["D", ]
    list(theta = unname(co["Estimate"]), se = unname(co["Std. Error"]))
  }
}

# completion_mode: "draw" for ordinary SUDO, "rb" for the conditional mean.
sudo_estimator <- function(rhs, completion_mode, B = B_DRAWS, n_folds = 5L) {
  force(rhs); force(completion_mode); force(B); force(n_folds)
  function(d) {
    X <- as.data.frame(d$X)
    n <- nrow(X)
    folds <- make_folds(n, n_folds)
    fm <- fit_full_model(d, rhs)
    d_residual <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)

    draws <- sapply(seq_len(B), function(b) {
      par <- MASS::mvrnorm(1, fm$par_hat, fm$covariance)
      index <- as.numeric(fm$design %*% par[fm$beta_idx])
      cutpoints <- c(-Inf, par[seq_len(fm$n_cat - 1L)], Inf)
      completion <- if (completion_mode == "rb") {
        rb_completion(d$Y, index, cutpoints)
      } else {
        complete_surrogate(d$Y, index, cutpoints, "logit")
      }
      ell <- crossfit(X, completion, fit_gam, folds)
      fit <- fwl_theta(completion - ell, d_residual)
      c(theta = fit$theta, var = fit$var)
    })
    pooled <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
    list(theta = pooled$theta, se = pooled$se,
         ci_lo = pooled$ci_lo, ci_hi = pooled$ci_hi,
         W = pooled$W, B_between = pooled$B_between)
  }
}

RHS_LINEAR <- "X1 + X2 + X3"
RHS_SPLINE <- "ns(X1, 3) + ns(X2, 3) + ns(X3, 3)"

# Arms are paired so each comparison isolates one thing. clm_direct vs sudo
# share the linear imputation model, so their gap is what completion plus
# flexible partialling does to a misspecified index. clm_spline vs
# sudo_spline share the spline one, so their gap is the efficiency price of
# completion when both are adequately specified. sudo vs sudo_rb differ only
# in whether the completion is drawn or conditioned.
ARMS <- list(
  list(name = "clm_direct",  est = direct_estimator(RHS_LINEAR)),
  list(name = "clm_spline",  est = direct_estimator(RHS_SPLINE)),
  list(name = "sudo",        est = sudo_estimator(RHS_LINEAR, "draw")),
  list(name = "sudo_spline", est = sudo_estimator(RHS_SPLINE, "draw")),
  list(name = "sudo_rb",     est = sudo_estimator(RHS_LINEAR, "rb"))
)

DESIGNS <- list(
  list(name = "D1_correct",   dgp = dgp_d1),
  list(name = "D2_nonlinear", dgp = dgp_d2)
)

make_dgp <- function(dgp_fn, n, theta0) {
  force(dgp_fn); force(n); force(theta0)
  function() dgp_fn(n, theta0)
}

cluster <- mc_cluster(c(
  "dgp_d1", "dgp_d2", "fit_full_model", "direct_estimator",
  "sudo_estimator", "rb_completion", "binary_entropy",
  "RHS_LINEAR", "RHS_SPLINE", "B_DRAWS", "CUTS"
))
invisible(parallel::clusterEvalQ(cluster, suppressPackageStartupMessages({
  library(ordinal)
  library(splines)
})))

cat(sprintf(
  "Stage 8: comparators, n=%d, B=%d, %d replications, theta0=%.1f\n\n",
  N_OBS, B_DRAWS, N_REPS, THETA0
))

rows <- list()
for (design in DESIGNS) {
  cat(sprintf("== %s ==\n", design$name))
  for (arm in ARMS) {
    df <- run_mc_par(cluster, N_REPS, make_dgp(design$dgp, N_OBS, THETA0),
                     arm$est, THETA0, seed = 8000)
    s <- summarize_mc(df)
    print_mc(sprintf("%-11s", arm$name), s)
    rows[[length(rows) + 1L]] <- cbind(
      stage = 8, design = design$name, estimator = arm$name, s
    )
  }
  cat("\n")
}
parallel::stopCluster(cluster)

summary_table <- do.call(rbind, rows)

# Relative efficiency against the best arm within each design, so the
# efficiency price of completion is readable directly off the table.
summary_table$rel_var <- unlist(lapply(split(summary_table, summary_table$design),
  function(block) block$sd^2 / min(block$sd^2)))[rownames(summary_table)]

# ---- acceptance -------------------------------------------------------------
# Computational invariants, plus one sanity check that the harness works: with
# every model correctly specified in D1, direct read-off must be unbiased.
# Nothing here presupposes that SUDO compares favourably.
stopifnot(
  nrow(summary_table) == length(DESIGNS) * length(ARMS),
  all(summary_table$n_reps == N_REPS),
  all(is.finite(summary_table$bias)),
  all(is.finite(summary_table$sd)),
  all(is.finite(summary_table$mean_se)),
  all(summary_table$sd > 0)
)
d1_direct <- summary_table[summary_table$design == "D1_correct" &
                             summary_table$estimator == "clm_direct", ]
stopifnot(abs(d1_direct$bias) < 3 * d1_direct$mc_se_bias)

cat("PASS: all cells finite, correctly-specified direct read-off unbiased\n")
cat("      (comparison stage: no arm is asserted to win)\n")

dir.create("R/results", showWarnings = FALSE)
write.csv(summary_table, "R/results/stage8_summary.csv", row.names = FALSE)
cat("wrote R/results/stage8_summary.csv\n")
