# Theory check: first-order generated-surrogate expansion for cross-fitted
# partially linear imputation learners.
#
# For a fitted index v_hat and true index v0, define
#
#   Delta(v_hat, v0) = E[S(v_hat) | D, X] - v0
#                    = psi(v_hat; v0) - v0.
#
# The learner-class expansion uses
#
#   E[R Delta(v_hat, v0)]
#     = E[R c(v0) (v_hat - v0)] + second-order remainder.
#
# This script evaluates that identity for the supported GAM, MARS, and neural
# backfit learners on the stage-3 DGP. It does not assume a finite-dimensional
# parameter vector for any learner.
#
# Full defaults: n in {1000, 2000}, 50 replications, two signals.
# Smoke check:
#   SUDO_ML_EXP_NS=500 SUDO_ML_EXP_REPS=2 \
#   Rscript R/theory_ml_expansion.R
#
# Acceptance:
#   - the analytic pass-through factor matches numerical differentiation;
#   - the pointwise Taylor remainder is second order in index error;
#   - the residual-projected remainder is smaller than the first-order scale;
#   - every learner returns finite quantities in every cell.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")
source("R/sudo/pl.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

env_ints <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(as.integer(default))
  as.integer(strsplit(value, ",", fixed = TRUE)[[1]])
}

psi_logit <- function(v, e) {
  p_v <- pmin(pmax(plogis(v), 1e-10), 1 - 1e-10)
  q_v <- 1 - p_v
  entropy <- -(p_v * log(p_v) + q_v * log(q_v))
  mu_plus <- entropy / p_v
  mu_minus <- -entropy / q_v
  p_e <- plogis(e)
  v + p_e * mu_plus + (1 - p_e) * mu_minus
}

dgp_ml_expansion <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  propensity <- plogis(X[, 4] + cos(X[, 5]))
  D <- rbinom(n, 1, propensity)
  eta <- theta * D + X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  Y <- rbinom(n, 1, plogis(eta))
  list(X = X, D = D, Y = Y, eta = eta, propensity = propensity)
}

fit_index <- function(d, learner, folds) {
  switch(
    learner,
    pl_gam = fit_pl_gam(d, folds),
    pl_mars = fit_pl_mars(d, folds, degree = 2, nk = 21, penalty = 3),
    pl_backfit = fit_pl_backfit(
      d, folds, nn_size = 4, nn_decay = 0.3, n_iter = 5)
  )$lp_hat
}

one_expansion_rep <- function(seed, n, theta, learner) {
  set.seed(seed)
  d <- dgp_ml_expansion(n, theta)
  folds <- make_folds(n, 5)
  v_hat <- fit_index(d, learner, folds)
  R <- d$D - d$propensity
  delta <- v_hat - d$eta
  exact_point <- psi_logit(v_hat, d$eta) - d$eta
  linear_point <- pass_c(d$eta) * delta
  point_remainder <- exact_point - linear_point
  J <- mean(R^2)

  c(
    exact_shift = mean(R * exact_point) / J,
    linear_shift = mean(R * linear_point) / J,
    projected_remainder = mean(R * point_remainder) / J,
    mean_abs_point_remainder = mean(abs(point_remainder)),
    mean_delta_sq = mean(delta^2),
    index_rmse = sqrt(mean(delta^2)),
    max_abs_index = max(abs(v_hat)),
    all_finite = as.numeric(all(is.finite(c(
      v_hat, exact_point, linear_point, point_remainder))))
  )
}

# Independent pointwise derivative check.
e_grid <- seq(-6, 6, length.out = 49)
h <- 1e-5
numeric_derivative <- (
  psi_logit(e_grid + h, e_grid) - psi_logit(e_grid - h, e_grid)
) / (2 * h)
derivative_error <- max(abs(numeric_derivative - pass_c(e_grid)))
cat(sprintf("max derivative error: %.3e\n", derivative_error))
stopifnot(derivative_error < 2e-5)

ns <- env_ints("SUDO_ML_EXP_NS", c(1000L, 2000L))
n_reps <- env_int("SUDO_ML_EXP_REPS", 50L)
mc_cores <- env_int("SUDO_ML_EXP_CORES", 2L)
learners <- c("pl_gam", "pl_mars", "pl_backfit")

cl <- mc_cluster(c(
  "dgp_ml_expansion", "psi_logit", "fit_index", "one_expansion_rep",
  "fit_pl_gam", "fit_pl_mars", "fit_pl_backfit"
), n_cores = mc_cores)

rows <- list()
for (n in ns) {
  for (theta in c(1.5, 3)) {
    for (learner in learners) {
      parallel::clusterExport(
        cl, c("n", "theta", "learner"), envir = environment())
      values <- parallel::parSapply(cl, seq_len(n_reps), function(r) {
        one_expansion_rep(92000 + r, n, theta, learner)
      })
      mean_abs_projected <- mean(abs(values["projected_remainder", ]))
      mean_abs_linear <- mean(abs(values["linear_shift", ]))
      row <- data.frame(
        n = n, theta = theta, learner = learner, n_reps = n_reps,
        mean_exact_shift = mean(values["exact_shift", ]),
        mean_linear_shift = mean(values["linear_shift", ]),
        mean_abs_projected_remainder = mean_abs_projected,
        mean_abs_linear_shift = mean_abs_linear,
        mean_abs_point_remainder =
          mean(values["mean_abs_point_remainder", ]),
        mean_delta_sq = mean(values["mean_delta_sq", ]),
        taylor_ratio =
          mean(values["mean_abs_point_remainder", ]) /
          mean(values["mean_delta_sq", ]),
        mean_index_rmse = mean(values["index_rmse", ]),
        max_abs_index = max(values["max_abs_index", ]),
        all_finite = all(values["all_finite", ] == 1)
      )
      rows[[length(rows) + 1L]] <- row
      cat(sprintf(
        "n=%d theta=%.1f %-10s exact=%+.4f linear=%+.4f remainder=%.4f Taylor ratio=%.3f\n",
        n, theta, learner, row$mean_exact_shift, row$mean_linear_shift,
        row$mean_abs_projected_remainder, row$taylor_ratio))
    }
  }
}
parallel::stopCluster(cl)
out <- do.call(rbind, rows)

stopifnot(
  all(out$all_finite),
  all(out$taylor_ratio < 1),
  all(out$mean_abs_projected_remainder <
        pmax(out$mean_abs_linear_shift, 0.05))
)

cat("PASS: the learner-class generated-outcome expansion is first order",
    "with a second-order map remainder in every cell\n")
dir.create("R/results", showWarnings = FALSE)
write.csv(out, "R/results/theory_ml_expansion.csv", row.names = FALSE)
cat("wrote R/results/theory_ml_expansion.csv\n")
