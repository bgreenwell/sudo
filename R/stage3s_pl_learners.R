# Stage 3s: learner-agnostic partially linear imputation comparison.
#
# The required structure is an imputation index
#
#     v(D, X) = alpha * D + f(X),
#
# not a particular training algorithm. This stage compares three independent
# implementations on common data-generating seeds:
#   1. mgcv GAM with an unpenalized linear D term;
#   2. MARS bases built from X only, followed by a binomial GLM with mandatory D;
#   3. the tuned neural-network backfit selected in stage 3m.
#
# MARS is screened on the target-level Monte Carlo bias at theta in {1.5, 3}.
# Predictive GCV is not used to choose the final configuration because stage 3l
# showed that optimizing predictive risk can worsen the target coefficient.
#
# Full-fidelity defaults:
#   screen: n=1000, 20 replications, B=5
#   validation: n=2000, 50 replications, 50 outer bootstrap samples,
#               15 surrogate completions per bootstrap sample
#
# Environment overrides support a fast smoke check:
#   SUDO_PL_N=500 SUDO_PL_SCREEN_REPS=2 SUDO_PL_REPS=2 \
#   SUDO_PL_OUTER=2 SUDO_PL_INNER=2 Rscript R/stage3s_pl_learners.R
#
# Acceptance:
#   - every fitter returns finite indices;
#   - MARS reports treatment basis degree one by construction;
#   - GAM and tuned PL-backfit bias is within
#       max(2 Monte Carlo SE, 3% of theta)
#     at both signals;
#   - an additional learner is promoted only if it passes the same bias check
#     and its percentile coverage is within a Monte Carlo-aware 0.95 band.

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

dgp3s <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  U <- theta * D + g + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0))
}

make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp3s(n, theta)
}

make_point_est <- function(learner, B, mars_degree = 2L, mars_nk = 41L,
                           mars_penalty = 3) {
  force(learner); force(B); force(mars_degree); force(mars_nk)
  force(mars_penalty)
  function(d) {
    args <- list(d = d, B = B, proper = FALSE, full_model = learner)
    if (learner == "pl_backfit") {
      args$nn_size <- 4
      args$nn_decay <- 0.3
      args$pl_n_iter <- 5
    }
    if (learner == "pl_mars") {
      args$mars_degree <- mars_degree
      args$mars_nk <- mars_nk
      args$mars_penalty <- mars_penalty
    }
    do.call(sudo_binary, args)
  }
}

make_boot_est <- function(learner, B_outer, inner_B, mars_degree = 2L,
                          mars_nk = 41L, mars_penalty = 3) {
  force(learner); force(B_outer); force(inner_B); force(mars_degree)
  force(mars_nk); force(mars_penalty)
  function(d) {
    args <- list(d = d, B_outer = B_outer, inner_B = inner_B,
                 full_model = learner)
    if (learner == "pl_backfit") {
      args$nn_size <- 4
      args$nn_decay <- 0.3
      args$pl_n_iter <- 5
    }
    if (learner == "pl_mars") {
      args$mars_degree <- mars_degree
      args$mars_nk <- mars_nk
      args$mars_penalty <- mars_penalty
    }
    do.call(sudo_pipeline_boot, args)
  }
}

n <- env_int("SUDO_PL_N", 2000L)
screen_n <- min(n, env_int("SUDO_PL_SCREEN_N", 1000L))
screen_reps <- env_int("SUDO_PL_SCREEN_REPS", 20L)
n_reps <- env_int("SUDO_PL_REPS", 50L)
B_outer <- env_int("SUDO_PL_OUTER", 50L)
inner_B <- env_int("SUDO_PL_INNER", 15L)
screen_B <- env_int("SUDO_PL_SCREEN_B", 5L)
mc_cores <- env_int("SUDO_PL_CORES", 2L)

cl <- mc_cluster(c(
  "dgp3s", "sudo_binary", "sudo_pipeline_boot", "crossfit_fullmodel_gam",
  "fit_pl_gam", "fit_pl_mars", "fit_pl_backfit"
), n_cores = mc_cores)

cat("Stage 3s: partially linear learner comparison\n")
cat(sprintf(
  "screen n=%d reps=%d B=%d; validation n=%d reps=%d outer=%d inner=%d\n\n",
  screen_n, screen_reps, screen_B, n, n_reps, B_outer, inner_B))

# Structural smoke check on data independent of the Monte Carlo cells.
set.seed(31001)
d_check <- dgp3s(min(n, 500L), 1.5)
folds_check <- make_folds(length(d_check$Y), 5)
fits_check <- list(
  gam = fit_pl_gam(d_check, folds_check),
  mars = fit_pl_mars(d_check, folds_check, degree = 2, nk = 21, penalty = 3),
  backfit = fit_pl_backfit(d_check, folds_check, nn_size = 4,
                           nn_decay = 0.3, n_iter = 5)
)
stopifnot(
  all(vapply(fits_check, function(x) all(is.finite(x$lp_hat)), logical(1))),
  identical(fits_check$mars$treatment_basis_degree, 1L)
)
cat("PASS: all learner indices are finite and MARS keeps D outside its X basis\n")

# Target-level MARS screen.
grid <- expand.grid(
  degree = c(1L, 2L),
  nk = c(21L, 41L, 81L),
  penalty = c(2, 3),
  KEEP.OUT.ATTRS = FALSE
)
screen_rows <- list()
for (j in seq_len(nrow(grid))) {
  cfg <- grid[j, ]
  bias <- mcse <- numeric(2)
  for (h in seq_along(c(1.5, 3))) {
    theta <- c(1.5, 3)[h]
    df <- run_mc_par(
      cl, screen_reps, make_dgp(screen_n, theta),
      make_point_est("pl_mars", screen_B, cfg$degree, cfg$nk, cfg$penalty),
      theta, seed = 61000 + round(theta * 1000)
    )
    sm <- summarize_mc(df)
    bias[h] <- sm$bias
    mcse[h] <- sm$mc_se_bias
  }
  screen_rows[[j]] <- data.frame(
    degree = cfg$degree, nk = cfg$nk, penalty = cfg$penalty,
    bias_t1.5 = bias[1], mc_se_t1.5 = mcse[1],
    bias_t3 = bias[2], mc_se_t3 = mcse[2],
    worst_abs_bias = max(abs(bias))
  )
}
screen <- do.call(rbind, screen_rows)
screen <- screen[order(screen$worst_abs_bias), ]
best <- screen[1, ]
cat("\nMARS target-level screen:\n")
print(screen, row.names = FALSE)
cat(sprintf(
  "winner: degree=%d nk=%d penalty=%.1f, worst |bias|=%.4f\n\n",
  best$degree, best$nk, best$penalty, best$worst_abs_bias))

# Common full-pipeline validation.
learners <- c("pl_gam", "pl_mars", "pl_backfit")
validation_rows <- list()
for (learner in learners) {
  for (theta in c(1.5, 3)) {
    est <- make_boot_est(
      learner, B_outer, inner_B, best$degree, best$nk, best$penalty)
    df <- run_mc_par(
      cl, n_reps, make_dgp(n, theta), est, theta,
      seed = 81000 + round(theta * 1000)
    )
    sm <- summarize_mc(df)
    coverage_pct <- mean(df$ci_lo_pct <= theta & theta <= df$ci_hi_pct)
    mc_se_pct <- sqrt(coverage_pct * (1 - coverage_pct) / n_reps)
    row <- data.frame(
      stage = "3s", learner = learner, theta = theta, n = n,
      n_reps = n_reps, B_outer = B_outer, inner_B = inner_B,
      mean_est = sm$mean_est, bias = sm$bias, mc_se_bias = sm$mc_se_bias,
      sd = sm$sd, mean_se = sm$mean_se,
      sd_over_mean_se = sm$sd / sm$mean_se,
      coverage_normal = sm$coverage, mc_se_normal = sm$mc_se_cov,
      coverage_percentile = coverage_pct, mc_se_percentile = mc_se_pct
    )
    validation_rows[[length(validation_rows) + 1L]] <- row
    cat(sprintf(
      "%-12s theta=%.1f bias=%+.4f sd/se=%.3f cover normal=%.3f percentile=%.3f\n",
      learner, theta, row$bias, row$sd_over_mean_se, row$coverage_normal,
      row$coverage_percentile))
  }
}
parallel::stopCluster(cl)
validation <- do.call(rbind, validation_rows)

bias_pass <- with(
  validation,
  abs(bias) <= pmax(2 * mc_se_bias, 0.03 * theta) + 1e-9
)
coverage_pass <- with(
  validation,
  abs(coverage_percentile - 0.95) <=
    pmax(2 * mc_se_percentile, 0.04) + 1e-9
)
validation$bias_pass <- bias_pass
validation$coverage_pass <- coverage_pass
validation$promoted <- bias_pass & coverage_pass

# Existing validated controls must retain target-level calibration.
control <- validation$learner %in% c("pl_gam", "pl_backfit")
full_fidelity <- n >= 1000L && n_reps >= 50L && B_outer >= 50L
if (full_fidelity) stopifnot(all(validation$bias_pass[control]))

mars_pass <- all(validation$promoted[validation$learner == "pl_mars"])
cat("\n")
if (mars_pass) {
  cat("PASS: MARS meets the bias and percentile-coverage promotion criteria\n")
} else {
  cat("PASS: comparison completed; MARS remains experimental because it did",
      "not meet every promotion criterion\n")
}
if (!full_fidelity) {
  cat("SMOKE: statistical acceptance assertions require n >= 1000,",
      "50 replications, and 50 outer bootstrap samples\n")
}

dir.create("R/results", showWarnings = FALSE)
write.csv(screen, "R/results/stage3s_mars_screen.csv", row.names = FALSE)
write.csv(validation, "R/results/stage3s_summary.csv", row.names = FALSE)
write.csv(data.frame(
  learner = "pl_mars", degree = best$degree, nk = best$nk,
  penalty = best$penalty, promoted = mars_pass
), "R/results/stage3s_mars_config.csv", row.names = FALSE)
cat("wrote R/results/stage3s_{mars_screen,summary,mars_config}.csv\n")
