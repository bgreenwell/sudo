# Sensitivity analysis for the wine application.
#
# Two robustness checks the paper promises but did not yet deliver:
#   1. Adjustment-set robustness. Rerun ordinal SUDO with the full covariate
#      set and with every leave-one-covariate-out set. If the effect stays
#      clearly negative across all of them, it does not hinge on any single
#      adjustment choice (a partial guard against a mis-specified set, e.g. a
#      covariate that is really a mediator or proxy).
#   2. Unmeasured-confounding robustness value (Cinelli and Hazlett 2020,
#      omitted-variable-bias framework applied to the latent-scale FWL
#      coefficient). RV is the fraction of residual variation in BOTH the
#      treatment and the latent outcome that an unobserved confounder would
#      have to explain to drive the estimate to zero. Larger RV = more robust.
#
# Linear cumulative-link full model throughout (the headline spec). Run from
# repo root: Rscript R/wine_sensitivity.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
suppressPackageStartupMessages({ library(ordinal); library(mgcv) })

DATA <- "manuscript/data/wine"

# ordinal SUDO, linear clm full model, for a given adjustment set X
sudo_wine <- function(y, D, X, B = 50, n_folds = 5, seed = 1) {
  set.seed(seed)
  n <- length(y)
  folds <- make_folds(n, n_folds)
  D_res <- D - crossfit(X, D, fit_gam, folds)          # continuous D
  dat <- data.frame(Y = factor(y), D = D, X)
  covs <- names(X)
  fit <- clm(as.formula(paste("Y ~ D +", paste(covs, collapse = " + "))),
             data = dat)
  alpha <- fit$alpha; beta <- fit$beta
  par_hat <- c(alpha, beta)
  V <- vcov(fit)[names(par_hat), names(par_hat)]
  mm <- model.matrix(fit)$X[, names(beta), drop = FALSE]
  J <- length(alpha) + 1
  b_idx <- seq(J, length(par_hat))
  cuts <- function(par) c(-Inf, par[1:(J - 1)], Inf)

  S_hat <- crossfit(X, complete_surrogate(
    y, as.numeric(mm %*% par_hat[b_idx]), cuts(par_hat), "logit"),
    fit_gam, folds)
  draws <- sapply(seq_len(B), function(b) {
    par <- MASS::mvrnorm(1, par_hat, V)
    S <- complete_surrogate(y, as.numeric(mm %*% par[b_idx]), cuts(par), "logit")
    f <- fwl_theta(S - S_hat, D_res)
    c(theta = f$theta, var = f$var)
  })
  p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
  df_res <- n - length(par_hat) - 1                    # approx residual df
  list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi,
       df = df_res)
}

# Cinelli-Hazlett robustness value (q = 1: reduce point estimate to 0)
robustness_value <- function(theta, se, df, q = 1) {
  fq <- q * abs(theta / se) / sqrt(df)
  0.5 * (sqrt(fq^4 + 4 * fq^2) - fq^2)
}

analyse_sens <- function(which, B = 50) {
  d <- read.csv(file.path(DATA, sprintf("winequality-%s.csv", which)),
                sep = ";")
  y <- as.integer(factor(d$quality))
  D <- as.numeric(scale(d$volatile.acidity))
  covs <- setdiff(names(d), c("volatile.acidity", "quality"))
  Xfull <- as.data.frame(scale(d[covs]))

  full <- sudo_wine(y, D, Xfull, B = B)
  rv <- robustness_value(full$theta, full$se, full$df)

  # leave-one-covariate-out
  loo <- lapply(covs, function(cv) {
    r <- sudo_wine(y, D, Xfull[setdiff(covs, cv)], B = B)
    data.frame(wine = which, adjustment = paste0("drop_", cv),
               theta = r$theta, se = r$se, ci_lo = r$ci_lo, ci_hi = r$ci_hi)
  })
  loo <- do.call(rbind, loo)

  full_row <- data.frame(wine = which, adjustment = "full",
                         theta = full$theta, se = full$se,
                         ci_lo = full$ci_lo, ci_hi = full$ci_hi)
  list(rows = rbind(full_row, loo), rv = rv,
       full_theta = full$theta, full_ci = c(full$ci_lo, full$ci_hi),
       loo_range = range(loo$theta), loo_max_ci_hi = max(loo$ci_hi))
}

cat("Wine sensitivity: adjustment-set robustness and omitted-variable bound\n\n")
all_rows <- list()
for (w in c("red", "white")) {
  r <- analyse_sens(w)
  all_rows[[w]] <- r$rows
  cat(sprintf("== %s ==\n", w))
  cat(sprintf("  full set:        theta = %+.3f  95%% CI [%.3f, %.3f]\n",
              r$full_theta, r$full_ci[1], r$full_ci[2]))
  cat(sprintf("  leave-one-out:   theta in [%+.3f, %+.3f], worst CI upper = %.3f\n",
              r$loo_range[1], r$loo_range[2], r$loo_max_ci_hi))
  cat(sprintf("  robustness value: RV = %.3f (confounder must explain %.0f%% of\n",
              r$rv, 100 * r$rv))
  cat("                    both residual variances to nullify the effect)\n\n")
  # every adjustment set must keep the effect clearly negative
  stopifnot(all(r$rows$ci_hi < 0))
}
sm <- do.call(rbind, all_rows)
dir.create("R/results", showWarnings = FALSE)
write.csv(sm, "R/results/wine_sensitivity.csv", row.names = FALSE)
cat("PASS: effect stays clearly negative across all leave-one-out sets.\n")
cat("wrote R/results/wine_sensitivity.csv\n")
