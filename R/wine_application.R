# Application: the effect of volatile acidity on wine quality.
#
# Wine quality is an ORDINAL sensory score (the median of blind tasters,
# 3-8 for red, 3-9 for white) -- exactly the outcome type no existing DML
# estimator handles, which is why this showcases SUDO rather than competing
# with anything.
#
# Causal question, under an assumed DAG (stated as an assumption -- this is
# observational data, not an experiment): the effect of volatile acidity
# (VA, the acetic-acid "vinegar/nail-polish" fault) on latent quality,
# holding the other physicochemical covariates fixed. VA has an
# uncontroversial causal direction (tasters penalise it by definition) and
# is a fault endpoint rather than upstream of the other chemistry, so
# adjusting for the rest does not block its causal path.
#
#   Y = quality (ordinal)              -> surrogate-completed
#   D = volatile acidity (continuous)  -> treatment
#   X = the other 10 chemistry covariates
#
# Method: ordinal SUDO -- cumulative-link (clm) full model, proper MI draws
# (thresholds + coefficients redrawn from the clm covariance), cross-fitted
# gam partialling nuisances E[S|X], E[D|X], Rubin pooling. Continuous D just
# means E[D|X] is a gaussian gam. Two full-model specifications are run as a
# robustness check on the covariate functional form:
#   linear     -- clm(quality ~ D + X)
#   PL-spline  -- clm(quality ~ D + ns(X_j, 3)): treatment linear, each
#                 covariate a natural spline (the partially-linear structure
#                 the methodology recommends; D still cannot interact with X)
# A stable theta across both is the robustness the pass-through theory says
# to check, since full-model error is not orthogonalised away.
# Reported against the naive "read the coefficient straight off the full
# model" baseline.
#
# Run from repo root: Rscript R/wine_application.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
suppressPackageStartupMessages({
  library(ordinal); library(mgcv); library(splines)
})

DATA <- "manuscript/data/wine"

analyse <- function(which, B = 50, n_folds = 5, seed = 1) {
  set.seed(seed)
  d <- read.csv(file.path(DATA, sprintf("winequality-%s.csv", which)),
                sep = ";")
  y <- as.integer(factor(d$quality))          # ordinal codes 1..J
  D <- as.numeric(scale(d$volatile.acidity))  # standardized treatment
  covs <- setdiff(names(d), c("volatile.acidity", "quality"))
  X <- as.data.frame(scale(d[covs]))
  n <- nrow(X)
  r2_overlap <- summary(lm(D ~ ., data = cbind(D = D, X)))$r.squared

  folds <- make_folds(n, n_folds)
  D_res <- D - crossfit(X, D, fit_gam, folds)  # continuous D -> gaussian gam
  dat <- data.frame(Y = factor(y), D = D, X)

  # one full-model specification -> ordinal SUDO estimate
  sudo_spec <- function(rhs) {
    fit <- clm(as.formula(paste("Y ~ D +", rhs)), data = dat)
    alpha <- fit$alpha; beta <- fit$beta
    par_hat <- c(alpha, beta)
    V <- vcov(fit)[names(par_hat), names(par_hat)]
    mm <- model.matrix(fit)$X[, names(beta), drop = FALSE]  # index design
    J <- length(alpha) + 1
    b_idx <- seq(J, length(par_hat))
    cuts <- function(par) c(-Inf, par[1:(J - 1)], Inf)

    S_hat <- crossfit(X, complete_surrogate(
      y, as.numeric(mm %*% par_hat[b_idx]), cuts(par_hat), "logit"),
      fit_gam, folds)
    draws <- sapply(seq_len(B), function(b) {
      par <- MASS::mvrnorm(1, par_hat, V)
      S <- complete_surrogate(y, as.numeric(mm %*% par[b_idx]), cuts(par),
                              "logit")
      f <- fwl_theta(S - S_hat, D_res)
      c(theta = f$theta, var = f$var)
    })
    p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
    co <- summary(fit)$coefficients["D", ]
    list(naive_theta = unname(co["Estimate"]), naive_se = unname(co["Std. Error"]),
         theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi)
  }

  list(which = which, n = n, J = max(y), r2_overlap = r2_overlap,
       linear = sudo_spec(paste(covs, collapse = " + ")),
       pl_spline = sudo_spec(paste(sprintf("ns(%s, 3)", covs),
                                   collapse = " + ")))
}

cat("Wine quality application: effect of volatile acidity on ordinal quality\n")
cat("theta = latent-quality effect per 1 SD of volatile acidity; expect < 0.\n\n")

rows <- list()
for (which in c("red", "white")) {
  r <- analyse(which)
  cat(sprintf("== %s wine  (n=%d, J=%d ordinal levels) ==\n", r$which, r$n, r$J))
  cat(sprintf("  overlap R2(D ~ X) = %.3f  (healthy; Plan B was 0.98-0.997)\n",
              r$r2_overlap))
  cat(sprintf("  naive clm read-off        : theta = %+.3f  se = %.3f\n",
              r$linear$naive_theta, r$linear$naive_se))
  cat(sprintf("  SUDO, linear full model   : theta = %+.3f  se = %.3f  95%% CI [%.3f, %.3f]\n",
              r$linear$theta, r$linear$se, r$linear$ci_lo, r$linear$ci_hi))
  cat(sprintf("  SUDO, PL-spline full model: theta = %+.3f  se = %.3f  95%% CI [%.3f, %.3f]\n\n",
              r$pl_spline$theta, r$pl_spline$se, r$pl_spline$ci_lo,
              r$pl_spline$ci_hi))
  for (spec in c("linear", "pl_spline")) {
    s <- r[[spec]]
    rows[[length(rows) + 1]] <- data.frame(
      wine = r$which, n = r$n, n_levels = r$J, overlap_r2 = r$r2_overlap,
      full_model = spec, naive_theta = s$naive_theta, naive_se = s$naive_se,
      sudo_theta = s$theta, sudo_se = s$se, ci_lo = s$ci_lo, ci_hi = s$ci_hi)
  }
}
sm <- do.call(rbind, rows)

# VA is a fault: the effect on latent quality must be clearly negative, and
# stable across the two full-model specifications (robustness to functional
# form, which the pass-through theory says to check)
stopifnot(all(sm$sudo_theta < 0), all(sm$ci_hi < 0))
for (w in c("red", "white")) {
  ths <- sm$sudo_theta[sm$wine == w]
  stopifnot(abs(diff(ths)) < 0.15)  # linear vs PL-spline agree
}
cat("PASS: volatile acidity clearly lowers latent quality in both wines,",
    "stable across linear and PL-spline full models.\n")

dir.create("R/results", showWarnings = FALSE)
write.csv(sm, "R/results/wine_application.csv", row.names = FALSE)
cat("wrote R/results/wine_application.csv\n")
