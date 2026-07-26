# Bayesian (BART) imputation models for SUDO, two arms.
#
# Both return the list(lp_hat, draw_lp) shape crossfit_fullmodel_gam() returns,
# so sudo_binary()'s gam branch works unchanged: draw_lp() hands back draw b's
# out-of-fold index and the caller completes the surrogate at it.
#
# Why a posterior is the natural draw scheme. Rubin's proper-MI requirement is
# that each completion come from a fresh draw of the imputation model itself.
# A parametric model supplies that from its covariance; a black box has none,
# which is why stages 3d-3p reach for a bootstrap. A Bayesian ensemble supplies
# it directly: every posterior sweep IS a complete alternative imputation
# model. This tests whether that works in practice.
#
# The augmentation identity. Under a probit imputation model the Albert-Chib
# draw Z_i | Y_i, eta ~ N(eta_i, 1) truncated to the interval Y_i implies is
# exactly the SUDO surrogate (rnorm_trunc() in surrogate.R). In sample, the
# Gibbs chain therefore produces surrogate completions as a byproduct. Out of
# fold we predict the index and complete in the usual way, which is what the
# estimator does, so the identity is an observation about the sampler rather
# than a shortcut the pipeline takes.
#
# Two arms:
#   partially_linear = TRUE   V = beta*D + f(X), BART on X only, beta drawn
#                             in its own Gibbs step. D cannot interact with X.
#   partially_linear = FALSE  V = f(D, X), BART on everything. The control:
#                             stages 3j-3l predict this is biased, because a
#                             sum of trees is free to interact D with X.
#
# Requires dbarts; fwl.R (make_folds) and surrogate.R (rnorm_trunc) sourced.

# One Gibbs chain per fold. Returns an (n_test x n_draws) matrix of out-of-fold
# index draws, thinned, plus the retained beta chain for the PL arm.
.bart_gibbs_fold <- function(Y, D, X, train, test, partially_linear,
                             n_draws, burn, thin, n_trees) {
  Ytr <- Y[train]; Dtr <- D[train]
  Xtr <- X[train, , drop = FALSE]
  Xte <- X[test, , drop = FALSE]; Dte <- D[test]

  # binary cutpoint at 0: Y=1 truncates to (0, Inf), Y=0 to (-Inf, 0]
  lo <- ifelse(Ytr == 1, 0, -Inf)
  hi <- ifelse(Ytr == 1, Inf, 0)

  Btr <- if (partially_linear) Xtr else cbind(D = Dtr, Xtr)
  Bte <- if (partially_linear) Xte else cbind(D = Dte, Xte)

  sampler <- dbarts::dbarts(
    resp ~ ., data = cbind(resp = rep(0, length(train)), Btr), test = Bte,
    control = dbarts::dbartsControl(n.samples = 1L, n.burn = 0L, n.chains = 1L,
                                    n.threads = 1L, n.trees = n_trees,
                                    updateState = TRUE, verbose = FALSE),
    sigma = 1.0)

  beta <- 0
  f_tr <- rep(0, length(train)); f_te <- rep(0, length(test))
  out <- matrix(NA_real_, length(test), n_draws)
  beta_keep <- numeric(if (partially_linear) n_draws else 0)
  dd <- sum(Dtr^2)
  kept <- 0L

  for (sweep in seq_len(burn + n_draws * thin)) {
    # (1) augmentation: Z is the surrogate at the current index
    eta_tr <- if (partially_linear) beta * Dtr + f_tr else f_tr
    Z <- rnorm_trunc(eta_tr, lo, hi)

    # (2) f | Z, beta. sigma is known to be 1 under probit augmentation, so
    # pin it before each sweep and discard the sampler's own sigma draw.
    sampler$setResponse(if (partially_linear) Z - beta * Dtr else Z)
    sampler$setSigma(1)
    s <- sampler$run(0L, 1L)
    f_tr <- s$train[, 1]; f_te <- s$test[, 1]

    # (3) beta | Z, f, unit error variance
    if (partially_linear) {
      beta <- rnorm(1, sum(Dtr * (Z - f_tr)) / dd, sqrt(1 / dd))
    }

    if (sweep > burn && (sweep - burn) %% thin == 0L) {
      kept <- kept + 1L
      out[, kept] <- if (partially_linear) beta * Dte + f_te else f_te
      if (partially_linear) beta_keep[kept] <- beta
    }
  }
  list(index = out, beta = beta_keep)
}

# Cross-fitted BART imputation model. BART is a flexible learner, so the index
# must be out-of-fold (one chain per fold); fitting once on all the data would
# leak Y_i into V_hat_i at first order.
fit_bart_index <- function(d, folds, partially_linear = TRUE, B = 25,
                           burn = 200, thin = 5, n_trees = 50) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  LP <- matrix(NA_real_, n, B)
  betas <- list()

  for (test in folds) {
    fold <- .bart_gibbs_fold(d$Y, d$D, X, setdiff(seq_len(n), test), test,
                             partially_linear, B, burn, thin, n_trees)
    LP[test, ] <- fold$index
    if (partially_linear) betas[[length(betas) + 1]] <- fold$beta
  }

  # Rubin's rules assume independent imputations while Gibbs sweeps are
  # serially correlated, so carry the retained lag-1 autocorrelation out for
  # the caller to report. Residual autocorrelation would deflate B_between and
  # under-cover in the same way an improper draw does.
  acf1 <- if (partially_linear && B >= 10) {
    mean(vapply(betas, function(b) {
      if (stats::sd(b) < 1e-12) 0 else stats::acf(b, lag.max = 1,
                                                  plot = FALSE)$acf[2]
    }, numeric(1)))
  } else NA_real_

  i_draw <- 0L
  list(lp_hat = rowMeans(LP),
       draw_lp = function() { i_draw <<- i_draw + 1L; LP[, i_draw] },
       acf1 = acf1)
}
