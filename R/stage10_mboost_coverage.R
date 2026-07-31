# Stage 10: does SUDO with an ML imputation model beat reading the
# coefficient off that same model, on coverage?
#
# Stage 9 established the mechanism. Regularisation that shrinks the
# treatment base-learner (boosting) biases the direct coefficient, and the
# completion recovers a fixed fraction of that bias equal to the Proposition
# A3 pass-through factor. In the mstop path the empirical ratio was 0.560
# against a theoretical cbar_1 of 0.605, flat across a 64-fold change in
# regularisation. There is an operating window (mstop 6400 to 12800 for this
# design) where SUDO is unbiased while direct read-off is still shrunk.
#
# The open question is whether that bias advantage survives into INFERENCE.
# This stage answers it with the only comparison that matters: SUDO and the
# direct coefficient, from the SAME fitted learner, on the SAME outer
# bootstrap resamples, so the arms are paired and differ only in what is
# computed from each fit.
#
#   sudo    completion + FWL, full-pipeline bootstrap SE
#   direct  alpha_hat = v(1,X) - v(0,X), same bootstrap SE
#
# alpha_hat is read as a contrast rather than via coef(): mboost's default
# Binomial(type = "adaboost") fits HALF the logit, so a naive coef() read
# would be out by a factor of two. The contrast is parameterisation-proof and
# is exactly what a user would compute.
#
# mstop is FIXED, not tuned. That is deliberate and is a stated limitation:
# it was chosen from stage 9's bias path, which used the true theta. This
# stage therefore establishes that the estimator CAN attain nominal coverage
# with an ML imputation model, not that the tuning is available in practice.
# cvrisk is not used: stage 3l showed it selects deep in the shrinkage
# regime, which stage 9 now explains.
#
# Run from the repository root:
#   Rscript R/stage10_mboost_coverage.R
#
# Optional smoke-test overrides:
#   SUDO_STAGE10_N=600 SUDO_STAGE10_REPS=3 SUDO_STAGE10_BOOT=5 \
#     SUDO_STAGE10_MSTOP=400 Rscript R/stage10_mboost_coverage.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages({library(mboost); library(mgcv)})

env_int <- function(name, default) {
  v <- Sys.getenv(name, unset = "")
  if (nzchar(v)) as.integer(v) else as.integer(default)
}

N_OBS   <- env_int("SUDO_STAGE10_N", 2000)
N_REPS  <- env_int("SUDO_STAGE10_REPS", 100)
B_OUTER <- env_int("SUDO_STAGE10_BOOT", 40)
INNER_B <- env_int("SUDO_STAGE10_INNER", 10)
MSTOP   <- env_int("SUDO_STAGE10_MSTOP", 400)
THETA0  <- 1.5

g0_fun <- function(X) sin(2 * X[, 1]) + 0.5 * X[, 2]^2
m0_fun <- function(X) plogis(1.5 * X[, 1] + 0.8 * X[, 2])
clip <- function(p) pmin(pmax(p, 1e-6), 1 - 1e-6)

dgp10 <- function(n) {
  X <- matrix(rnorm(n * 2), n, 2, dimnames = list(NULL, c("X1", "X2")))
  D <- rbinom(n, 1, m0_fun(X))
  Y <- rbinom(n, 1, plogis(THETA0 * D + g0_fun(X)))
  list(X = X, D = D, Y = Y)
}

# Cross-fitted partially-linear mboost index. Returns the out-of-fold index
# at the observed D, and alpha_hat as the fold-size-weighted average of
# v(1,x) - v(0,x). The PL structure makes that contrast constant in x, which
# the stage asserts rather than assumes.
fit_pl_mboost_both <- function(d, folds, mstop) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  dat <- data.frame(Y = factor(d$Y, levels = c(0, 1)), D = d$D, X)
  rhs <- paste(c("bols(D)", sprintf("bbs(%s)", colnames(X))), collapse = " + ")
  fml <- as.formula(paste("Y ~", rhs))

  lp_hat <- numeric(n)
  alpha_k <- numeric(length(folds))
  pl_dev <- numeric(length(folds))
  for (k in seq_along(folds)) {
    test <- folds[[k]]
    train <- setdiff(seq_len(n), test)
    m <- suppressWarnings(mboost::gamboost(
      fml, data = dat[train, , drop = FALSE], family = mboost::Binomial(),
      control = mboost::boost_control(mstop = mstop)))
    lp <- function(nd) qlogis(clip(as.numeric(
      predict(m, newdata = nd, type = "response"))))
    te <- dat[test, , drop = FALSE]
    lp_hat[test] <- lp(te)
    contrast <- lp(transform(te, D = 1)) - lp(transform(te, D = 0))
    alpha_k[k] <- mean(contrast)
    pl_dev[k] <- stats::sd(contrast)
  }
  list(lp_hat = lp_hat,
       alpha = sum(alpha_k * lengths(folds)) / n,
       pl_dev = max(pl_dev))
}

# UNCONSTRAINED index: a single-hidden-layer net fit on (D, X) jointly with
# no partially-linear restriction, so v(1,x) - v(0,x) varies with x and NO
# constant treatment coefficient exists. This is the setting where the
# completion is the only route to a scalar latent-scale effect; the only
# available competitor is the averaged index contrast (g-computation), which
# is a different estimand and also needs a bootstrap. A net rather than a
# forest because the repo found a forest's piecewise-constant index
# attenuates (see fit_blackbox_index).
fit_unconstrained_both <- function(d, folds, size = 8, decay = 0.01) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  feat <- data.frame(X, D = d$D)
  ydat <- factor(d$Y)
  lp_hat <- numeric(n)
  contrast <- numeric(n)
  for (test in folds) {
    train <- setdiff(seq_len(n), test)
    m <- nnet::nnet(ydat[train] ~ ., data = feat[train, , drop = FALSE],
                    size = size, decay = decay, maxit = 500, trace = FALSE)
    pr <- function(nd) qlogis(clip(as.numeric(predict(m, newdata = nd))))
    te <- feat[test, , drop = FALSE]
    lp_hat[test] <- pr(te)
    contrast[test] <- pr(transform(te, D = 1)) - pr(transform(te, D = 0))
  }
  list(lp_hat = lp_hat, alpha = mean(contrast), pl_dev = stats::sd(contrast))
}

# One dataset -> point estimates for every arm, all from cross-fitted indices
# on the SAME folds so the comparison is paired at the fit level.
estimate_both <- function(d, mstop, inner_B, n_folds = 5L) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  folds <- make_folds(n, n_folds)
  d_res <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)

  sudo_from <- function(lp) mean(vapply(seq_len(inner_B), function(b) {
    S <- complete_surrogate(d$Y + 1L, lp, c(-Inf, 0, Inf), "logit")
    ell <- crossfit(X, S, fit_gam, folds)
    fwl_theta(S - ell, d_res)$theta
  }, numeric(1)))

  pl <- fit_pl_mboost_both(d, folds, mstop)
  unc <- fit_unconstrained_both(d, folds)
  list(sudo = sudo_from(pl$lp_hat), direct = pl$alpha,
       sudo_unc = sudo_from(unc$lp_hat), gcomp_unc = unc$alpha,
       pl_dev = pl$pl_dev, unc_dev = unc$pl_dev)
}

one_rep <- function(seed, mstop, b_outer, inner_B, n_obs) {
  set.seed(seed)
  d <- dgp10(n_obs)
  point <- estimate_both(d, mstop, inner_B)
  n <- length(d$Y)
  boot <- do.call(rbind, lapply(seq_len(b_outer), function(b) {
    idx <- sample.int(n, n, replace = TRUE)
    db <- list(X = d$X[idx, , drop = FALSE], D = d$D[idx], Y = d$Y[idx])
    e <- estimate_both(db, mstop, inner_B)
    data.frame(sudo = e$sudo, direct = e$direct,
               sudo_unc = e$sudo_unc, gcomp_unc = e$gcomp_unc)
  }))
  mk <- function(est, bs, dev) {
    se <- stats::sd(bs)
    q <- stats::quantile(bs, c(0.025, 0.975), names = FALSE)
    data.frame(est = est, se = se,
               lo = est - 1.96 * se, hi = est + 1.96 * se,
               plo = q[1], phi = q[2], contrast_sd = dev)
  }
  rbind(cbind(arm = "sudo_pl",   mk(point$sudo,      boot$sudo,      point$pl_dev)),
        cbind(arm = "direct_pl", mk(point$direct,    boot$direct,    point$pl_dev)),
        cbind(arm = "sudo_unc",  mk(point$sudo_unc,  boot$sudo_unc,  point$unc_dev)),
        cbind(arm = "gcomp_unc", mk(point$gcomp_unc, boot$gcomp_unc, point$unc_dev)),
        make.row.names = FALSE)
}

cl <- mc_cluster(c("dgp10", "g0_fun", "m0_fun", "clip", "fit_pl_mboost_both",
                   "fit_unconstrained_both", "estimate_both", "one_rep",
                   "THETA0"))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages({
  library(mboost); library(nnet)
})))
parallel::clusterExport(cl, c("MSTOP", "B_OUTER", "INNER_B", "N_OBS"),
                        envir = environment())

cat(sprintf(paste0("Stage 10: mboost PL imputation, n=%d, mstop=%d (fixed),\n",
                   "  %d replications, %d outer resamples, %d completions\n\n"),
            N_OBS, MSTOP, N_REPS, B_OUTER, INNER_B))

res <- do.call(rbind, parallel::parLapply(cl, seq_len(N_REPS), function(r) {
  one_rep(10000 + r, MSTOP, B_OUTER, INNER_B, N_OBS)
}))
parallel::stopCluster(cl)

agg <- do.call(rbind, lapply(split(res, res$arm), function(b) data.frame(
  arm = b$arm[1], n_reps = nrow(b),
  bias = mean(b$est) - THETA0,
  mc_se_bias = stats::sd(b$est) / sqrt(nrow(b)),
  sd = stats::sd(b$est),
  mean_se = mean(b$se),
  sd_over_mean_se = stats::sd(b$est) / mean(b$se),
  coverage_normal = mean(b$lo <= THETA0 & THETA0 <= b$hi),
  coverage_pct = mean(b$plo <= THETA0 & THETA0 <= b$phi),
  contrast_sd = mean(b$contrast_sd))))
agg$mc_se_cov <- sqrt(agg$coverage_normal * (1 - agg$coverage_normal) /
                        agg$n_reps)
agg <- agg[match(c("sudo_pl","direct_pl","sudo_unc","gcomp_unc"), agg$arm), ]

cat(sprintf("%-10s %9s %8s %9s %10s %11s %12s\n",
            "arm", "bias", "sd", "mean_se", "cover_nrm", "cover_pct", "contrast_sd"))
for (i in seq_len(nrow(agg))) {
  cat(sprintf("%-10s %+9.4f %8.4f %9.4f %10.3f %11.3f %12.4f\n",
              agg$arm[i], agg$bias[i], agg$sd[i], agg$mean_se[i],
              agg$coverage_normal[i], agg$coverage_pct[i], agg$contrast_sd[i]))
}
cat(sprintf("\nMonte Carlo SE on coverage: %.3f\n", max(agg$mc_se_cov)))

# Computational invariants only. Whether SUDO beats direct read-off is the
# question under test, so nothing here presupposes an answer.
stopifnot(
  nrow(agg) == 4L,
  all(agg$n_reps == N_REPS),
  all(is.finite(unlist(agg[, sapply(agg, is.numeric)]))),
  all(agg$sd > 0), all(agg$mean_se > 0)
)
cat("PASS: both arms finite over all replications on common resamples\n")

dir.create("R/results", showWarnings = FALSE)
write.csv(agg, "R/results/stage10_mboost_coverage.csv", row.names = FALSE)
cat("wrote R/results/stage10_mboost_coverage.csv\n")
