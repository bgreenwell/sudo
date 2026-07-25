# Stage 3g: is the full model's fitted index calibrated against the truth?
#
# Motivation: the Neyman-orthogonality guarantee (sudo.md Sec 5) covers the
# PARTIALLING nuisances E[S|X], E[D|X] — first-order errors there wash out
# of theta_hat by construction. It does NOT cover the full (imputation)
# model: errors in V_hat shift where each surrogate draw is centered, and
# that error is baked into the completed data itself, not differenced away.
# Stages 3d-3f found theta bias/coverage differences across full-model
# choices; this stage checks the mechanism directly, in simulation, where
# the true systematic index eta = theta*D + g(X) is known: regress
# eta ~ V_hat and read off the slope (1 = perfectly calibrated, <1 =
# attenuation toward the mean, >1 = overshoot) and R^2 (how much of the
# index variation V_hat captures at all).
#
# Learners: gam (known-unbiased control), ranger, nnet decay=0.01 (the
# untuned trap from 3f), nnet decay=0.1 (tuned fix). n = 2000, theta in
# {1.5, 3}, 100 reps each. No surrogate draws, no B loop, no FWL — just
# the cross-fitted index vs the truth, so this is cheap and isolates
# full-model quality from everything downstream.
#
# Result (see NOTE at the bottom of the script): index-scale calibration
# does NOT map monotonically onto theta bias. All black-box learners
# attenuate the index (slope < 1); ranger's index fit is the WORST of the
# four (theta=3: slope 0.57, R2 0.41) yet its theta point estimate is the
# best (stage 3d/3f: bias ~0); nnet decay=0.01 has a BETTER index fit
# (slope 0.65, R2 0.63) yet the worst theta bias (+0.43 to +0.47). A
# well-specified structural model (gam) is trustworthy on this diagnostic;
# a black-box model's index calibration is necessary to check but not
# sufficient — it cannot substitute for full theta-level MC validation.
# Run from repo root: Rscript R/stage3g_index_calibration.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")

dgp_eta <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  eta <- theta * D + g
  U <- eta + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0), eta = eta)
}

# one rep: fit the cross-fitted index for a given learner, regress the true
# systematic index on it
calib_one <- function(d, learner, nn_decay = 0.01) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  folds <- make_folds(n, 5)
  lp_hat <- if (learner == "gam") {
    crossfit_fullmodel_gam(d$Y, d$D, d$X, folds)$lp_hat
  } else {
    fit_blackbox_index(d, learner, TRUE, folds, FALSE, 8, nn_decay)$lp_hat
  }
  fit <- lm(d$eta ~ lp_hat)
  c(slope = unname(coef(fit)[2]), intercept = unname(coef(fit)[1]),
    r2 = summary(fit)$r.squared)
}

cl <- mc_cluster(c("dgp_eta", "calib_one"))
invisible(parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages({library(nnet); library(ranger)})
  source("R/sudo/estimator.R")
}))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp_eta(n, theta)
}

learners <- list(
  list(name = "gam",         learner = "gam",   decay = NA),
  list(name = "ranger",      learner = "ranger", decay = NA),
  list(name = "nnet_d0.01",  learner = "nnet",  decay = 0.01),
  list(name = "nnet_d0.1",   learner = "nnet",  decay = 0.1))

n_reps <- 100
cat(sprintf("index calibration: eta ~ V_hat, n = 2000, %d reps\n", n_reps))
cat("slope 1 = calibrated, <1 = attenuated, >1 = overshoot; R2 = index",
    "variance explained\n\n")

rows <- list()
for (th in c(1.5, 3)) {
  parallel::clusterExport(cl, "th", envir = environment())
  for (L in learners) {
    parallel::clusterExport(cl, c("L"), envir = environment())
    res <- parallel::parSapply(cl, seq_len(n_reps), function(r) {
      set.seed(1e5 + r)
      d <- dgp_eta(2000, th)
      calib_one(d, L$learner, if (is.na(L$decay)) 0.01 else L$decay)
    })
    m <- rowMeans(res)
    se <- apply(res, 1, sd) / sqrt(n_reps)
    cat(sprintf("theta=%.1f %-12s slope %.3f (se %.3f)  intercept %+.3f (se %.3f)  R2 %.3f\n",
                th, L$name, m["slope"], se["slope"], m["intercept"],
                se["intercept"], m["r2"]))
    rows[[length(rows) + 1]] <- data.frame(
      stage = "3g", theta = th, learner = L$name,
      slope = m["slope"], slope_se = se["slope"],
      intercept = m["intercept"], intercept_se = se["intercept"],
      r2 = m["r2"], n_reps = n_reps)
  }
  cat("\n")
}
parallel::stopCluster(cl)
sm <- do.call(rbind, rows)

g <- function(learner, th) sm[sm$learner == learner & sm$theta == th, ]
# gam must be well-calibrated (known-unbiased control) and explain the most
# index variance of any learner at both signal strengths
stopifnot(abs(g("gam", 1.5)$slope - 1) < 0.1)
stopifnot(abs(g("gam", 3)$slope - 1) < 0.15)
for (th in c(1.5, 3)) {
  r2s <- c(ranger = g("ranger", th)$r2, nnet_d0.01 = g("nnet_d0.01", th)$r2,
          nnet_d0.1 = g("nnet_d0.1", th)$r2)
  stopifnot(g("gam", th)$r2 > max(r2s))
}
# tuning improves index-variance capture at both signal strengths (even
# though, surprisingly below, it does not cleanly predict theta bias)
stopifnot(g("nnet_d0.1", 1.5)$r2 > g("nnet_d0.01", 1.5)$r2)
stopifnot(g("nnet_d0.1", 3)$r2 > g("nnet_d0.01", 3)$r2)
stopifnot(g("nnet_d0.1", 1.5)$r2 > g("ranger", 1.5)$r2)
stopifnot(g("nnet_d0.1", 3)$r2 > g("ranger", 3)$r2)
cat("PASS: gam best-calibrated; tuned nnet explains more index variance",
    "than ranger or the untuned net at both signal strengths\n")
cat("\nNOTE (documented, not asserted): index calibration does NOT",
    "monotonically predict downstream theta bias. At theta=3, ranger has",
    "the WORST index fit here (slope 0.57, R2 0.41) yet the best theta",
    "point estimate (stage 3d/3f: bias ~0); nnet_d0.01 has BETTER index",
    "fit (slope 0.65, R2 0.63) yet the worst theta bias (+0.43 to +0.47).",
    "A well-calibrated full model (gam) is reliable; a black-box full",
    "model's index-scale calibration is a necessary sanity check but not",
    "sufficient to predict theta bias -- full end-to-end MC validation",
    "of theta remains necessary and cannot be skipped.\n")

write.csv(sm, "R/results/stage3g_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3g_summary.csv\n")
