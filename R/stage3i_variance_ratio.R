# Stage 3i: does treatment-conditional heteroscedasticity predict theta
# bias? (the lead from stage 3h: the untuned net's residual variance ratio
# Var(R|D=1)/Var(R|D=0) was 0.89 at theta=3, the furthest from 1 of any
# learner, while mean-structure and index-calibration diagnostics both
# failed to predict theta bias.)
#
# This is a rigorous version of that check: for each surrogate residual
# R = S - V_hat, run var.test(R[D==1], R[D==0]) — a proper F-test for
# unequal variances, not just the raw ratio — and correlate both the
# ratio and the test's significance against the single-draw theta bias
# on the SAME dataset. Same seeds, same DGP, same 4 learners, same theta
# in {1.5, 3}, 100 reps as stage 3h, so var_ratio here should numerically
# match stage 3h's (a built-in consistency check) and everything else is
# a direct extension.
# Run from repo root: Rscript R/stage3i_variance_ratio.R

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

get_lp_hat <- function(d, learner, folds, nn_decay = 0.01) {
  if (learner == "gam") {
    crossfit_fullmodel_gam(d$Y, d$D, d$X, folds)$lp_hat
  } else {
    fit_blackbox_index(d, learner, TRUE, folds, FALSE, 8, nn_decay)$lp_hat
  }
}

diag_one <- function(d, learner, nn_decay, theta_true) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  folds <- make_folds(n, 5)
  lp_hat <- get_lp_hat(d, learner, folds, nn_decay)

  S <- complete_surrogate(d$Y + 1L, lp_hat, c(-Inf, 0, Inf), "logit")
  R <- S - lp_hat

  vt <- var.test(R[d$D == 1], R[d$D == 0])
  var_ratio <- unname(vt$estimate)
  p_het <- vt$p.value

  D_res <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)
  S_hat <- crossfit(X, S, fit_gam, folds)
  theta_hat <- fwl_theta(S - S_hat, D_res)$theta

  c(var_ratio = var_ratio, p_het = p_het, theta_hat = theta_hat,
    theta_bias = theta_hat - theta_true)
}

cl <- mc_cluster(c("dgp_eta", "get_lp_hat", "diag_one"))
invisible(parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages({library(nnet); library(ranger)})
  source("R/sudo/estimator.R")
}))

learners <- list(
  list(name = "gam",        learner = "gam",    decay = 0.01),
  list(name = "ranger",     learner = "ranger", decay = 0.01),
  list(name = "nnet_d0.01", learner = "nnet",   decay = 0.01),
  list(name = "nnet_d0.1",  learner = "nnet",   decay = 0.1))

n_reps <- 100
cat(sprintf("treatment-conditional variance diagnostic vs theta bias, n = 2000, %d reps\n",
            n_reps))
cat("var_ratio = Var(R|D=1)/Var(R|D=0), p_het from var.test();",
    "same seeds as stage 3h\n\n")

rows <- list()
cross_learner <- list()
for (th in c(1.5, 3)) {
  parallel::clusterExport(cl, "th", envir = environment())
  for (L in learners) {
    parallel::clusterExport(cl, "L", envir = environment())
    res <- parallel::parSapply(cl, seq_len(n_reps), function(r) {
      set.seed(2e5 + r)
      d <- dgp_eta(2000, th)
      diag_one(d, L$learner, L$decay, th)
    })
    res <- t(res)
    frac_sig <- mean(res[, "p_het"] < 0.05)
    ratio_dev <- abs(res[, "var_ratio"] - 1)
    corr_ratio <- cor(ratio_dev, abs(res[, "theta_bias"]))
    corr_p <- cor(-log10(res[, "p_het"]), abs(res[, "theta_bias"]))
    tt <- t.test(res[, "var_ratio"], mu = 1)
    cat(sprintf(
      "theta=%.1f %-12s mean var_ratio %.3f (t-test vs 1: p=%.2e)  frac p_het<0.05: %.2f  mean|theta_bias| %.3f  cor(|ratio-1|,|bias|) %+.3f  cor(-log10 p,|bias|) %+.3f\n",
      th, L$name, mean(res[, "var_ratio"]), tt$p.value, frac_sig,
      mean(abs(res[, "theta_bias"])), corr_ratio, corr_p))
    cross_learner[[length(cross_learner) + 1]] <- data.frame(
      theta = th, learner = L$name,
      mean_var_ratio = mean(res[, "var_ratio"]),
      var_ratio_ttest_p = tt$p.value,
      frac_p_het_sig = frac_sig,
      mean_abs_theta_bias = mean(abs(res[, "theta_bias"])),
      cor_ratio_dev = corr_ratio, cor_neglogp = corr_p, n_reps = n_reps)
    rows[[length(rows) + 1]] <- data.frame(
      theta = th, learner = L$name, rep = seq_len(n_reps),
      var_ratio = res[, "var_ratio"], p_het = res[, "p_het"],
      theta_hat = res[, "theta_hat"], theta_bias = res[, "theta_bias"])
  }
  cat("\n")
}
parallel::stopCluster(cl)
sm <- do.call(rbind, cross_learner)
detail <- do.call(rbind, rows)

cat("== cross-learner ranking check ==\n")
for (th in c(1.5, 3)) {
  s <- sm[sm$theta == th, ]
  ord_ratio <- s$learner[order(-abs(s$mean_var_ratio - 1))]
  ord_theta <- s$learner[order(-s$mean_abs_theta_bias)]
  cat(sprintf("theta=%.1f  ranking by |var_ratio-1| (worst first): %s\n",
              th, paste(ord_ratio, collapse = " > ")))
  cat(sprintf("theta=%.1f  ranking by |theta_bias| (worst first):  %s\n",
              th, paste(ord_theta, collapse = " > ")))
}

# consistency check against stage 3h's var_ratio (same seeds/draws)
h <- read.csv("R/results/stage3h_summary.csv")
for (i in seq_len(nrow(sm))) {
  h_row <- h[h$learner == sm$learner[i] & h$theta == sm$theta[i], ]
  if (nrow(h_row) == 1) {
    diff <- abs(sm$mean_var_ratio[i] - h_row$mean_var_ratio)
    if (diff > 1e-6) {
      cat(sprintf("WARNING: var_ratio mismatch vs stage 3h for %s theta=%s: %.6f vs %.6f\n",
                  sm$learner[i], sm$theta[i], sm$mean_var_ratio[i],
                  h_row$mean_var_ratio))
    }
  }
}
cat("consistency check vs stage 3h var_ratio: done (see WARNINGs above, if any)\n")

write.csv(sm, "R/results/stage3i_summary.csv", row.names = FALSE)
write.csv(detail, "R/results/stage3i_detail.csv", row.names = FALSE)
cat("wrote R/results/stage3i_summary.csv and stage3i_detail.csv\n")
