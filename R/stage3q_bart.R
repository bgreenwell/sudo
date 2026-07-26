# Stage 3q: can a Bayesian imputation model supply proper draws directly?
#
# Every black-box result so far (stages 3d-3p) manufactures proper draws by
# resampling, because a black box has no covariance to draw from. A Bayesian
# ensemble has something better: each posterior sweep IS a complete alternative
# imputation model, which is exactly what Rubin's proper-MI requires. This
# stage tests whether that works in practice.
#
# Under a probit imputation model the Albert-Chib augmentation draw
#   Z_i | Y_i, eta ~ N(eta_i, 1) truncated to the interval Y_i implies
# is identical to the SUDO surrogate, so in sample the Gibbs chain produces
# completions as a byproduct (see R/sudo/bart.R).
#
# DGP: a probit twin of the stage-3 DGP. Dividing the systematic part by
# sd(logistic) = pi/sqrt(3) matches the signal-to-noise ratio, so event rates
# line up with stage 3 (0.786 -> 0.781 at theta_l=1.5, 0.846 -> 0.840 at 3)
# and theta_probit = theta_logit / 1.8138. Arms are labelled by theta_logit so
# they read against the stage 3m/3p tables; the estimand is theta_probit.
#
# Four arms, same DGP and seeds:
#   C  PL-BART       V = beta*D + f(X), BART on X only     posterior draws
#   D  vanilla BART  V = f(D, X)                           posterior draws
#   A  PL-backfit    tuned (size 4, decay 0.3, n_iter 5)   recentered boot
#   B  PL-backfit    same                                  full-pipeline boot
#
# D is the control. Stages 3j-3l predict it is biased: a sum of trees is free
# to interact D with X, and a posterior fixes variance propagation, not bias.
# If D shows sd/mean_se ~ 1 with worse bias while C is unbiased, that
# separates "posterior fixes variance" from "structure fixes bias".
#
# Caveat on the reference arms. The PL-backfit config (size 4, decay 0.3,
# n_iter 5) was selected by stage 3m on the LOGISTIC DGP, by MC theta bias.
# Nothing retunes it for the probit link, so A and B are a transferred-tuning
# baseline rather than a best-case one, and may understate what a properly
# tuned backfit would do here. Read a BART win over them with that in mind.
#
# fit_pl_backfit is link-aware; the other black-box fitters are logit-only and
# sudo_binary now errors rather than silently rescaling theta by the link
# ratio (a logit index scored against a probit completion is out by ~1.8).
#
# Reference (stage 3m/3p, logistic DGP, not directly comparable):
#   3m theta=1.5 bias -0.027 cover 0.950 sd/mean_se 1.12
#      theta=3   bias -0.028 cover 0.860 sd/mean_se 1.31
#   3p theta=3                cover 0.95  sd/mean_se 0.94
#
# This is an exploratory stage, so the assertions cover mechanical correctness
# (finite estimates, negligible MCMC autocorrelation) and the scientific
# verdict is printed for a human to read.
#
# Run from repo root: Rscript R/stage3q_bart.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")
source("R/sudo/pl.R")
source("R/sudo/bart.R")

SD_LOGIS <- pi / sqrt(3)

dgp3_probit <- function(n, theta_l) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  lat <- (theta_l * D + g) / SD_LOGIS + rnorm(n)
  list(X = X, D = D, Y = as.integer(lat > 0))
}

make_dgp <- function(n, theta_l) {
  force(n); force(theta_l)
  function() dgp3_probit(n, theta_l)
}

# BART arms: the fitter carries its own draw_lp, which sudo_binary now uses
# in preference to manufacturing bootstrap draws. acf1 is stashed so the
# retained-chain autocorrelation rides out with the MC row.
make_est_bart <- function(pl, B, burn, thin) {
  force(pl); force(B); force(burn); force(thin)
  function(d) {
    box <- new.env()
    fm_fn <- function(dd, folds) {
      f <- fit_bart_index(dd, folds, partially_linear = pl, B = B,
                          burn = burn, thin = thin)
      box$acf1 <- f$acf1
      f
    }
    r <- sudo_binary(d, B = B, full_model = fm_fn, link = "probit")
    r$acf1 <- box$acf1
    r
  }
}

make_est_backfit <- function(boot) {
  force(boot)
  function(d) sudo_binary(d, B = 25, full_model = "pl_backfit",
                          proper_boot = boot, nn_size = 4, nn_decay = 0.3,
                          pl_n_iter = 5, link = "probit")
}

make_est_pipeboot <- function(B_outer, inner_B) {
  force(B_outer); force(inner_B)
  function(d) sudo_pipeline_boot(d, B_outer = B_outer, inner_B = inner_B,
                                 full_model = "pl_backfit", nn_size = 4,
                                 nn_decay = 0.3, pl_n_iter = 5,
                                 link = "probit")
}

n <- 2000
B_draws <- 25
cl <- mc_cluster(c("dgp3_probit", "SD_LOGIS"))
invisible(parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages(library(nnet))
  suppressPackageStartupMessages(library(dbarts))
  source("R/sudo/estimator.R"); source("R/sudo/pl.R"); source("R/sudo/bart.R")
}))
on.exit(parallel::stopCluster(cl), add = TRUE)

# cheapest first, so the headline comparison lands before the 185s/rep
# pipeline-bootstrap reference finishes
arms <- list(
  list(tag = "C pl-bart",      reps = 200, est = make_est_bart(TRUE,  B_draws, 200, 20)),
  list(tag = "D vanilla-bart", reps = 200, est = make_est_bart(FALSE, B_draws, 200, 20)),
  list(tag = "A backfit-boot", reps = 200, est = make_est_backfit(TRUE)),
  list(tag = "B pipeline-boot", reps = 100, est = make_est_pipeboot(100, 15))
)

cat(sprintf("stage 3q: Bayesian imputation model, probit DGP, n=%d, B=%d\n\n",
            n, B_draws))

out <- list()
for (a in arms) {
  for (theta_l in c(1.5, 3)) {
    theta_true <- theta_l / SD_LOGIS
    t0 <- Sys.time()
    df <- run_mc_par(cl, a$reps, make_dgp(n, theta_l), a$est, theta_true,
                     seed = if (theta_l == 1.5) 8000 else 2000)
    s <- summarize_mc(df)
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    print_mc(sprintf("%s t=%.1f", a$tag, theta_l), s)
    out[[length(out) + 1]] <- cbind(
      stage = "3q", arm = a$tag, theta_logit = theta_l,
      theta_true = theta_true, n = n, s,
      sd_over_mean_se = s$sd / s$mean_se,
      acf1 = if ("acf1" %in% names(df)) mean(df$acf1, na.rm = TRUE) else NA_real_,
      minutes = el)
    # write incrementally so partial results survive an interrupted run
    dir.create("R/results", showWarnings = FALSE)
    write.csv(do.call(rbind, out), "R/results/stage3q_summary.csv",
              row.names = FALSE)
  }
}

res <- do.call(rbind, out)
cat("\n")
print(format(res[, c("arm", "theta_logit", "bias", "mc_se_bias", "sd",
                     "mean_se", "sd_over_mean_se", "coverage", "mc_se_cov",
                     "acf1", "minutes")], digits = 3), row.names = FALSE)

# mechanical correctness, not the scientific verdict
stopifnot(all(is.finite(res$bias)), all(is.finite(res$mean_se)),
          all(res$mean_se > 0))
acf_pl <- res$acf1[res$arm == "C pl-bart"]
stopifnot(all(is.na(acf_pl) | abs(acf_pl) < 0.2))
cat("\nPASS: all arms produced finite estimates;",
    "PL-BART retained-chain lag-1 acf below 0.2\n")

cat("\nVerdict (read, do not assert):\n")
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  cat(sprintf("  %-16s t=%.1f  bias %+.3f  cover %.3f  sd/mean_se %.2f\n",
              r$arm, r$theta_logit, r$bias, r$coverage, r$sd_over_mean_se))
}
cat("\nwrote R/results/stage3q_summary.csv\n")
