# Stage 3n: oracle-index control — is the coverage gap the full model, or
# the inference machinery itself?
#
# Stage 3m left a sharp question. The tuned PL-backfit has essentially the
# same (tiny) bias at both signal strengths, yet coverage 0.950 at
# theta=1.5 and 0.860 at theta=3, with sd/mean_se = 1.31 at theta=3. So
# the residual gap is variance, not bias. Two candidate sources:
#   (a) full-model error (V_hat != eta) that the bootstrap draws fail to
#       represent, or
#   (b) something in the inference machinery itself — the sandwich W
#       treating the partialling nuisances as fixed, fold randomness, the
#       Rubin/bootstrap variance construction.
# Substituting the TRUE index for V_hat separates them cleanly.
#
# Arms:
#   O0  oracle index + oracle nuisances   -> the machinery alone
#   O1  oracle index + gam nuisances      -> + nuisance estimation
#   O2  gam full model + gam nuisances    -> current baseline
#   O3  tuned nnet (size 8, decay 0.1)    -> best non-PL
#   O4  tuned PL backfit (3m winner)      -> best PL, the 0.950/0.860 arm
#
# DRAW SCHEME, and why `proper = FALSE` is correct for O0/O1 rather than
# "improper": with a KNOWN index there is no imputation-model parameter
# uncertainty to propagate, so redrawing only S *is* a draw from the true
# posterior predictive of U | Y, D, X. Rubin's rules are congenial here.
# The repo's usual "improper draws under-cover" framing does not apply.
#
# Oracle nuisances are closed form for this DGP:
#   m_0(X) = plogis(X4 + cos(X5))
#   l_0(X) = E[S|X] = E[U|X] = g(X) + theta*m_0(X)
# NOTE l_0 is the true E[S|X] ONLY when the index is the oracle; pairing it
# with a fitted full model would be a misspecification, so it is used only
# in O0.
#
# Run from repo root: Rscript R/stage3n_oracle_control.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")
source("R/sudo/pl.R")

dgp_eta <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  eta <- theta * D + g
  U <- eta + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0), eta = eta, U = U)
}

# injectable full models -------------------------------------------------
oracle_index <- function(d, folds)
  list(lp_hat = d$eta, fit_fold = NULL, to_lp = identity)

# oracle partialling nuisances (fitter contract: f(X,y) -> f(Xnew))
oracle_m <- function(X, y) function(Xnew) plogis(Xnew$X4 + cos(Xnew$X5))
make_oracle_l <- function(theta) {
  force(theta)
  function(X, y) function(Xnew)
    Xnew$X1^2 + sin(Xnew$X2) + 0.5 * Xnew$X3 +
      theta * plogis(Xnew$X4 + cos(Xnew$X5))
}

cl <- mc_cluster(c("dgp_eta", "oracle_index", "oracle_m", "make_oracle_l"))
invisible(parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages({library(nnet); library(ranger)})
  source("R/sudo/estimator.R"); source("R/sudo/pl.R")
}))

make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp_eta(n, theta)
}
make_arm <- function(kind, theta) {
  force(kind); force(theta)
  ol <- make_oracle_l(theta)
  switch(kind,
    O0 = function(d) sudo_binary(d, full_model = oracle_index,
                                 fit_l = ol, fit_m = oracle_m),
    O1 = function(d) sudo_binary(d, full_model = oracle_index),
    O2 = function(d) sudo_binary(d, full_model = "gam", proper_boot = FALSE),
    O3 = function(d) sudo_binary(d, full_model = "nnet", nn_decay = 0.1,
                                 proper_boot = TRUE),
    O4 = function(d) sudo_binary(d, full_model = "pl_backfit",
                                 pl_engine = "nnet", nn_size = 4,
                                 nn_decay = 0.3, pl_n_iter = 5,
                                 proper_boot = TRUE))
}

arms <- list(
  list(id = "O0", lab = "oracle index + oracle nuisances", reps = 500),
  list(id = "O1", lab = "oracle index + gam nuisances",    reps = 500),
  list(id = "O2", lab = "gam full model (proper draws)",   reps = 200),
  list(id = "O3", lab = "tuned nnet + bootstrap draws",    reps = 200),
  list(id = "O4", lab = "tuned PL backfit + bootstrap",    reps = 200))

cat("oracle-index control, n = 2000, B = 25\n")
cat("known-SE coverage = coverage of theta_hat +/- 1.96*mean(se); the gap\n")
cat("to actual coverage isolates SE *variability* from SE *level*.\n\n")

rows <- list()
for (th in c(1.5, 3)) {
  cat(sprintf("--- theta = %.1f ---\n", th))
  cat(sprintf("%-34s %8s %8s %8s %9s %9s %9s\n",
              "arm", "bias", "sd", "mean_se", "sd/mnse", "cover", "knownSE"))
  for (a in arms) {
    df <- run_mc_par(cl, a$reps, make_dgp(2000, th), make_arm(a$id, th), th,
                     seed = 6000 + round(th * 1000))
    s <- summarize_mc(df)
    known <- mean(abs(df$est - th) <= 1.96 * mean(df$se))
    ratio <- s$sd / s$mean_se
    cat(sprintf("%-34s %+8.4f %8.4f %8.4f %9.3f %9.3f %9.3f\n",
                paste0(a$id, " ", a$lab), s$bias, s$sd, s$mean_se, ratio,
                s$coverage, known))
    rows[[length(rows) + 1]] <- data.frame(
      stage = "3n", theta = th, arm = a$id, label = a$lab, n_reps = a$reps,
      bias = s$bias, mc_se_bias = s$mc_se_bias, sd = s$sd,
      mean_se = s$mean_se, sd_over_mean_se = ratio,
      coverage = s$coverage, mc_se_cov = s$mc_se_cov,
      known_se_coverage = known,
      mean_W = mean(df$W), mean_B_between = mean(df$B_between),
      mean_df = mean(df$df), mean_max_abs_lp = mean(df$max_abs_lp))
  }
  cat("\n")
}
parallel::stopCluster(cl)
sm <- do.call(rbind, rows)

g <- function(id, th) sm[sm$arm == id & sm$theta == th, ]
cat("== decomposition ==\n")
for (th in c(1.5, 3)) {
  o0 <- g("O0", th); o1 <- g("O1", th); o4 <- g("O4", th)
  cat(sprintf("theta=%.1f  machinery alone (O0) cover %.3f, sd/mean_se %.3f\n",
              th, o0$coverage, o0$sd_over_mean_se))
  cat(sprintf("          + nuisance estimation (O1): %+.3f cover, sd/mean_se %.3f\n",
              o1$coverage - o0$coverage, o1$sd_over_mean_se))
  cat(sprintf("          + full-model error (O4):    %+.3f cover, sd/mean_se %.3f\n",
              o4$coverage - o1$coverage, o4$sd_over_mean_se))
}

# Assertions encode what the design can support, not what we hope to see.
# O0 has a known imputation model, so it is the one arm where nominal
# coverage is a genuine prediction rather than an aspiration.
o0_15 <- g("O0", 1.5); o0_3 <- g("O0", 3)
stopifnot(abs(o0_15$bias) < 3 * o0_15$mc_se_bias)
stopifnot(abs(o0_3$bias) < 3 * o0_3$mc_se_bias)
cat("\nPASS: oracle-index arm is unbiased at both signal strengths.\n")
cat("Coverage of O0 is REPORTED, not asserted -- it is the quantity under test.\n")

write.csv(sm, "R/results/stage3n_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3n_summary.csv\n")
