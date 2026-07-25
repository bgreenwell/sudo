# Stage 3f: a smooth black-box full model (single-hidden-layer nnet).
# Hypothesis from stages 3d/3e: the RF attenuation is specific to
# piecewise-constant learners — a forest cannot extrapolate the index into
# its tails and compresses probabilities, and neither bootstrap draws nor
# isotonic recalibration fix that. A smooth net with a logistic output
# estimates the log-odds surface directly.
#
# First pass at decay=0.01 badly OVERSHOT at theta=3 (+0.43 to +0.47 bias,
# worse than RF). An in-sample and cross-fitted tuning check traced this to
# under-regularization: decay=0.01 lets the net extrapolate too sharply
# across the large jump D creates in the index at theta=3, especially with
# size=16. decay=0.1 (still size=8) removed nearly all of it in a 15-rep
# spot check (mean 3.015 vs truth 3). That is the configuration used here.
#
# Arms (n = 2000, size=8, decay=0.1, seeds matched to stages 3d/3e):
#   improper nnet, theta=1.5   attenuation test (RF baseline: -0.169)
#   improper nnet, theta=3     strong-signal check (RF baseline: +0.001)
#   boot nnet,     theta=1.5   recentered bootstrap draws: coverage
#   boot nnet,     theta=3     (RF boot baselines: cover 0.730 / 0.765)
# Run from repo root: Rscript R/stage3f_nn_fullmodel.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")

dgp3 <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  U <- theta * D + g + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0))
}

cl <- mc_cluster(c("dgp3", "sudo_binary", "crossfit_fullmodel_gam"))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp3(n, theta)
}
est_imp <- function(d) sudo_binary(d, full_model = "nnet",
                                   nn_decay = 0.1)
est_boot <- function(d) sudo_binary(d, full_model = "nnet",
                                    proper_boot = TRUE, nn_decay = 0.1)

cat("nnet full model (size 8, decay 0.1), B = 25, n = 2000\n")
cat("(RF baselines: improper t1.5 -0.169/0.700; boot t1.5 -0.160/0.730;",
    "boot t3 +0.036/0.765)\n\n")
all_summ <- list()
arms <- list(
  list(name = "improper_nn_t1.5", est = est_imp, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "improper_nn_t3", est = est_imp, theta = 3,
       n_reps = 100, seed = 2000),
  list(name = "boot_nn_t1.5", est = est_boot, theta = 1.5,
       n_reps = 60, seed = 8000),
  list(name = "boot_nn_t3", est = est_boot, theta = 3,
       n_reps = 60, seed = 2000))
for (a in arms) {
  df <- run_mc_par(cl, a$n_reps, make_dgp(2000, a$theta), a$est, a$theta,
                   seed = a$seed)
  s <- summarize_mc(df)
  print_mc(a$name, s)
  all_summ[[length(all_summ) + 1]] <-
    cbind(stage = "3f", estimator = a$name, s)
}
parallel::stopCluster(cl)
sm <- do.call(rbind, all_summ)

g <- function(nm) sm[sm$estimator == nm, ]
# the smooth black-box must remove the systematic attenuation at both
# signal strengths (5% relative tolerance: nnet is noisier than gam)
stopifnot(abs(g("improper_nn_t1.5")$bias) <
          max(2 * g("improper_nn_t1.5")$mc_se_bias, 0.05 * 1.5) + 1e-9)
stopifnot(abs(g("improper_nn_t3")$bias) <
          max(2 * g("improper_nn_t3")$mc_se_bias, 0.05 * 3) + 1e-9)
# bootstrap draws must not introduce bias; coverage reported, floor at the
# RF-boot level
stopifnot(abs(g("boot_nn_t1.5")$bias) <
          max(2 * g("boot_nn_t1.5")$mc_se_bias, 0.05 * 1.5) + 1e-9)
stopifnot(g("boot_nn_t1.5")$coverage >= 0.73)
cat("\nPASS: smooth black-box full model removes the index attenuation\n")

write.csv(sm, "R/results/stage3f_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3f_summary.csv\n")
