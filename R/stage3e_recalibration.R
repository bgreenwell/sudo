# Stage 3e: isotonic recalibration of the black-box full model.
# Stage 3d found the RF index attenuates theta at moderate signal (-0.16 at
# theta=1.5 under any draw scheme): RF probabilities are compressed toward
# the base rate, so their logit understates the latent index. Here every
# probability vector (base fit and each bootstrap refit) is isotonic-
# recalibrated against the observed y before the link inverse.
# Arms (n = 2000, seeds matched to stage 3d):
#   improper + recal, theta=1.5   does recalibration remove the -0.169 bias?
#   boot + recal,     theta=1.5   bias and coverage together
#   boot + recal,     theta=3     regression check on the strong-signal case
# Run from repo root: Rscript R/stage3e_recalibration.R

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

cl <- mc_cluster(c("dgp3", "sudo_binary", "crossfit_fullmodel_gam",
                   "recal_iso"))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp3(n, theta)
}
est_imp_recal <- function(d) sudo_binary(d, full_model = "ranger",
                                         recalibrate = TRUE)
est_boot_recal <- function(d) sudo_binary(d, full_model = "ranger",
                                          proper_boot = TRUE,
                                          recalibrate = TRUE)

cat("RF full model with isotonic recalibration, B = 25, n = 2000\n")
cat("(stage-3d baselines: improper t1.5 bias -0.169 cover 0.700;",
    "boot t1.5 bias -0.160 cover 0.730; boot t3 bias +0.036 cover 0.765)\n\n")
all_summ <- list()
arms <- list(
  list(name = "improper_recal_t1.5", est = est_imp_recal, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "boot_recal_t1.5", est = est_boot_recal, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "boot_recal_t3", est = est_boot_recal, theta = 3,
       n_reps = 200, seed = 2000))
for (a in arms) {
  df <- run_mc_par(cl, a$n_reps, make_dgp(2000, a$theta), a$est, a$theta,
                   seed = a$seed)
  s <- summarize_mc(df)
  print_mc(a$name, s)
  all_summ[[length(all_summ) + 1]] <-
    cbind(stage = "3e", estimator = a$name, s)
}
parallel::stopCluster(cl)
sm <- do.call(rbind, all_summ)

g <- function(nm) sm[sm$estimator == nm, ]
# Findings encoded, not hopes:
# (1) recalibration reduces the theta=1.5 attenuation (stage-3d baselines
#     -0.169 improper / -0.160 boot) but does not remove it
stopifnot(abs(g("improper_recal_t1.5")$bias) < 0.169)
stopifnot(abs(g("boot_recal_t1.5")$bias) < 0.160)
# (2) recalibration HARMS the strong-signal case: isotonic regression pools
#     the near-separated extreme probabilities into ties, flattening the
#     tail index variation theta=3 depends on (stage-3d boot t3: +0.036)
stopifnot(g("boot_recal_t3")$bias < -0.2)
cat("\nPASS (documented findings): isotonic recalibration mildly reduces\n",
    " the moderate-signal attenuation but badly biases the strong-signal\n",
    " case -- not a viable fix for the black-box full model.\n")

write.csv(sm, "R/results/stage3e_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3e_summary.csv\n")
