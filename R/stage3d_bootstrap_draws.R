# Stage 3d: proper draws for a black-box full model, via the bootstrap.
# The stage-3 ranger arm showed an RF full model gives an unbiased point
# estimate but coverage 0.665: with no parameter covariance to redraw from,
# its draws are improper. Here each draw refits the forest on a within-fold
# resample of the training data (nonparametric bootstrap) so the draws
# carry the model-fit uncertainty. Same DGP/config/seed as the stage-3
# ranger arm for direct comparison; a moderate-theta config checks the fix
# is not specific to the extreme-signal case.
# Run from repo root: Rscript R/stage3d_bootstrap_draws.R

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
est_boot <- function(d) sudo_binary(d, full_model = "ranger",
                                    proper_boot = TRUE)

est_improper <- function(d) sudo_binary(d, full_model = "ranger")

cat("RF full model, recentered-bootstrap vs improper draws, B = 25, n = 2000\n")
cat("(stage-3 improper-RF baseline at theta=3: bias +0.001, coverage 0.665)\n\n")
all_summ <- list()
arms <- list(
  list(name = "boot_rf_t3_n2000", est = est_boot, theta = 3,
       n_reps = 200, seed = 2000),
  list(name = "boot_rf_t1.5_n2000", est = est_boot, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "improper_rf_t1.5_n2000", est = est_improper, theta = 1.5,
       n_reps = 100, seed = 8000))
for (a in arms) {
  df <- run_mc_par(cl, a$n_reps, make_dgp(2000, a$theta), a$est, a$theta,
                   seed = a$seed)
  s <- summarize_mc(df)
  print_mc(a$name, s)
  all_summ[[length(all_summ) + 1]] <- cbind(stage = "3d", estimator = a$name, s)
}
parallel::stopCluster(cl)
sm <- do.call(rbind, all_summ)

g <- function(nm) sm[sm$estimator == nm, ]
s3 <- g("boot_rf_t3_n2000")
s15b <- g("boot_rf_t1.5_n2000")
s15i <- g("improper_rf_t1.5_n2000")
# Findings encoded, not hopes:
# (1) recentering removes the raw-bootstrap location bias at theta=3
#     (raw bootstrap gave +0.125; improper baseline +0.001)
stopifnot(abs(s3$bias) < max(2 * s3$mc_se_bias, 0.02 * 3) + 1e-9)
# (2) bootstrap draws improve coverage over improper (0.665) but do NOT
#     reach nominal — black-box proper draws remain open
stopifnot(s3$coverage > 0.70)
# (3) the theta=1.5 bias is the RF index itself, not the draw scheme:
#     improper and bootstrap arms should agree within MC error
stopifnot(abs(s15b$bias - s15i$bias) <
          3 * sqrt(s15b$mc_se_bias^2 + s15i$mc_se_bias^2))
cat("\nPASS (documented findings): recentered draws unbiased at theta=3;\n",
    " coverage improves but stays short of nominal; theta=1.5 bias traces\n",
    " to the RF index (draw-scheme independent). Black-box proper draws\n",
    " remain the top open item.\n")

write.csv(sm, "R/results/stage3d_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3d_summary.csv\n")
