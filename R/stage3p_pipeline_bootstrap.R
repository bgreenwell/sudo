# Stage 3p: does a full-pipeline bootstrap close the theta=3 coverage gap?
#
# The oracle control (stage 3n) localised the residual under-coverage
# precisely: with the tuned PL-backfit imputation model, bias is solved at
# both signal strengths but at theta=3 the reported SE is ~31% too small
# (sd/mean_se = 1.31, coverage 0.860). The cause is the recentered
# within-fold bootstrap under-propagating the imputation model's OWN
# sampling variance -- it holds the folds and partialling nuisances fixed
# and perturbs only the index.
#
# This replaces that draw scheme with a full-pipeline nonparametric
# bootstrap (sudo_pipeline_boot): resample the whole dataset, rerun the
# COMPLETE pipeline per replicate, take the SE as the SD of theta across
# replicates. By construction it captures every source of sampling
# variability. Same tuned PL config and seeds as stage 3m for a direct
# comparison.
#
# Baseline to beat (stage 3m, recentered bootstrap, tuned PL backfit):
#   theta=1.5  bias -0.027  coverage 0.950  sd/mean_se 1.12
#   theta=3    bias -0.028  coverage 0.860  sd/mean_se 1.31
# Success = coverage ~0.95 at theta=3 with sd/mean_se ~1.
#
# Run from repo root: Rscript R/stage3p_pipeline_bootstrap.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")
source("R/sudo/pl.R")

dgp3 <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  U <- theta * D + g + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0))
}

cl <- mc_cluster(c("dgp3", "sudo_binary", "sudo_pipeline_boot",
                  "crossfit_fullmodel_gam"))
invisible(parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages(library(nnet))
  source("R/sudo/pl.R")
}))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp3(n, theta)
}
# tuned PL-backfit config (stage 3m winner), full-pipeline bootstrap SE
make_est <- function(B_outer, inner_B) {
  force(B_outer); force(inner_B)
  function(d) sudo_pipeline_boot(d, B_outer = B_outer, inner_B = inner_B,
                                 full_model = "pl_backfit", pl_engine = "nnet",
                                 nn_size = 4, nn_decay = 0.3, pl_n_iter = 5)
}

B_outer <- 100
inner_B <- 15
n_reps <- 100
cat(sprintf("full-pipeline bootstrap, tuned PL-backfit, B_outer=%d inner_B=%d, n=2000, %d reps\n\n",
            B_outer, inner_B, n_reps))

rows <- list()
for (th in c(1.5, 3)) {
  df <- run_mc_par(cl, n_reps, make_dgp(2000, th), make_est(B_outer, inner_B),
                   th, seed = ifelse(th == 1.5, 8000, 2000))
  s <- summarize_mc(df)
  # percentile-CI coverage from the carried-through columns
  pct_cov <- mean(df$ci_lo_pct <= th & th <= df$ci_hi_pct)
  ratio <- s$sd / s$mean_se
  cat(sprintf("theta=%.1f  bias %+.4f  sd %.4f  mean_se %.4f  sd/se %.3f  cover(normal) %.3f  cover(pct) %.3f\n",
              th, s$bias, s$sd, s$mean_se, ratio, s$coverage, pct_cov))
  rows[[length(rows) + 1]] <- data.frame(
    stage = "3p", theta = th, estimator = "pipeline_boot_pltuned",
    n_reps = n_reps, bias = s$bias, mc_se_bias = s$mc_se_bias, sd = s$sd,
    mean_se = s$mean_se, sd_over_mean_se = ratio,
    coverage = s$coverage, coverage_pct = pct_cov, mc_se_cov = s$mc_se_cov)
}
parallel::stopCluster(cl)
sm <- do.call(rbind, rows)

cat("\n== vs stage-3m recentered bootstrap ==\n")
cat("theta=1.5  3m: cover 0.950, sd/se 1.12  ->  3p above\n")
cat("theta=3    3m: cover 0.860, sd/se 1.31  ->  3p above\n")

g <- function(th) sm[sm$theta == th, ]
# bias must be unchanged (same point-estimate procedure); coverage at
# theta=3 must improve materially over the recentered bootstrap's 0.860
stopifnot(abs(g(3)$bias) < max(2 * g(3)$mc_se_bias, 0.03 * 3) + 1e-9)
stopifnot(g(3)$coverage > 0.860 + g(3)$mc_se_cov)
cat("\nPASS: full-pipeline bootstrap improves theta=3 coverage over the",
    "recentered scheme (bias unchanged).\n")

write.csv(sm, "R/results/stage3p_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3p_summary.csv\n")
