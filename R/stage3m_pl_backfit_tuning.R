# Stage 3m: CV'd hyperparameter search for backfitting-PL, selecting on
# theta-level MC performance directly.
#
# Every proxy tried in this program has failed to predict theta bias:
# deviance/index-R2 (stage 3g), D-residual mean structure (3h),
# treatment-conditional variance (3i, and again on the PL models in
# 3j-3l). mboost's own "principled" cvrisk() tuning (stage 3l) actively
# HURT theta by optimizing predictive deviance instead. So this search
# does not touch any of those proxies: the selection criterion is the
# actual Monte Carlo bias of theta, at both signal strengths, full stop.
#
# Three phases:
#   (1) screen (nnet size, decay) on a grid, cheap improper-draw MC
#       (n_reps=40, B=10, no bootstrap refits), scored by the WORST-CASE
#       |bias| across theta in {1.5, 3} -- a config that is fine on
#       average but fails badly at one signal strength is not what we
#       want, exactly the failure mode nnet_d0.01 without PL showed.
#   (2) refine n_iter (backfit iterations) around the phase-1 winner,
#       same cheap screening setup.
#   (3) validate the final winner at full fidelity (n_reps=100, B=25,
#       improper + bootstrap, both theta) -- the exact stage-3j design,
#       for direct comparison against the fixed (size=8, decay=0.01,
#       n_iter=5) baseline already on record.
#
# Run from repo root: Rscript R/stage3m_pl_backfit_tuning.R

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

cl <- mc_cluster(c("dgp3", "sudo_binary", "crossfit_fullmodel_gam",
                  "fit_pl_backfit"))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages(
  library(nnet))))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp3(n, theta)
}
make_est_screen <- function(size, decay, n_iter) {
  force(size); force(decay); force(n_iter)
  function(d) sudo_binary(d, full_model = "pl_backfit", pl_engine = "nnet",
                          nn_size = size, nn_decay = decay,
                          pl_n_iter = n_iter, B = 10)
}

screen_config <- function(size, decay, n_iter, n_reps = 40, seed0 = 50000) {
  biases <- sapply(c(1.5, 3), function(th) {
    df <- run_mc_par(cl, n_reps, make_dgp(2000, th),
                     make_est_screen(size, decay, n_iter), th,
                     seed = seed0 + round(th * 1000))
    summarize_mc(df)$bias
  })
  names(biases) <- c("bias_t1.5", "bias_t3")
  c(biases, worst_abs = max(abs(biases)))
}

cat("== phase 1: screen (nnet size, decay), n_iter=5 fixed, n_reps=40 per theta ==\n")
cat("(baseline on record: size=8, decay=0.01, n_iter=5 -- see stage 3j)\n\n")
grid1 <- expand.grid(size = c(4, 8, 16), decay = c(0.003, 0.01, 0.03, 0.1, 0.3))
res1 <- t(apply(grid1, 1, function(r) screen_config(r["size"], r["decay"], 5)))
grid1 <- cbind(grid1, res1)
grid1 <- grid1[order(grid1$worst_abs), ]
print(grid1, row.names = FALSE)
best1 <- grid1[1, ]
cat(sprintf("\nphase-1 winner: size=%g decay=%g  (worst|bias|=%.4f)\n",
            best1$size, best1$decay, best1$worst_abs))

cat("\n== phase 2: refine n_iter around the phase-1 winner ==\n")
grid2 <- data.frame(n_iter = c(3, 5, 8, 12))
res2 <- t(sapply(grid2$n_iter, function(ni)
  screen_config(best1$size, best1$decay, ni)))
grid2 <- cbind(grid2, res2)
grid2 <- grid2[order(grid2$worst_abs), ]
print(grid2, row.names = FALSE)
best_n_iter <- grid2$n_iter[1]
cat(sprintf("\nphase-2 winner: n_iter=%g  (worst|bias|=%.4f)\n",
            best_n_iter, grid2$worst_abs[1]))

final_size <- best1$size; final_decay <- best1$decay
cat(sprintf(
  "\n== final config: size=%g decay=%g n_iter=%g ==\n\n",
  final_size, final_decay, best_n_iter))

cat("== phase 3: full-fidelity validation, n_reps=100, B=25 ==\n")
cat("(stage-3j baseline, size=8/decay=0.01/n_iter=5: improper t1.5",
    "-0.039/0.880, t3 -0.116/0.680; boot t1.5 -0.023/0.910, t3",
    "-0.106/0.830)\n\n")
# force()d factory, not a plain closure: PSOCK workers don't share
# globalenv, so free variables (final_size etc.) must be baked into the
# function's own environment via force() to survive serialization (the
# same bug class hit in stages 3/3d earlier in this program)
make_est_final <- function(size, decay, n_iter, boot) {
  force(size); force(decay); force(n_iter); force(boot)
  function(d) sudo_binary(d, full_model = "pl_backfit", pl_engine = "nnet",
                          nn_size = size, nn_decay = decay,
                          pl_n_iter = n_iter, proper_boot = boot)
}
est_imp <- make_est_final(final_size, final_decay, best_n_iter, FALSE)
est_boot <- make_est_final(final_size, final_decay, best_n_iter, TRUE)
arms <- list(
  list(name = "improper_pltuned_t1.5", est = est_imp, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "improper_pltuned_t3", est = est_imp, theta = 3,
       n_reps = 100, seed = 2000),
  list(name = "boot_pltuned_t1.5", est = est_boot, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "boot_pltuned_t3", est = est_boot, theta = 3,
       n_reps = 100, seed = 2000))
all_summ <- list()
for (a in arms) {
  df <- run_mc_par(cl, a$n_reps, make_dgp(2000, a$theta), a$est, a$theta,
                   seed = a$seed)
  s <- summarize_mc(df)
  print_mc(a$name, s)
  all_summ[[length(all_summ) + 1]] <- cbind(stage = "3m", estimator = a$name, s)
}
parallel::stopCluster(cl)
sm <- do.call(rbind, all_summ)

cat("\n== summary vs stage-3j fixed baseline ==\n")
print(sm[, c("estimator", "bias", "coverage")])

write.csv(grid1, "R/results/stage3m_grid1_size_decay.csv", row.names = FALSE)
write.csv(grid2, "R/results/stage3m_grid2_niter.csv", row.names = FALSE)
write.csv(sm, "R/results/stage3m_summary.csv", row.names = FALSE)
write.csv(data.frame(size = final_size, decay = final_decay,
                     n_iter = best_n_iter),
          "R/results/stage3m_final_config.csv", row.names = FALSE)
cat("wrote R/results/stage3m_{grid1_size_decay,grid2_niter,summary,final_config}.csv\n")
