# Stage 3l: partially-linear full model via mboost componentwise boosting.
# gamboost(Y ~ bols(D) + bbs(X1) + ... + bbs(X5)): D gets a plain linear
# (ordinary-least-squares) base-learner, each X_j its own P-spline
# base-learner. No interaction base-learner exists in the formula, so D
# structurally cannot pick up interaction effects regardless of mstop.
# mstop chosen via cvrisk() (k-fold cross-validated risk) rather than
# picked manually — the one PL route where tuning is handled by the
# package's own established procedure instead of by us.
#
# Three sub-analyses, same pattern as 3d-3k:
#   (a) bias/coverage MC (improper + bootstrap, theta in {1.5,3})
#   (b) index-calibration diagnostic (eta ~ V_hat)
#   (c) variance-ratio diagnostic (var.test on D arms)
# Run from repo root: Rscript R/stage3l_pl_mboost.R

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
dgp_eta <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  eta <- theta * D + g
  U <- eta + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0), eta = eta)
}

cl <- mc_cluster(c("dgp3", "dgp_eta", "sudo_binary", "crossfit_fullmodel_gam",
                  "fit_pl", "fit_pl_mboost"))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages(
  library(mboost))))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp3(n, theta)
}

cat("== (a) bias/coverage MC: pl_mboost (gamboost, cvrisk mstop), B=25, n=2000 ==\n\n")
est_imp <- function(d) sudo_binary(d, full_model = "pl_mboost",
                                   pl_use_cvrisk = TRUE)
est_boot <- function(d) sudo_binary(d, full_model = "pl_mboost",
                                    pl_use_cvrisk = TRUE, proper_boot = TRUE)
arms <- list(
  list(name = "improper_plmb_t1.5", est = est_imp, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "improper_plmb_t3", est = est_imp, theta = 3,
       n_reps = 100, seed = 2000),
  list(name = "boot_plmb_t1.5", est = est_boot, theta = 1.5,
       n_reps = 100, seed = 8000),
  list(name = "boot_plmb_t3", est = est_boot, theta = 3,
       n_reps = 100, seed = 2000))
all_summ <- list()
for (a in arms) {
  df <- run_mc_par(cl, a$n_reps, make_dgp(2000, a$theta), a$est, a$theta,
                   seed = a$seed)
  s <- summarize_mc(df)
  print_mc(a$name, s)
  all_summ[[length(all_summ) + 1]] <- cbind(stage = "3l", estimator = a$name, s)
}
cat("\n")

cat("== (b) index calibration: eta ~ V_hat, n=2000, 100 reps ==\n")
calib_one <- function(d, theta_true) {
  X <- as.data.frame(d$X); folds <- make_folds(nrow(X), 5)
  lp_hat <- fit_pl(d, folds, "mboost", pl_use_cvrisk = TRUE)$lp_hat
  fit <- lm(d$eta ~ lp_hat)
  c(slope = unname(coef(fit)[2]), intercept = unname(coef(fit)[1]),
    r2 = summary(fit)$r.squared)
}
parallel::clusterExport(cl, "calib_one")
calib_rows <- list()
for (th in c(1.5, 3)) {
  parallel::clusterExport(cl, "th", envir = environment())
  res <- parallel::parSapply(cl, seq_len(100), function(r) {
    set.seed(2e5 + r)
    d <- dgp_eta(2000, th)
    calib_one(d, th)
  })
  m <- rowMeans(res)
  cat(sprintf("theta=%.1f pl_mboost  slope %.3f  intercept %+.3f  R2 %.3f\n",
              th, m["slope"], m["intercept"], m["r2"]))
  calib_rows[[length(calib_rows) + 1]] <- data.frame(
    stage = "3l", theta = th, learner = "pl_mboost",
    slope = m["slope"], intercept = m["intercept"], r2 = m["r2"],
    n_reps = 100)
}
calib_sm <- do.call(rbind, calib_rows)
cat("\n")

cat("== (c) variance ratio: var.test(R|D=1, R|D=0), n=2000, 100 reps ==\n")
diag_one <- function(d, theta_true) {
  X <- as.data.frame(d$X); n <- nrow(X); folds <- make_folds(n, 5)
  lp_hat <- fit_pl(d, folds, "mboost", pl_use_cvrisk = TRUE)$lp_hat
  S <- complete_surrogate(d$Y + 1L, lp_hat, c(-Inf, 0, Inf), "logit")
  R <- S - lp_hat
  vt <- var.test(R[d$D == 1], R[d$D == 0])
  D_res <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)
  S_hat <- crossfit(X, S, fit_gam, folds)
  theta_hat <- fwl_theta(S - S_hat, D_res)$theta
  c(var_ratio = unname(vt$estimate), p_het = vt$p.value,
    theta_bias = theta_hat - theta_true)
}
parallel::clusterExport(cl, "diag_one")
vr_rows <- list()
for (th in c(1.5, 3)) {
  parallel::clusterExport(cl, "th", envir = environment())
  res <- t(parallel::parSapply(cl, seq_len(100), function(r) {
    set.seed(2e5 + r)
    d <- dgp_eta(2000, th)
    diag_one(d, th)
  }))
  tt <- t.test(res[, "var_ratio"], mu = 1)
  cat(sprintf(
    "theta=%.1f pl_mboost  mean var_ratio %.3f (t-test p=%.2e)  frac p_het<0.05: %.2f  mean|theta_bias| %.3f\n",
    th, mean(res[, "var_ratio"]), tt$p.value, mean(res[, "p_het"] < 0.05),
    mean(abs(res[, "theta_bias"]))))
  vr_rows[[length(vr_rows) + 1]] <- data.frame(
    stage = "3l", theta = th, learner = "pl_mboost",
    mean_var_ratio = mean(res[, "var_ratio"]), var_ratio_ttest_p = tt$p.value,
    frac_p_het_sig = mean(res[, "p_het"] < 0.05),
    mean_abs_theta_bias = mean(abs(res[, "theta_bias"])), n_reps = 100)
}
vr_sm <- do.call(rbind, vr_rows)
parallel::stopCluster(cl)

sm <- do.call(rbind, all_summ)
cat("\n== summary ==\n")
print(sm[, c("estimator", "bias", "coverage")])

write.csv(sm, "R/results/stage3l_summary.csv", row.names = FALSE)
write.csv(calib_sm, "R/results/stage3l_calib.csv", row.names = FALSE)
write.csv(vr_sm, "R/results/stage3l_varratio.csv", row.names = FALSE)
cat("wrote R/results/stage3l_{summary,calib,varratio}.csv\n")
