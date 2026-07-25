# Stage 3h: do full-model surrogate residuals diagnose theta bias?
#
# Stage 3g's marginal calibration check (slope/R2 of eta ~ V_hat) did NOT
# predict theta bias across learners. This stage tests a sharper,
# D-specific diagnostic: theta is the coefficient on D, so the relevant
# question is not "how good is the fit overall" but "does the full
# model's residual carry D-specific structure". For each learner, draw a
# surrogate residual R = S - V_hat and fit gam(R ~ D + s(X)); under a
# trustworthy full model R should show no D structure (coefficient ~ 0).
# Also check residual variance by treatment arm (heteroscedasticity by D).
#
# Two tests, same reps:
#   (a) cross-learner comparison (like 3g): does the ranking of
#       |residual-on-D coefficient| match the known theta-bias ranking
#       (ranger best, tuned-NN middle, untuned-NN worst)?
#   (b) within-learner correlation, dataset by dataset: does a large
#       |residual-on-D coefficient| on a GIVEN rep predict a large theta
#       error on that SAME rep, using a cheap single-draw theta (no B
#       loop/bootstrap — this is a per-dataset trust diagnostic, not the
#       estimator itself)? This is the practically relevant question: you
#       only get one dataset in practice and want to know if you can
#       trust its full model.
#
# n = 2000, theta in {1.5, 3}, 100 reps, learners: gam, ranger,
# nnet decay=0.01 (untuned), nnet decay=0.1 (tuned).
# Also saves an illustrative residual-vs-D panel plot.
# Run from repo root: Rscript R/stage3h_residual_diagnostics.R

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

# one rep: residual-on-D diagnostic + a cheap single-draw theta on the SAME
# surrogate draw, so the two are directly comparable dataset by dataset
diag_one <- function(d, learner, nn_decay, theta_true) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  folds <- make_folds(n, 5)
  lp_hat <- get_lp_hat(d, learner, folds, nn_decay)

  S <- complete_surrogate(d$Y + 1L, lp_hat, c(-Inf, 0, Inf), "logit")
  R <- S - lp_hat

  dat <- data.frame(R = R, D = d$D, X)
  rhs <- paste(sprintf("s(%s)", colnames(X)), collapse = " + ")
  rfit <- gam(as.formula(paste("R ~ D +", rhs)), data = dat)
  pt <- summary(rfit)$p.table
  resid_D_coef <- unname(pt["D", "Estimate"])
  resid_D_pval <- unname(pt["D", "Pr(>|t|)"])
  var_ratio <- var(R[d$D == 1]) / var(R[d$D == 0])

  D_res <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)
  S_hat <- crossfit(X, S, fit_gam, folds)
  theta_hat <- fwl_theta(S - S_hat, D_res)$theta

  c(resid_D_coef = resid_D_coef, resid_D_pval = resid_D_pval,
    var_ratio = var_ratio, theta_hat = theta_hat,
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
cat(sprintf("residual-on-D diagnostic vs theta bias, n = 2000, %d reps\n",
            n_reps))
cat("resid_D_coef ~ 0 = no D structure in the residual (trustworthy);",
    "var_ratio ~ 1 = homoscedastic across D\n\n")

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
    corr <- cor(abs(res[, "resid_D_coef"]), abs(res[, "theta_bias"]))
    cat(sprintf(
      "theta=%.1f %-12s |resid_D_coef| mean %.3f (sd %.3f)  var_ratio %.2f  mean theta_bias %+.3f  within-rep cor(|resid_D_coef|,|theta_bias|) %.3f\n",
      th, L$name, mean(abs(res[, "resid_D_coef"])),
      sd(res[, "resid_D_coef"]), mean(res[, "var_ratio"]),
      mean(res[, "theta_bias"]), corr))
    cross_learner[[length(cross_learner) + 1]] <- data.frame(
      theta = th, learner = L$name,
      mean_abs_resid_D = mean(abs(res[, "resid_D_coef"])),
      mean_var_ratio = mean(res[, "var_ratio"]),
      mean_theta_bias = mean(res[, "theta_bias"]),
      mean_abs_theta_bias = mean(abs(res[, "theta_bias"])),
      within_rep_cor = corr, n_reps = n_reps)
    rows[[length(rows) + 1]] <- data.frame(
      theta = th, learner = L$name, rep = seq_len(n_reps),
      resid_D_coef = res[, "resid_D_coef"],
      resid_D_pval = res[, "resid_D_pval"],
      var_ratio = res[, "var_ratio"], theta_hat = res[, "theta_hat"],
      theta_bias = res[, "theta_bias"])
  }
  cat("\n")
}
parallel::stopCluster(cl)
sm <- do.call(rbind, cross_learner)
detail <- do.call(rbind, rows)

cat("== cross-learner ranking check ==\n")
for (th in c(1.5, 3)) {
  s <- sm[sm$theta == th, ]
  ord_resid <- s$learner[order(s$mean_abs_resid_D)]
  ord_theta <- s$learner[order(s$mean_abs_theta_bias)]
  cat(sprintf("theta=%.1f  ranking by |resid_D_coef|: %s\n", th,
              paste(ord_resid, collapse = " < ")))
  cat(sprintf("theta=%.1f  ranking by |theta_bias|:   %s\n", th,
              paste(ord_theta, collapse = " < ")))
}

# --- illustrative residual-vs-D panel (theta=3, one representative seed) ---
dir.create("manuscript/figures", showWarnings = FALSE, recursive = TRUE)
png("manuscript/figures/stage3h_residual_by_D.png", width = 1400, height = 400,
    res = 120)
op <- par(mfrow = c(1, 4), mar = c(4, 4, 3, 1))
set.seed(2e5 + 1)
d3 <- dgp_eta(2000, 3)
folds3 <- make_folds(2000, 5)
for (L in learners) {
  lp <- get_lp_hat(d3, L$learner, folds3, L$decay)
  S <- complete_surrogate(d3$Y + 1L, lp, c(-Inf, 0, Inf), "logit")
  R <- S - lp
  boxplot(R ~ d3$D, xlab = "D", ylab = "surrogate residual",
          main = sprintf("%s (theta=3)", L$name), col = "grey85")
  abline(h = 0, lty = 2)
}
par(op); invisible(dev.off())
cat("\nwrote manuscript/figures/stage3h_residual_by_D.png\n")

write.csv(sm, "R/results/stage3h_summary.csv", row.names = FALSE)
write.csv(detail, "R/results/stage3h_detail.csv", row.names = FALSE)
cat("wrote R/results/stage3h_summary.csv and stage3h_detail.csv\n")
