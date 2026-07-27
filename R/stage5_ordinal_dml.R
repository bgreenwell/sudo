# Stage 5: ordinal SUDO with flexible nuisances (the full method).
# DGP: U = theta*D + g(X) + logis with the stage-3 nonlinear g and propensity,
# cut into J=3. Full model: cross-fitted ordinal::clm on D + natural-spline
# bases of X — linear in known columns, so its vcov (thresholds included)
# supports exact proper-MI draws. Partialling nuisances E[S|X], E[D|X]: gam.
# Run from repo root: Rscript R/stage5_ordinal_dml.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages({library(ordinal); library(splines)})

theta0 <- 1.0
n_reps <- 200

dgp5 <- function(n, theta, cuts = c(0, 2)) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  U <- theta * D + g + rlogis(n)
  list(X = X, D = D, Y = 1L + findInterval(U, cuts))
}

# clm full model on D + ns bases, fit once on all data (the stage-2/4
# design: a parametric model does not overfit, and cross-fitting the full
# model injects per-fold index noise that badly inflates the between-draw
# variance). Proper draws perturb (thresholds, beta) jointly.
sudo_ordinal <- function(d, B = 25, n_folds = 5, df_ns = 4) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  J <- max(d$Y)
  folds <- make_folds(n, n_folds)
  basis <- do.call(cbind, lapply(X, function(x) ns(x, df = df_ns)))
  colnames(basis) <- paste0("B", seq_len(ncol(basis)))
  mm <- cbind(D = d$D, basis)
  dat <- data.frame(Y = factor(d$Y), mm)

  fit <- clm(Y ~ ., data = dat)
  par_hat <- c(fit$alpha, fit$beta)
  V <- vcov(fit)[names(par_hat), names(par_hat)]
  draw_S <- function(perturb) {
    par <- if (perturb) MASS::mvrnorm(1, par_hat, V) else par_hat
    eta <- as.numeric(mm %*% par[J:length(par)])
    complete_surrogate(d$Y, eta, c(-Inf, par[1:(J - 1)], Inf), "logit")
  }

  D_res <- d$D - crossfit(X, d$D, fit_gam_binomial, folds)
  S_hat <- crossfit(X, draw_S(FALSE), fit_gam, folds)

  draws <- sapply(seq_len(B), function(b) {
    f <- fwl_theta(draw_S(TRUE) - S_hat, D_res)
    c(theta = f$theta, var = f$var)
  })
  p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
  list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi)
}

cl <- mc_cluster(c("dgp5", "sudo_ordinal"))
invisible(parallel::clusterEvalQ(cl,
  suppressPackageStartupMessages({library(ordinal); library(splines)})))
make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp5(n, theta)
}

cat(sprintf("true theta = %.1f, J = 3, clm(ns) full model, gam nuisances\n\n",
            theta0))
all_summ <- list()
for (n in c(1000, 2000)) {
  df <- run_mc_par(cl, n_reps, make_dgp(n, theta0),
                   function(d) sudo_ordinal(d), theta0, seed = 5000)
  s <- summarize_mc(df)
  print_mc(sprintf("ordinal SUDO n=%d", n), s)
  all_summ[[length(all_summ) + 1]] <-
    cbind(stage = 5, estimator = sprintf("ordinal_dml_n%d", n), s)
  # 2.5% relative tolerance: the fixed ns(4) basis leaves a small
  # approximation bias that does not vanish with n; richer bases trade it
  # for worse small-sample behavior (df=6 doubles the n=1000 bias)
  stopifnot(abs(s$bias) < max(2 * s$mc_se_bias, 0.025 * abs(theta0)) + 1e-9)
  # Coverage floor only. This design is conservative, but
  # theory_ordinal_variance_terms.R shows that ordinal Rubin pooling itself
  # differs from the target variance by only a few percent in a clean
  # parametric testbed. The much larger gap here remains a nuisance-side
  # finite-sample question, not a general safety property of Rubin's rules.
  stopifnot(s$coverage >= 0.925)
}
parallel::stopCluster(cl)
cat("\nPASS: bias within tolerance; coverage >= 0.925 (conservative)\n")

write.csv(do.call(rbind, all_summ), "R/results/stage5_summary.csv",
          row.names = FALSE)
cat("wrote R/results/stage5_summary.csv\n")
