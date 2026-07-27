# Stage 7: how does SUDO hold up across DGP variations?
#
# Motivation. Every simulation design in the paper uses a BINARY treatment,
# while the wine application's treatment (volatile acidity) is CONTINUOUS, and
# Proposition A3's bias formula is explicitly restricted to binary D (the
# continuous case has a different multiplier, E[c(eta_0) D R] / E[R^2]). So
# the continuous-treatment arms here close a real validation gap rather than
# just adding variety. The remaining arms probe structure the paper's two
# covariate settings do not: an interaction in g_0 that additive gam nuisances
# cannot represent, higher dimension, and weak overlap.
#
# All arms use a correctly specified imputation model, so any bias is
# attributable to the DGP feature under test rather than to full-model error.
#
# Designs (all latent-logistic, n = 2000, theta_0 in {1, 2.5}):
#   base        S5 as in the paper: binary D, binary Y            reference
#   contD       continuous D, binary Y                            the gap
#   contD_ord   continuous D, ordinal J=3                         wine analogue
#   interact    binary D, g_0 gains an X1*X2 term                 additive nuisances misspecified
#   p10         binary D, 10 covariates, g_0 sparse in them       dimension
#   weak_ovl    binary D, propensity scaled 2.5x                  overlap
#
# Exploratory stage: the assertions cover mechanical correctness and the
# scientific reading is printed for a human.
#
# Run from repo root: Rscript R/stage7_dgp_variation.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages({library(mgcv); library(ordinal); library(splines)})
source("R/sudo/estimator.R")

# ---- data-generating processes -------------------------------------------
# Each returns list(X, D, Y). Y is binary 0/1 unless the design is ordinal,
# in which case it is 1..J.

make_dgp <- function(design, n, theta) {
  force(design); force(n); force(theta)
  function() {
    p <- if (design == "p10") 10 else 5
    X <- matrix(rnorm(n * p), n, p,
                dimnames = list(NULL, paste0("X", seq_len(p))))
    g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
    if (design == "interact") g <- g + X[, 1] * X[, 2]
    pi_x <- X[, 4] + cos(X[, 5])
    if (design == "weak_ovl") pi_x <- 2.5 * pi_x
    D <- if (design %in% c("contD", "contD_ord")) {
      0.7 * pi_x + rnorm(n)                    # Var(D | X) = 1, good overlap
    } else {
      rbinom(n, 1, plogis(pi_x))
    }
    U <- theta * D + g + rlogis(n)
    Y <- if (design == "contD_ord") 1L + findInterval(U, c(0, 2)) else
      as.integer(U > 0)
    list(X = X, D = D, Y = Y)
  }
}

# ---- ordinal estimator with a pluggable treatment nuisance ---------------
# stage5's sudo_ordinal hardcodes a binomial nuisance for D; continuous D
# needs a Gaussian one. Otherwise identical: clm on D plus spline bases, fit
# once (parametric full model), proper draws from its covariance.

sudo_ordinal2 <- function(d, B = 25, n_folds = 5, df_ns = 4,
                          fit_m = fit_gam) {
  X <- as.data.frame(d$X)
  n <- nrow(X); J <- max(d$Y)
  folds <- make_folds(n, n_folds)
  basis <- do.call(cbind, lapply(X, function(x) ns(x, df = df_ns)))
  colnames(basis) <- paste0("B", seq_len(ncol(basis)))
  mm <- cbind(D = d$D, basis)
  fit <- clm(Y ~ ., data = data.frame(Y = factor(d$Y), mm))
  par_hat <- c(fit$alpha, fit$beta)
  V <- vcov(fit)[names(par_hat), names(par_hat)]
  draw_S <- function(perturb) {
    par <- if (perturb) MASS::mvrnorm(1, par_hat, V) else par_hat
    eta <- as.numeric(mm %*% par[J:length(par)])
    complete_surrogate(d$Y, eta, c(-Inf, par[1:(J - 1)], Inf), "logit")
  }
  D_res <- d$D - crossfit(X, d$D, fit_m, folds)
  S_hat <- crossfit(X, draw_S(FALSE), fit_gam, folds)
  draws <- sapply(seq_len(B), function(b) {
    f <- fwl_theta(draw_S(TRUE) - S_hat, D_res)
    c(theta = f$theta, var = f$var)
  })
  p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
  list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi)
}

make_est <- function(design) {
  force(design)
  cont_D <- design %in% c("contD", "contD_ord")
  fm <- if (cont_D) fit_gam else fit_gam_binomial   # nuisance for E[D|X]
  if (design == "contD_ord") {
    function(d) sudo_ordinal2(d, B = 25, fit_m = fm)
  } else {
    function(d) sudo_binary(d, B = 25, full_model = "gam", fit_m = fm)
  }
}

# ---- run -----------------------------------------------------------------

n <- 2000
n_reps <- 200
designs <- c("base", "contD", "contD_ord", "interact", "p10", "weak_ovl")

cl <- mc_cluster(c("make_dgp", "sudo_ordinal2"))
invisible(parallel::clusterEvalQ(cl, {
  suppressPackageStartupMessages({library(ordinal); library(splines)})
  source("R/sudo/estimator.R")
}))
on.exit(parallel::stopCluster(cl), add = TRUE)

cat(sprintf("stage 7: DGP variation, n = %d, %d reps, correctly specified full model\n\n",
            n, n_reps))

out <- list()
for (design in designs) {
  for (theta0 in c(1, 2.5)) {
    t0 <- Sys.time()
    df <- run_mc_par(cl, n_reps, make_dgp(design, n, theta0),
                     make_est(design), theta0,
                     seed = if (theta0 == 1) 4000 else 6000)
    s <- summarize_mc(df)
    print_mc(sprintf("%-10s theta=%.1f", design, theta0), s)
    out[[length(out) + 1]] <- cbind(
      stage = 7, design = design, theta0 = theta0, n = n, s,
      sd_over_mean_se = s$sd / s$mean_se,
      minutes = as.numeric(difftime(Sys.time(), t0, units = "mins")))
    dir.create("R/results", showWarnings = FALSE)
    write.csv(do.call(rbind, out), "R/results/stage7_summary.csv",
              row.names = FALSE)
  }
}

res <- do.call(rbind, out)
cat("\n")
print(format(res[, c("design", "theta0", "bias", "mc_se_bias", "sd",
                     "mean_se", "sd_over_mean_se", "coverage", "mc_se_cov",
                     "minutes")], digits = 3), row.names = FALSE)

stopifnot(all(is.finite(res$bias)), all(res$mean_se > 0))
cat("\nPASS: every design produced finite estimates and positive standard errors\n")

cat("\nReading (relative bias and coverage):\n")
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  flag <- if (abs(r$bias) > max(3 * r$mc_se_bias, 0.03 * r$theta0)) " <- bias" else
    if (r$coverage < 0.90) " <- coverage" else ""
  cat(sprintf("  %-10s theta=%.1f  bias %+.3f (%.1f%%)  cover %.3f  sd/mean_se %.2f%s\n",
              r$design, r$theta0, r$bias, 100 * r$bias / r$theta0,
              r$coverage, r$sd_over_mean_se, flag))
}
cat("\nwrote R/results/stage7_summary.csv\n")
