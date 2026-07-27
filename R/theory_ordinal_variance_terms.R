# Theory check: fixed-B variance terms for an ordinal cumulative-logit model.
#
# This is the ordinal counterpart of R/theory_variance_terms.R. It isolates
# threshold estimation and Rubin pooling from flexible-nuisance behavior:
#
#   X, R iid N(0, 1), D = 0.8 X + R,
#   U = theta0 D + X + logistic error,
#   Y = j iff cut_j-1 < U <= cut_j.
#
# The cumulative-link imputation model and both partialling regressions are
# correctly specified. Population terms include the threshold block of G:
#
#   V_B = J^-2 [sigma_e^2 + G' Sigma G + 2 G'C
#          + (G' Sigma G + sigma_u^2) / B],
#
#   E[n T] - V_B = 2 J^-2 (sigma_u^2 - G'C).
#
# Run from the repository root:
#   Rscript R/theory_ordinal_variance_terms.R
#
# Optional smoke-test overrides:
#   SUDO_THEORY_N=3000 SUDO_THEORY_REPS=100 Rscript ...

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
suppressPackageStartupMessages({
  library(ordinal)
  library(parallel)
})

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

logit_trunc_mean <- function(lo, hi) {
  partial <- function(t) {
    if (is.infinite(t) && t < 0) return(0)
    if (is.infinite(t) && t > 0) return(0)
    q <- plogis(t)
    if (q <= 0 || q >= 1) return(0)
    q * log(q) + (1 - q) * log1p(-q)
  }
  prob <- plogis(hi) - plogis(lo)
  (partial(hi) - partial(lo)) / prob
}

ordinal_conditional <- function(eta, cuts) {
  n <- length(eta)
  n_cat <- length(cuts) + 1L
  lo <- c(-Inf, cuts)
  hi <- c(cuts, Inf)
  prob <- matrix(NA_real_, n, n_cat)
  mu <- matrix(NA_real_, n, n_cat)
  for (j in seq_len(n_cat)) {
    lower <- lo[j] - eta
    upper <- hi[j] - eta
    prob[, j] <- plogis(upper) - plogis(lower)
    mu[, j] <- mapply(logit_trunc_mean, lower, upper)
  }
  list(prob = prob, mu = mu)
}

pop_terms_ordinal <- function(theta0, cuts = c(0, 2),
                              ngrid = 301L, span = 5) {
  node <- seq(-span, span, length.out = ngrid)
  wt <- dnorm(node) * (node[2] - node[1])
  X <- rep(node, times = ngrid)
  R <- rep(node, each = ngrid)
  w <- rep(wt, times = ngrid) * rep(wt, each = ngrid)
  w <- w / sum(w)

  D <- 0.8 * X + R
  eta <- theta0 * D + X
  z <- cbind(D, X)
  cond <- ordinal_conditional(eta, cuts)
  prob <- cond$prob
  mu <- cond$mu
  n_cat <- ncol(prob)
  n_cut <- length(cuts)

  density_cut <- vapply(cuts, function(cut) dlogis(cut - eta),
                        numeric(length(eta)))
  if (n_cut == 1L) density_cut <- matrix(density_cut, ncol = 1L)
  threshold_kernel <- density_cut * (mu[, 2:n_cat, drop = FALSE] -
                                       mu[, 1:(n_cat - 1L), drop = FALSE])
  pass <- 1 - rowSums(threshold_kernel)

  G_threshold <- colSums(w * R * threshold_kernel)
  G_coefficient <- as.numeric(crossprod(z, w * R * pass))
  G <- c(G_threshold, G_coefficient)

  n_par <- n_cut + ncol(z)
  info <- matrix(0, n_par, n_par)
  score_eR <- numeric(n_par)
  for (j in seq_len(n_cat)) {
    dp_cut <- matrix(0, nrow = length(eta), ncol = n_cut)
    if (j <= n_cut) dp_cut[, j] <- density_cut[, j]
    if (j > 1L) dp_cut[, j - 1L] <- dp_cut[, j - 1L] -
      density_cut[, j - 1L]

    f_lo <- if (j == 1L) 0 else density_cut[, j - 1L]
    f_hi <- if (j == n_cat) 0 else density_cut[, j]
    dp_beta <- (f_lo - f_hi) * z
    score <- cbind(dp_cut, dp_beta) / prob[, j]
    info <- info + crossprod(score, w * prob[, j] * score)
    score_eR <- score_eR +
      as.numeric(crossprod(score, w * prob[, j] * mu[, j] * R))
  }
  Sigma <- solve(info)
  C <- as.numeric(Sigma %*% score_eR)

  J <- sum(w * R^2)
  sig_e <- sum(w * R^2 * rowSums(prob * mu^2))
  sig_psi <- pi^2 / 3 * J
  sig_u <- sig_psi - sig_e
  GtSigmaG <- drop(G %*% Sigma %*% G)
  GtC <- drop(G %*% C)

  list(
    theta0 = theta0, cuts = cuts, J = J, G = G, Sigma = Sigma, C = C,
    sig_e = sig_e, sig_u = sig_u, sig_psi = sig_psi,
    GtSigmaG = GtSigmaG, GtC = GtC,
    V_B = function(B) {
      (sig_e + GtSigmaG + 2 * GtC + (GtSigmaG + sig_u) / B) / J^2
    },
    T_B = function(B) {
      (sig_e + sig_u + (1 + 1 / B) * (GtSigmaG + sig_u)) / J^2
    }
  )
}

one_rep_ordinal <- function(seed, n, theta0, cuts, B_max) {
  set.seed(seed)
  X <- rnorm(n)
  D <- 0.8 * X + rnorm(n)
  U <- theta0 * D + X + rlogis(n)
  Y <- 1L + findInterval(U, cuts)

  dat <- data.frame(Y = ordered(Y), D = D, X = X)
  fit <- suppressWarnings(clm(Y ~ D + X, data = dat, link = "logit"))
  par_hat <- c(fit$alpha, fit$beta)
  covariance <- vcov(fit)[names(par_hat), names(par_hat), drop = FALSE]
  model_matrix <- cbind(D, X)
  n_cat <- length(cuts) + 1L

  q <- qr(cbind(1, X))
  rD <- qr.resid(q, D)
  theta_draw <- numeric(B_max)
  variance_draw <- numeric(B_max)
  for (b in seq_len(B_max)) {
    par_b <- MASS::mvrnorm(1, par_hat, covariance)
    index_b <- as.numeric(model_matrix %*% par_b[n_cat:length(par_b)])
    surrogate <- complete_surrogate(
      Y, index_b, c(-Inf, par_b[seq_len(n_cat - 1L)], Inf), "logit"
    )
    fit_b <- fwl_theta(qr.resid(q, surrogate), rD)
    theta_draw[b] <- fit_b$theta
    variance_draw[b] <- fit_b$var
  }
  c(theta_draw, variance_draw)
}

run_sim_ordinal <- function(n, theta0, cuts, B_set, n_reps,
                            base_seed = 91000L) {
  B_max <- max(B_set)
  n_core <- detectCores()
  if (is.na(n_core)) n_core <- 1L
  result <- mclapply(
    seq_len(n_reps),
    function(rep_id) {
      one_rep_ordinal(base_seed + rep_id, n, theta0, cuts, B_max)
    },
    mc.cores = max(1L, n_core - 1L)
  )
  result <- do.call(rbind, result)
  theta_draw <- result[, seq_len(B_max), drop = FALSE]
  variance_draw <- result[, B_max + seq_len(B_max), drop = FALSE]

  do.call(rbind, lapply(B_set, function(B) {
    theta_bar <- rowMeans(theta_draw[, seq_len(B), drop = FALSE])
    V_emp <- n * var(theta_bar)
    if (B >= 2L) {
      nT <- n * (
        rowMeans(variance_draw[, seq_len(B), drop = FALSE]) +
          (1 + 1 / B) *
          apply(theta_draw[, seq_len(B), drop = FALSE], 1, var)
      )
      T_emp <- mean(nT)
      T_emp_se <- sd(nT) / sqrt(n_reps)
    } else {
      T_emp <- NA_real_
      T_emp_se <- NA_real_
    }
    data.frame(
      B = B,
      V_emp = V_emp,
      V_emp_se = V_emp * sqrt(2 / (n_reps - 1)),
      T_emp = T_emp,
      T_emp_se = T_emp_se
    )
  }))
}

n <- env_int("SUDO_THEORY_N", 10000L)
n_reps <- env_int("SUDO_THEORY_REPS", 1000L)
B_set <- c(1L, 2L, 5L, 20L)
cuts <- c(0, 2)
theta_set <- c(1, 2.5)

all_results <- list()
for (theta0 in theta_set) {
  population <- pop_terms_ordinal(theta0, cuts)
  cat("\n== Ordinal population terms, theta0 =", theta0, "==\n")
  cat(sprintf("  J_theta        %.5f\n", population$J))
  cat(sprintf("  sigma_e^2      %.5f\n", population$sig_e))
  cat(sprintf("  sigma_u^2      %.5f\n", population$sig_u))
  cat(sprintf("  G' Sigma G     %.5f\n", population$GtSigmaG))
  cat(sprintf("  G' C           %.5f\n", population$GtC))
  cat(sprintf("  threshold G    (%s)\n",
              paste(sprintf("%.5f", population$G[seq_along(cuts)]),
                    collapse = ", ")))
  cat(sprintf("  gap            %+.5f\n",
              population$sig_u - population$GtC))
  cat(sprintf("  predicted T/V  %.4f at B=20\n",
              population$T_B(20) / population$V_B(20)))
  cat("  simulation: n =", n, "reps =", n_reps, "\n")

  simulation <- run_sim_ordinal(
    n, theta0, cuts, B_set, n_reps,
    base_seed = 91000L + as.integer(theta0 * 1000)
  )
  simulation$V_pred <- vapply(simulation$B, population$V_B, numeric(1))
  simulation$T_pred <- vapply(simulation$B, population$T_B, numeric(1))
  simulation$V_z <- (simulation$V_emp - simulation$V_pred) /
    simulation$V_emp_se
  simulation$T_z <- (simulation$T_emp - simulation$T_pred) /
    simulation$T_emp_se
  simulation$ratio_pred <- simulation$T_pred / simulation$V_pred
  print(format(cbind(theta0 = theta0, simulation), digits = 4),
        row.names = FALSE)

  # Full validation uses at least 500 replications. Smoke tests exercise the
  # implementation without enforcing Monte Carlo agreement.
  if (n_reps >= 500L) {
    stopifnot(max(abs(simulation$V_z)) < 4)
    stopifnot(max(abs(simulation$T_z), na.rm = TRUE) < 4)
  }

  all_results[[length(all_results) + 1L]] <- cbind(
    theta0 = theta0,
    simulation,
    sig_e = population$sig_e,
    sig_u = population$sig_u,
    GtSigmaG = population$GtSigmaG,
    GtC = population$GtC,
    G_threshold_1 = population$G[1],
    G_threshold_2 = population$G[2]
  )
}

if (n_reps >= 500L) {
  cat("\nPASS: ordinal fixed-B and Rubin-variance predictions match",
      "Monte Carlo in every cell (|z| < 4)\n")
} else {
  cat("\nSMOKE PASS: rerun with SUDO_THEORY_REPS >= 500 for",
      "acceptance checks\n")
}

output <- do.call(rbind, all_results)
dir.create("R/results", showWarnings = FALSE)
write.csv(output, "R/results/theory_ordinal_variance_terms.csv",
          row.names = FALSE)
cat("wrote R/results/theory_ordinal_variance_terms.csv\n")
