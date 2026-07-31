# Stage 9: does the completion correction repair regularisation bias?
#
# Stage 8 showed that SUDO built on an unpenalised cumulative-link fit
# reproduces that model's own treatment coefficient (bias -0.1846 against
# -0.1840). That raises the obvious objection: if a partially-linear
# imputation model V = alpha*D + f(X) already contains an extractable
# alpha_hat, why complete the outcome and run FWL at all?
#
# The exact answer is an identity. With R = D - m_0(X) annihilating functions
# of X, and mu_Y = E[xi | Y, v_hat] the conditional mean of the truncated
# latent error,
#
#   theta_SUDO = alpha_hat + E[R * mu_Y] / E[R^2].
#
# For a binary logistic imputation model mu_Y = H(p)/{p(1-p)} * (Y - p) with
# H the binary entropy, so the correction is an entropy-weighted residual
# moment. SUDO reads the coefficient and then corrects it. The correction can
# be zero, helpful, or harmful.
#
# The one remaining argument for the completion detour is that PENALISED
# fitting biases alpha_hat toward zero while the correction, which uses the
# observed Y, repairs it. This stage tests that directly: it walks the mgcv
# smoothing parameter from nearly unpenalised to severe on common data and
# reports, at each penalty,
#
#   alpha_hat        the direct read-off
#   C_hat            the exact entropy-weighted correction
#   alpha_hat+C_hat  the identity's prediction
#   theta_sudo       operational SUDO with a Rao-Blackwellised completion
#   treat / leak     the first-order decomposition of the SUDO error
#
# The verdict:
#   - the completion route is justified if increasing penalisation moves
#     alpha_hat away from the truth while theta_sudo stays near it;
#   - it is not if theta_sudo simply tracks alpha_hat.
#
# The completion is analytic (the B -> infinity conditional mean), so no
# finite-B Monte Carlo noise enters. The treatment nuisance is the oracle
# m_0(X), isolating the imputation model as the only source of error.
#
# Run from the repository root:
#   Rscript R/stage9_penalty_path.R
#
# Optional smoke-test overrides:
#   SUDO_STAGE9_N=500 SUDO_STAGE9_REPS=4 Rscript R/stage9_penalty_path.R

source("R/sudo/fwl.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

N_OBS <- env_int("SUDO_STAGE9_N", 2000)
N_REPS <- env_int("SUDO_STAGE9_REPS", 100)
THETA0 <- 1.5
SP_GRID <- c(1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)

binary_entropy <- function(p) {
  out <- numeric(length(p))
  ok <- p > 0 & p < 1
  out[ok] <- -p[ok] * log(p[ok]) - (1 - p[ok]) * log1p(-p[ok])
  out
}

# g_0 is wiggly but representable by the s() bases used to fit it, so any
# gap is penalisation rather than approximation error. D depends strongly on
# both covariates while keeping overlap.
g0_fun <- function(X) sin(2 * X[, 1]) + 0.5 * X[, 2]^2
m0_fun <- function(X) plogis(1.5 * X[, 1] + 0.8 * X[, 2])

dgp9 <- function(n, theta0) {
  X <- matrix(rnorm(n * 2), n, 2, dimnames = list(NULL, c("X1", "X2")))
  m0 <- m0_fun(X)
  D <- rbinom(n, 1, m0)
  v0 <- theta0 * D + g0_fun(X)
  list(X = X, D = D, Y = rbinom(n, 1, plogis(v0)), m0 = m0, v0 = v0)
}

one_rep <- function(d, sp_grid, theta0, n_folds = 5L) {
  X <- as.data.frame(d$X)
  n <- nrow(X)
  folds <- make_folds(n, n_folds)
  R <- d$D - d$m0                       # oracle treatment residual
  J <- mean(R^2)
  c0 <- 1 - binary_entropy(plogis(d$v0))  # pass-through at the truth
  g0 <- g0_fun(d$X)
  dat <- data.frame(Y = d$Y, D = d$D, X)

  do.call(rbind, lapply(sp_grid, function(sp) {
    fit <- mgcv::gam(Y ~ D + s(X1) + s(X2), family = binomial, data = dat,
                     sp = c(sp, sp))
    alpha <- unname(coef(fit)["D"])
    v <- as.numeric(predict(fit))
    p_raw <- plogis(v)
    # H(p)/{p(1-p)} diverges like -log(p) at the boundary, so a nearly
    # unpenalised flexible fit can push it to Inf. Clip as the PL learners
    # do, and count how often it bites: a high rate is itself the finding.
    p <- pmin(pmax(p_raw, 1e-6), 1 - 1e-6)
    clip_rate <- mean(p_raw < 1e-6 | p_raw > 1 - 1e-6)
    Hp <- binary_entropy(p)

    # exact entropy-weighted correction
    C_hat <- sum(R * Hp / (p * (1 - p)) * (d$Y - p)) / sum(R^2)

    # analytic Rao-Blackwellised completion: E[xi | Y, v]
    mu_Y <- ifelse(d$Y == 1L, Hp / p, -Hp / (1 - p))
    S <- v + mu_Y
    ell <- crossfit(X, S, fit_gam, folds)
    theta_sudo <- sum((S - ell) * R) / sum(R^2)

    # first-order decomposition of the SUDO error
    f_hat <- v - alpha * d$D
    treat_term <- (alpha - theta0) * mean(R * c0 * d$D) / J
    leak_term <- mean(R * c0 * (f_hat - g0)) / J

    data.frame(sp = sp, alpha = alpha, C_hat = C_hat, clip_rate = clip_rate,
               alpha_plus_C = alpha + C_hat, theta_sudo = theta_sudo,
               treat_term = treat_term, leak_term = leak_term)
  }))
}

make_dgp <- function(n, theta0) {
  force(n); force(theta0)
  function() dgp9(n, theta0)
}

cluster <- mc_cluster(c("dgp9", "one_rep", "binary_entropy", "g0_fun",
                        "m0_fun", "SP_GRID", "THETA0"))
parallel::clusterExport(cluster, c("N_OBS"), envir = environment())

cat(sprintf("Stage 9: penalty path, n=%d, %d replications, theta0=%.1f\n",
            N_OBS, N_REPS, THETA0))
cat("Does the entropy-weighted correction repair regularisation bias?\n\n")

reps <- parallel::parLapply(cluster, seq_len(N_REPS), function(r) {
  set.seed(9000 + r)
  one_rep(dgp9(N_OBS, THETA0), SP_GRID, THETA0)
})
parallel::stopCluster(cluster)

all_reps <- do.call(rbind, reps)
agg <- do.call(rbind, lapply(split(all_reps, all_reps$sp), function(b) {
  data.frame(
    sp = b$sp[1],
    alpha_bias = mean(b$alpha) - THETA0,
    C_hat = mean(b$C_hat),
    identity_bias = mean(b$alpha_plus_C) - THETA0,
    sudo_bias = mean(b$theta_sudo) - THETA0,
    mc_se_sudo = sd(b$theta_sudo) / sqrt(nrow(b)),
    identity_gap = mean(abs(b$alpha_plus_C - b$theta_sudo)),
    treat_term = mean(b$treat_term),
    leak_term = mean(b$leak_term),
    clip_rate = mean(b$clip_rate),
    sd_alpha = sd(b$alpha),
    sd_sudo = sd(b$theta_sudo)
  )
}))
agg <- agg[order(agg$sp), ]

cat(sprintf("%9s %11s %9s %11s %11s %9s %9s\n",
            "sp", "alpha_bias", "C_hat", "sudo_bias", "|id gap|",
            "treat", "leak"))
for (i in seq_len(nrow(agg))) {
  cat(sprintf("%9.0e %+11.4f %+9.4f %+11.4f %11.2e %+9.4f %+9.4f\n",
              agg$sp[i], agg$alpha_bias[i], agg$C_hat[i], agg$sudo_bias[i],
              agg$identity_gap[i], agg$treat_term[i], agg$leak_term[i]))
}

# The identity theta_SUDO = alpha_hat + C_hat is asymptotic, not exact in
# finite samples: it needs the estimated outcome nuisance to remove the
# X-part of v_hat, which it does only to O(n^{-1/2}). The principled check is
# therefore that the identity slack is smaller than the estimator's own
# sampling noise, so it cannot distort the comparison being made. (At
# n = 200,000 with an oracle-quality nuisance the gap is 5e-5.)
stopifnot(
  nrow(agg) == length(SP_GRID),
  all(is.finite(unlist(agg))),
  all(agg$identity_gap < agg$sd_sudo)
)
cat(sprintf("\nPASS: identity slack %.4f max, below SUDO sampling SD %.4f\n",
            max(agg$identity_gap), min(agg$sd_sudo)))

worst_alpha <- agg$sp[which.max(abs(agg$alpha_bias))]
cat(sprintf("      direct alpha worst at sp=%.0e (bias %+.4f), SUDO there %+.4f\n",
            worst_alpha, agg$alpha_bias[agg$sp == worst_alpha],
            agg$sudo_bias[agg$sp == worst_alpha]))

dir.create("R/results", showWarnings = FALSE)
write.csv(agg, "R/results/stage9_penalty_path.csv", row.names = FALSE)
cat("wrote R/results/stage9_penalty_path.csv\n")
