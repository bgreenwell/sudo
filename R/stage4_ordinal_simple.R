# Stage 4: ordinal (J=3) surrogate SUDO, parametric everything.
# Latent U = theta*D + beta*X + logis, cut at (-1, 1) into Y in {1,2,3}.
# Full model: ordinal::clm — its vcov includes the thresholds, so proper-MI
# draws perturb (thresholds, beta) jointly from N(param_hat, vcov) before
# sampling S from the interval-truncated logistic. FWL + Rubin as in stage 2.
# clm parameterization: P(Y <= j | x) = plogis(alpha_j - eta), latent U = eta + e.
# Run from repo root: Rscript R/stage4_ordinal_simple.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(ordinal))

theta0 <- 1.0
n_obs <- 3000
B <- 25
n_reps <- 500

dgp_ord <- function(n = n_obs, theta = theta0, beta = 1.0, cuts = c(-1, 1)) {
  X <- rnorm(n)
  D <- rbinom(n, 1, plogis(0.8 * X))
  U <- theta * D + beta * X + rlogis(n)
  Y <- 1L + findInterval(U, cuts)
  list(X = X, D = D, Y = Y, U = U)
}

est_ord <- function(d) {
  rD <- resid(lm(d$D ~ d$X))
  fit <- clm(factor(Y) ~ D + X, data = data.frame(Y = d$Y, D = d$D, X = d$X))
  par_hat <- c(fit$alpha, fit$beta)
  V <- vcov(fit)[names(par_hat), names(par_hat)]
  mm <- cbind(D = d$D, X = d$X)
  J <- length(fit$alpha) + 1

  draw_one <- function(par) {
    alpha <- par[1:(J - 1)]
    eta <- as.numeric(mm %*% par[J:length(par)])
    S <- complete_surrogate(d$Y, eta, c(-Inf, alpha, Inf), "logit")
    f <- fwl_theta(resid(lm(S ~ d$X)), rD)
    c(theta = f$theta, var = f$var)
  }
  draws <- sapply(seq_len(B), function(b)
    draw_one(MASS::mvrnorm(1, par_hat, V)))
  p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = length(d$Y))

  naive <- draw_one(par_hat)
  oracle <- fwl_theta(resid(lm(d$U ~ d$X)), rD)$theta
  list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi,
       naive_theta = naive["theta"], naive_se = sqrt(naive["var"]),
       oracle = oracle, clm_beta = unname(fit$beta["D"]))
}

rows <- lapply(seq_len(n_reps), function(r) {
  set.seed(r)
  e <- est_ord(dgp_ord())
  data.frame(rep = r, est = e$theta, se = e$se,
             covered = e$ci_lo <= theta0 & theta0 <= e$ci_hi,
             naive = e$naive_theta,
             naive_cov = abs(e$naive_theta - theta0) <= 1.96 * e$naive_se,
             oracle = e$oracle, clm_beta = e$clm_beta)
})
df <- do.call(rbind, rows)
attr(df, "theta_true") <- theta0

cat(sprintf("true theta = %.1f, n = %d, J = 3, B = %d, %d reps\n\n",
            theta0, n_obs, B, n_reps))
s <- summarize_mc(df)
print_mc("SUDO ordinal (proper MI)", s)
cat(sprintf("%-28s bias %+ .4f                mean over reps (reference)\n",
            "clm coefficient on D", mean(df$clm_beta) - theta0))
cat(sprintf("%-28s bias %+ .4f                (exact linear partialling)\n",
            "oracle FWL on latent U", mean(df$oracle) - theta0))
cat(sprintf("%-28s cover %.3f                (single draw, no pooling)\n\n",
            "naive single-draw CI", mean(df$naive_cov)))

stopifnot(abs(s$bias) < 2 * s$mc_se_bias + 1e-9)
# upper edge 0.98: proper draws are mildly conservative; under-coverage is the
# failure mode that matters
stopifnot(s$coverage >= 0.925, s$coverage <= 0.98)
cat("PASS: bias < 2*MC-SE, Rubin-t coverage in [0.925, 0.98]\n")

write.csv(cbind(stage = 4, estimator = "sudo_ordinal_clm", s),
          "R/results/stage4_summary.csv", row.names = FALSE)
cat("wrote R/results/stage4_summary.csv\n")
