# Stage 6: link misspecification and diagnostics.
# Latent errors are Gumbel, the analyst assumes logistic — the wrong link
# biases the latent-scale theta; the right link removes it; sure surrogate
# residuals (Greenwell et al. 2018) flag the wrong link on an independent
# draw stream, so diagnosis never reuses estimation draws.
#   A  binary:  U = theta*D + beta*X + gumbel_max  (glm cloglog is the truth)
#   B  ordinal: U = theta*D + beta*X + gumbel_min  (clm cloglog is the truth)
#   C  diagnostic: variance drift of sure residuals in X flags the wrong link
# Run from repo root: Rscript R/stage6_link_misspec.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages({library(ordinal); library(sure)})

theta0 <- 1.0
n_obs <- 5000
B <- 25
n_reps <- 200
all_summ <- list()

pool_draws <- function(draws, n) {
  p <- pool_rubin(draws["theta", ], draws["var", ], n_obs = n)
  list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi)
}

cat("== A: binary, Gumbel-max errors (cloglog truth) ==\n")
dgp_bin <- function(n = n_obs) {
  X <- rnorm(n)
  D <- rbinom(n, 1, plogis(0.8 * X))
  U <- theta0 * D + X + (-log(-log(runif(n))))
  list(X = X, D = D, Y = as.integer(U > 0))
}

est_bin <- function(d, link) {
  glm_link <- if (link == "cloglog") "cloglog" else "logit"
  fit <- glm(d$Y ~ d$D + d$X, family = binomial(link = glm_link))
  mm <- model.matrix(fit)
  V <- vcov(fit)
  rD <- resid(lm(d$D ~ d$X))
  draws <- sapply(seq_len(B), function(b) {
    lp <- as.numeric(mm %*% MASS::mvrnorm(1, coef(fit), V))
    S <- complete_surrogate(d$Y + 1L, lp, c(-Inf, 0, Inf), link)
    f <- fwl_theta(resid(lm(S ~ d$X)), rD)
    c(theta = f$theta, var = f$var)
  })
  pool_draws(draws, length(d$Y))
}

for (link in c("logit", "cloglog")) {
  res <- t(sapply(seq_len(n_reps), function(r) {
    set.seed(r)
    e <- est_bin(dgp_bin(), link)
    c(est = e$theta, se = e$se,
      covered = e$ci_lo <= theta0 & theta0 <= e$ci_hi)
  }))
  df <- data.frame(est = res[, "est"], se = res[, "se"],
                   covered = as.logical(res[, "covered"]))
  s <- summarize_mc(df, theta0)
  print_mc(sprintf("binary %s", link), s)
  all_summ[[length(all_summ) + 1]] <-
    cbind(stage = 6, estimator = sprintf("binary_%s", link), s)
}

cat("\n== B: ordinal J=3, Gumbel-min errors (clm cloglog truth) ==\n")
dgp_ord <- function(n = n_obs, cuts = c(0, 2)) {
  X <- rnorm(n)
  D <- rbinom(n, 1, plogis(0.8 * X))
  U <- theta0 * D + X + log(-log(runif(n)))
  list(X = X, D = D, Y = 1L + findInterval(U, cuts))
}

est_ord <- function(d, link) {
  clm_link <- if (link == "cloglog_min") "cloglog" else "logit"
  fit <- clm(factor(Y) ~ D + X, link = clm_link,
             data = data.frame(Y = d$Y, D = d$D, X = d$X))
  par_hat <- c(fit$alpha, fit$beta)
  V <- vcov(fit)[names(par_hat), names(par_hat)]
  mm <- cbind(D = d$D, X = d$X)
  J <- length(fit$alpha) + 1
  rD <- resid(lm(d$D ~ d$X))
  draws <- sapply(seq_len(B), function(b) {
    par <- MASS::mvrnorm(1, par_hat, V)
    eta <- as.numeric(mm %*% par[J:length(par)])
    S <- complete_surrogate(d$Y, eta, c(-Inf, par[1:(J - 1)], Inf), link)
    f <- fwl_theta(resid(lm(S ~ d$X)), rD)
    c(theta = f$theta, var = f$var)
  })
  pool_draws(draws, length(d$Y))
}

bias_ord <- c()
for (link in c("logit", "cloglog_min")) {
  res <- t(sapply(seq_len(n_reps), function(r) {
    set.seed(r)
    e <- est_ord(dgp_ord(), link)
    c(est = e$theta, se = e$se,
      covered = e$ci_lo <= theta0 & theta0 <= e$ci_hi)
  }))
  df <- data.frame(est = res[, "est"], se = res[, "se"],
                   covered = as.logical(res[, "covered"]))
  s <- summarize_mc(df, theta0)
  print_mc(sprintf("ordinal %s", link), s)
  bias_ord[link] <- s$bias
  all_summ[[length(all_summ) + 1]] <-
    cbind(stage = 6, estimator = sprintf("ordinal_%s", link), s)
}

# wrong link visibly biased, right link clean, in both A and B
sm <- do.call(rbind, all_summ)
b <- function(lab) sm[sm$estimator == lab, ]
stopifnot(abs(b("binary_logit")$bias) > 4 * b("binary_logit")$mc_se_bias)
stopifnot(abs(b("binary_cloglog")$bias) < 2 * b("binary_cloglog")$mc_se_bias)
stopifnot(abs(b("ordinal_logit")$bias) > 4 * b("ordinal_logit")$mc_se_bias)
stopifnot(abs(b("ordinal_cloglog_min")$bias) <
          2 * b("ordinal_cloglog_min")$mc_se_bias)
cat("\nPASS: wrong link biased, right link unbiased (binary and ordinal)\n")

cat("\n== C: sure residual diagnostic (independent draw stream) ==\n")
# Under a correct link, surrogate residuals are homoscedastic (Cheng et al.
# 2021); a wrong link makes their conditional VARIANCE drift with X. Mean-
# structure tests are blind here — flexible thresholds absorb conditional-mean
# misfit — so we test var drift: gam(r^2 ~ s(X)) smooth p-value, median over
# 10 draws. Diagnosis uses its own RNG stream, never the estimation draws.
set.seed(987654)
d <- dgp_ord()
dat <- data.frame(Y = factor(d$Y), D = d$D, X = d$X)
fit_wrong <- clm(Y ~ D + X, link = "logit", data = dat)
fit_right <- clm(Y ~ D + X, link = "cloglog", data = dat)
var_drift_p <- function(fit) {
  median(replicate(10, {
    r <- as.numeric(resids(fit))
    summary(mgcv::gam(I(r^2) ~ s(dat$X)))$s.pv
  }))
}
p_wrong <- var_drift_p(fit_wrong)
p_right <- var_drift_p(fit_right)
cat(sprintf("var-drift p (logit fit, wrong):    %.2e  <- flags misspec\n",
            p_wrong))
cat(sprintf("var-drift p (cloglog fit, right):  %.3f\n", p_right))
stopifnot(p_wrong < 0.05, p_right > 0.05)
cat("PASS: diagnostic flags the wrong link and clears the right one\n")

sm <- rbind(sm, data.frame(stage = 6, estimator = c("vardrift_wrong", "vardrift_right"),
                           n_reps = 10, mean_est = c(p_wrong, p_right),
                           bias = NA, mc_se_bias = NA, sd = NA, rmse = NA,
                           mean_se = NA, coverage = NA, mc_se_cov = NA))
write.csv(sm, "R/results/stage6_summary.csv", row.names = FALSE)
cat("wrote R/results/stage6_summary.csv\n")
