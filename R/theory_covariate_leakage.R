# Theory check: exact and bounded covariate-direction leakage.
#
# For binary D with propensity m(X), w(X)=m(X){1-m(X)}, and a common
# covariate-direction index error h(X), the exact generated-outcome shift is
#
#   Delta_d(X) = h(X) cbar_d(X),
#   cbar_d(X) =
#     integral_0^1 partial_v psi{eta_d(X) + t h(X); eta_d(X)} dt.
#
# Therefore the population FWL bias is
#
#   leak = E[w h (cbar_1 - cbar_0)] / E[w],
#
# and weighted Cauchy-Schwarz gives
#
#   |leak| <= ||h||_(L2(w)) ||cbar_1-cbar_0||_(L2(w)) / E[w].
#
# A sharper local bound splits off the first-order term and applies
# Cauchy-Schwarz only to cbar_d-c(eta_d), which is second order in h. This
# stage evaluates the identity, both bounds, and the local approximation in
# the stage-7 interaction design. The analyst omits lambda X1 X2. Omitted
# logistic heterogeneity also moves the pseudo-true treatment coefficient,
# so the stage reports that treatment component separately. At lambda=1 the
# combined predicted bias is compared with the committed finite-sample
# stage-7 bias.
#
# Full default: two million population draws.
# Smoke check:
#   SUDO_LEAK_N=100000 Rscript R/theory_covariate_leakage.R

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

entropy_logit <- function(eta) {
  lp <- plogis(eta, log.p = TRUE)
  lq <- plogis(-eta, log.p = TRUE)
  -(exp(lp) * lp + exp(lq) * lq)
}

c_logit <- function(eta) 1 - entropy_logit(eta)

psi_logit <- function(v, eta_true) {
  lp_v <- plogis(v, log.p = TRUE)
  lq_v <- plogis(-v, log.p = TRUE)
  p_v <- exp(lp_v)
  q_v <- exp(lq_v)
  H_v <- -(p_v * lp_v + q_v * lq_v)
  p_true <- plogis(eta_true)
  q_true <- plogis(-eta_true)
  v + p_true * H_v / p_v - q_true * H_v / q_v
}

n_pop <- env_int("SUDO_LEAK_N", 2000000L)
set.seed(190000)
x1 <- rnorm(n_pop)
x2 <- rnorm(n_pop)
x3 <- rnorm(n_pop)
x4 <- rnorm(n_pop)
x5 <- rnorm(n_pop)
interaction <- x1 * x2
g_add <- x1^2 + sin(x2) + 0.5 * x3
m <- plogis(x4 + cos(x5))
w <- m * (1 - m)
J <- mean(w)

stage7 <- read.csv("R/results/stage7_summary.csv")
stage7 <- stage7[stage7$design == "interact", ]
lambda_set <- c(-1.5, -1, -0.5, 0.5, 1, 1.5)
rows <- list()

cat(sprintf(
  "covariate leakage: population draws=%d, J=%.4f\n", n_pop, J
))

for (theta0 in c(1, 2.5)) {
  for (lambda in lambda_set) {
    eta0 <- g_add + lambda * interaction
    eta1 <- theta0 + eta0
    p0 <- plogis(eta0)
    p1 <- plogis(eta1)

    score_sq <- function(par) {
      fitted0 <- plogis(par[1] + g_add)
      fitted1 <- plogis(par[1] + par[2] + g_add)
      score0 <- mean((1 - m) * (p0 - fitted0))
      score1 <- mean(m * (p1 - fitted1))
      score0^2 + score1^2
    }
    pseudo <- optim(c(0, theta0), score_sq, method = "BFGS",
                    control = list(reltol = 1e-12))$par
    v0 <- pseudo[1] + g_add
    v1 <- pseudo[1] + pseudo[2] + g_add
    h <- pseudo[1] - lambda * interaction

    delta0 <- psi_logit(v0, eta0) - eta0
    delta1 <- psi_logit(v1, eta1) - eta1
    total_bias <- mean(w * (delta1 - delta0)) / J

    # Pure covariate-direction path: apply the same h in both arms. The
    # difference between total and this path is the treatment-direction
    # component induced by the pseudo-true alpha.
    v1_x <- eta1 + h
    delta1_x <- psi_logit(v1_x, eta1) - eta1
    covariate_leak <- mean(w * (delta1_x - delta0)) / J
    first <- mean(w * h * (c_logit(eta1) - c_logit(eta0))) / J

    cbar0 <- delta0 / h
    cbar1 <- delta1_x / h
    near_zero <- abs(h) < 1e-8
    cbar0[near_zero] <- c_logit(eta0[near_zero])
    cbar1[near_zero] <- c_logit(eta1[near_zero])
    norm_h <- sqrt(mean(w * h^2))
    norm_cdiff <- sqrt(mean(w * (cbar1 - cbar0)^2))
    bound <- norm_h * norm_cdiff / J
    local_difference <- (cbar1 - c_logit(eta1)) -
      (cbar0 - c_logit(eta0))
    local_remainder_bound <- norm_h *
      sqrt(mean(w * local_difference^2)) / J
    local_bound <- abs(first) + local_remainder_bound

    row <- data.frame(
      theta0 = theta0, lambda = lambda, n_pop = n_pop, J = J,
      pseudo_intercept = pseudo[1], pseudo_alpha = pseudo[2],
      treatment_error = pseudo[2] - theta0,
      total_bias_pred = total_bias,
      covariate_leak = covariate_leak,
      treatment_component = total_bias - covariate_leak,
      first_order_leak = first,
      taylor_remainder = covariate_leak - first,
      remainder_over_lambda_sq = abs(covariate_leak - first) / lambda^2,
      weighted_bound = bound,
      local_remainder_bound = local_remainder_bound,
      local_bound = local_bound,
      bound_over_abs_leak = bound / abs(covariate_leak),
      local_bound_over_abs_leak = local_bound / abs(covariate_leak)
    )
    rows[[length(rows) + 1L]] <- row
    cat(sprintf(
      "theta=%.1f lambda=%+.1f alpha*=%.3f total=%+.4f X-leak=%+.4f local-bound=%.4f\n",
      theta0, lambda, pseudo[2], total_bias, covariate_leak, local_bound
    ))
  }
}

out <- do.call(rbind, rows)
out$bound_pass <- abs(out$covariate_leak) <= out$weighted_bound + 1e-10
out$local_bound_pass <- abs(out$covariate_leak) <= out$local_bound + 1e-10

stage_compare <- merge(
  out[out$lambda == 1,
      c("theta0", "total_bias_pred", "covariate_leak",
        "treatment_component")],
  stage7[, c("theta0", "bias", "mc_se_bias")],
  by = "theta0", all.x = TRUE
)
stage_compare$difference <- stage_compare$bias -
  stage_compare$total_bias_pred
stage_compare$tolerance <- pmax(
  3 * stage_compare$mc_se_bias, 0.03 * stage_compare$theta0
)

cat("\nstage-7 interaction comparison at lambda=1:\n")
print(format(stage_compare, digits = 4), row.names = FALSE)

full_fidelity <- n_pop >= 1000000L
stopifnot(all(out$bound_pass), all(out$local_bound_pass))
if (full_fidelity) {
  stopifnot(
    all(abs(stage_compare$difference) <= stage_compare$tolerance),
    all(out$local_bound_over_abs_leak <= 10),
    all(out$remainder_over_lambda_sq < 0.08)
  )
}

cat("PASS: the weighted bound covers the exact leak in every cell and the",
    "lambda=1 prediction matches stage 7\n")
if (!full_fidelity) {
  cat("SMOKE: stage-7 and order assertions require at least one million",
      "population draws\n")
}
dir.create("R/results", showWarnings = FALSE)
write.csv(out, "R/results/theory_covariate_leakage.csv", row.names = FALSE)
cat("wrote R/results/theory_covariate_leakage.csv\n")
