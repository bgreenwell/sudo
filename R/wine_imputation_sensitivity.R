# Imputation-model sensitivity for the wine application.
#
# The paper argues that SUDO's distinctive weak point is the imputation model:
# unlike the two adjustment nuisances it is not protected by orthogonality, and
# its treatment-direction error reaches theta linearly, damped by the
# pass-through constant cbar_1. The wine analysis reports a Cinelli-Hazlett
# robustness value for unmeasured confounding but nothing for this channel,
# which is the larger and more method-specific risk. This supplies it.
#
# The question. How wrong would the cumulative-link model have to be about
# volatile acidity's coefficient for the estimated effect to be zero? By
# Proposition A3 the bias is cbar_1 * delta_D, so
#
#     delta_D(null) = -theta_hat / cbar_1,
#
# reported both in absolute terms and as a multiple of the coefficient the
# model actually fits.
#
# Two details matter. The treatment is CONTINUOUS, so the binary-D multiplier
# E[w c] / E[w] does not apply; the continuous form is
#     cbar_1 = E[c(eta) D R] / E[R^2],   R = D - E[D|X].
# And the outcome is ORDINAL with 6 (red) or 7 (white) levels, so c is the
# general-J pass-through of Proposition A1, not the binary two-sided form.
#
# For a logistic error law the truncated means are closed-form: substituting
# u = F(x) gives the partial mean int_{-Inf}^t x f(x) dx = -H(F(t)), so every
# mu_j is a difference of entropies and c is exact and vectorised. That
# identity is checked against quadrature below before it is used.
#
# Run from repo root: Rscript R/wine_imputation_sensitivity.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
suppressPackageStartupMessages({ library(ordinal); library(mgcv) })

DATA <- "manuscript/data/wine"

# partial mean of the standard logistic up to t: -H(F(t)), with x log x -> 0
# handled at both tails so t = -Inf and t = +Inf both return 0
G_logis <- function(t) {
  q <- plogis(t)
  out <- numeric(length(t))
  ok <- q > 0 & q < 1
  out[ok] <- q[ok] * log(q[ok]) + (1 - q[ok]) * log1p(-q[ok])
  out
}

# E[xi | a < xi <= b] for the standard logistic
mu_trunc_logis <- function(a, b) {
  (G_logis(b) - G_logis(a)) / (plogis(b) - plogis(a))
}

# Proposition A1 pass-through at index e with interior thresholds kappa,
# vectorised over e
c_ord_logis <- function(e, kappa) {
  J <- length(kappa) + 1L
  lo <- cbind(-Inf, outer(rep(1, length(e)), kappa) - e)
  hi <- cbind(outer(rep(1, length(e)), kappa) - e, Inf)
  mu <- vapply(seq_len(J), function(j) mu_trunc_logis(lo[, j], hi[, j]),
               numeric(length(e)))
  if (length(e) == 1L) mu <- matrix(mu, nrow = 1L)
  dmu <- mu[, -1, drop = FALSE] - mu[, -J, drop = FALSE]
  1 - rowSums(dlogis(outer(rep(1, length(e)), kappa) - e) * dmu)
}

# check the closed form against quadrature before trusting it
chk <- local({
  kappa <- c(-0.8, 0.1, 1.4)
  e <- c(-1.2, 0, 0.9)
  quad <- vapply(e, function(ee) {
    lo <- c(-Inf, kappa) - ee; hi <- c(kappa, Inf) - ee
    mu <- vapply(seq_along(lo), function(j)
      integrate(function(x) x * dlogis(x), lo[j], hi[j],
                rel.tol = 1e-11)$value / (plogis(hi[j]) - plogis(lo[j])),
      numeric(1))
    1 - sum(dlogis(kappa - ee) * diff(mu))
  }, numeric(1))
  max(abs(quad - c_ord_logis(e, kappa)))
})
cat(sprintf("closed-form vs quadrature pass-through: max abs diff %.2e\n\n", chk))
stopifnot(chk < 1e-8)

analyse <- function(which, n_folds = 5, seed = 1) {
  f <- file.path(DATA, sprintf("winequality-%s.csv", which))
  d <- read.csv(f, sep = ";")
  covs <- setdiff(names(d), c("quality", "volatile.acidity"))
  y <- as.integer(factor(d$quality))
  D <- as.numeric(scale(d$volatile.acidity))
  X <- as.data.frame(scale(d[covs]))
  n <- nrow(X)

  set.seed(seed)
  folds <- make_folds(n, n_folds)
  R <- D - crossfit(X, D, fit_gam, folds)        # continuous D, Gaussian gam

  fit <- clm(as.formula(paste("Y ~ D +", paste(names(X), collapse = " + "))),
             data = data.frame(Y = factor(y), D = D, X))
  kappa <- unname(fit$alpha)
  beta_D <- unname(fit$beta["D"])
  mm <- model.matrix(fit)$X[, names(fit$beta), drop = FALSE]
  eta <- as.numeric(mm %*% fit$beta)

  cc <- c_ord_logis(eta, kappa)
  cbar1 <- sum(cc * D * R) / sum(R^2)            # continuous-D multiplier

  res <- read.csv("R/results/wine_application.csv")
  theta <- res$sudo_theta[res$wine == which & res$full_model == "linear"]

  data.frame(wine = which, n = n, J = length(kappa) + 1L,
             theta_hat = theta, clm_beta_D = beta_D,
             c_min = min(cc), c_mean = mean(cc), c_max = max(cc),
             cbar1 = cbar1,
             delta_null = -theta / cbar1,
             multiple_of_beta = abs(-theta / cbar1) / abs(beta_D))
}

out <- do.call(rbind, lapply(c("red", "white"), analyse))
print(format(out, digits = 3), row.names = FALSE)

cat("\nReading:\n")
for (i in seq_len(nrow(out))) {
  r <- out[i, ]
  cat(sprintf(paste0("  %-5s theta_hat %.3f, pass-through cbar_1 %.3f.\n",
                     "        The clm's volatile-acidity coefficient (%.3f) would have to be\n",
                     "        wrong by %+.3f, i.e. %.1f times its fitted value, to null the effect.\n"),
              r$wine, r$theta_hat, r$cbar1, r$clm_beta_D, r$delta_null,
              r$multiple_of_beta))
}

dir.create("R/results", showWarnings = FALSE)
write.csv(out, "R/results/wine_imputation_sensitivity.csv", row.names = FALSE)
cat("\nwrote R/results/wine_imputation_sensitivity.csv\n")
