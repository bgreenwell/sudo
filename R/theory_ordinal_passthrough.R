# Theory check: the ordinal pass-through map, its index derivative, and the
# threshold block of G (supports Proposition A1 of the paper's
# asymptotic-theory appendix).
#
# The surrogate draw for category y is S = v + xi with xi ~ F truncated to
# (c_{y-1} - v, c_y - v], matching complete_surrogate() in R/sudo/surrogate.R.
# Averaging over Y | D, X at the true index e and true thresholds c0 gives the
# pass-through map
#
#   psi(v, c) = v + sum_j p_j(e, c0) mu_j(v, c),
#   mu_j(v, c) = E[xi | c_{j-1} - v < xi <= c_j - v],
#   p_j(e, c0) = F(c0_j - e) - F(c0_{j-1} - e).
#
# Claims under test, all evaluated at (v, c) = (e, c0), with t_k = c0_k - e:
#
#   (1) d psi / d v   = 1 - sum_k f(t_k) (mu_{k+1} - mu_k)  =: c_ord(e)
#   (2) d psi / d c_k = f(t_k) (mu_{k+1} - mu_k)
#   (3) c_ord(e) = 1 - sum_k d psi / d c_k
#
# (2) is the threshold block of G, which is why the ordinal full model needs
# ordinal::clm (whose vcov covers the thresholds) and not mgcv::ocat. (3) holds
# because mu_j depends on (v, c) only through c - v. Binary is the J = 2,
# c_1 = 0 case, where (1) collapses to 1 - f(-e) (mu_+ - mu_-); note the
# argument is -e, which coincides with the symmetric-F shorthand f(e) only for
# the logit link, not for either Gumbel.
#
# Run from repo root: Rscript R/theory_ordinal_passthrough.R

source("R/sudo/surrogate.R")

# error laws: cdf and density on the latent scale. Sign conventions follow
# surrogate.R (binary glm cloglog is Gumbel-max, ordinal clm cloglog is
# Gumbel-min).
laws <- list(
  logit       = list(F = function(x) plogis(x),
                     f = function(x) dlogis(x)),
  probit      = list(F = function(x) pnorm(x),
                     f = function(x) dnorm(x)),
  cloglog     = list(F = function(x) exp(-exp(-x)),
                     f = function(x) exp(-x - exp(-x))),
  cloglog_min = list(F = function(x) 1 - exp(-exp(x)),
                     f = function(x) exp(x - exp(x)))
)

# E[xi | a < xi <= b] under the law, by quadrature (no Monte-Carlo noise, so
# the finite-difference comparison below is a sharp test)
mu_trunc <- function(law, a, b) {
  p <- law$F(b) - law$F(a)
  m <- integrate(function(t) t * law$f(t), a, b,
                 rel.tol = 1e-11, subdivisions = 1000L)$value
  m / p
}

mu_all <- function(law, v, cuts) {
  lo <- c(-Inf, cuts) - v
  hi <- c(cuts, Inf) - v
  vapply(seq_along(lo), function(j) mu_trunc(law, lo[j], hi[j]), numeric(1))
}

probs <- function(law, e, cuts0) {
  law$F(c(cuts0, Inf) - e) - law$F(c(-Inf, cuts0) - e)
}

psi_ord <- function(law, v, cuts, e, cuts0) {
  v + sum(probs(law, e, cuts0) * mu_all(law, v, cuts))
}

# analytic index derivative
c_ord <- function(law, e, cuts0) {
  mu <- mu_all(law, e, cuts0)
  1 - sum(law$f(cuts0 - e) * diff(mu))
}

# analytic threshold-derivative kernel, one entry per interior threshold
g_thresh <- function(law, e, cuts0) {
  mu <- mu_all(law, e, cuts0)
  law$f(cuts0 - e) * diff(mu)
}

central_diff <- function(fn, x, h = 1e-4) (fn(x + h) - fn(x - h)) / (2 * h)

# ---- checks ---------------------------------------------------------------

designs <- list(
  list(tag = "binary  J=2", cuts0 = 0),
  list(tag = "ordinal J=3", cuts0 = c(-0.5, 0.8)),
  list(tag = "ordinal J=4", cuts0 = c(-1.0, 0.2, 1.3))
)
e_grid <- c(-1.5, -0.5, 0, 0.7, 2.0)
tol <- 1e-5

cat("Ordinal pass-through: analytic derivatives vs central differences\n")
cat("tolerance", tol, "\n\n")

rows <- list()
for (lk in names(laws)) {
  law <- laws[[lk]]
  for (dz in designs) {
    cuts0 <- dz$cuts0
    for (e in e_grid) {
      # (1) index derivative
      cv <- c_ord(law, e, cuts0)
      cv_fd <- central_diff(function(v) psi_ord(law, v, cuts0, e, cuts0), e)

      # (2) threshold derivatives
      gk <- g_thresh(law, e, cuts0)
      gk_fd <- vapply(seq_along(cuts0), function(k) {
        central_diff(function(ck) {
          cc <- cuts0; cc[k] <- ck
          psi_ord(law, e, cc, e, cuts0)
        }, cuts0[k])
      }, numeric(1))

      rows[[length(rows) + 1]] <- data.frame(
        link = lk, design = dz$tag, e = e,
        c_ord = cv,
        err_index = abs(cv - cv_fd),
        err_thresh = max(abs(gk - gk_fd)),
        # (3) the two blocks are the same kernel
        err_identity = abs(cv - (1 - sum(gk))))
    }
  }
}
out <- do.call(rbind, rows)
print(format(out, digits = 4), row.names = FALSE)

stopifnot(max(out$err_index) < tol,
          max(out$err_thresh) < tol,
          max(out$err_identity) < 1e-10)
cat("\nPASS: index derivative, threshold block, and the c_ord = 1 - sum(G_c)",
    "identity all hold\n")

# binary special case reproduces the two-sided form used in the paper
for (lk in names(laws)) {
  law <- laws[[lk]]
  for (e in e_grid) {
    mu_m <- mu_trunc(law, -Inf, -e)
    mu_p <- mu_trunc(law, -e, Inf)
    stopifnot(abs(c_ord(law, e, 0) - (1 - law$f(-e) * (mu_p - mu_m))) < 1e-10)
  }
}
cat("PASS: J = 2 collapses to 1 - f(-e) (mu_+ - mu_-)\n")

# the quadrature psi matches what complete_surrogate() actually samples, so
# the theory above describes the implemented estimator and not just the model
set.seed(1)
n_mc <- 4e5
cat("\ncomplete_surrogate() vs quadrature psi (MC check)\n")
mc <- list()
for (lk in names(laws)) {
  law <- laws[[lk]]
  cuts0 <- c(-0.5, 0.8)
  e <- 0.4
  v <- 0.55                                  # off the truth, so psi is probed
  p <- probs(law, e, cuts0)
  Y <- sample.int(length(p), n_mc, replace = TRUE, prob = p)
  S <- complete_surrogate(Y, rep(v, n_mc), c(-Inf, cuts0, Inf), lk)
  target <- psi_ord(law, v, cuts0, e, cuts0)
  se <- sd(S) / sqrt(n_mc)
  mc[[lk]] <- data.frame(link = lk, mc_mean = mean(S), psi = target,
                         z = (mean(S) - target) / se)
}
mc <- do.call(rbind, mc)
print(format(mc, digits = 5), row.names = FALSE)
stopifnot(max(abs(mc$z)) < 4)
cat("\nPASS: sampler agrees with the pass-through map (|z| < 4)\n")

dir.create("R/results", showWarnings = FALSE)
write.csv(out, "R/results/theory_ordinal_passthrough.csv", row.names = FALSE)
cat("wrote R/results/theory_ordinal_passthrough.csv\n")
