# Theory check: the fixed-B variance decomposition and the cross term C
# (supports Theorems A1 and A2 of the paper's asymptotic-theory appendix).
#
# Split the truncated draw noise into a data-measurable part and pure
# imputation Monte-Carlo noise,
#     xi_i^(b) = e_i + u_i^(b),   e_i = E[xi_i | W_i],   E[u_i^(b) | W_i] = 0,
# which matters because e_i is NOT mean-zero given W_i (only after averaging
# over Y | D, X at the truth). With beta^(b) = beta_hat + n^{-1/2} Z_b and
# Z_b | data ~ N(0, Sigma_beta) iid, pooling B draws gives
#
#   V_B = J^{-2} [ sigma_e^2 + G'Sg G + 2 G'C + (G'Sg G + sigma_u^2) / B ]
#
#   sigma_e^2 = E[e^2 R^2],  sigma_u^2 = E[Var(xi | W) R^2],
#   G = E[c(eta_0) z R],     C = E[e R IF_beta],   Sg = Sigma_beta.
#
# So the imputation Monte-Carlo error does NOT vanish at fixed B, and the
# B -> Inf limit is sigma_e^2, not the single-draw sigma_e^2 + sigma_u^2.
#
# Rubin's T = W_bar + (1 + 1/B) B_between targets
#   J^{-2} [ sigma_e^2 + sigma_u^2 + (1 + 1/B) (G'Sg G + sigma_u^2) ],
# whose 1/B terms cancel against V_B's, leaving
#
#   T - V_B  ->  2 J^{-2} ( sigma_u^2 - G'C ).
#
# Rubin's variance is therefore asymptotically EXACT here iff G'C = sigma_u^2,
# conservative iff G'C < sigma_u^2. That identity, not the pooled ratio, is
# what the congenial testbed should be asked about.
#
# Testbed (same DGP as R/theory_rubin_check.R):
#   X ~ N(0,1); D = 0.8X + N(0,1); Y = 1{theta0 D + X + logistic > 0}
# so the imputation model glm(Y ~ D + X) and both partialling nuisances are
# correctly specified, and every population quantity above is closed-form:
#
#   c(e)  = 1 - H(p),                p = plogis(e), H = binary entropy in nats
#   E[e^2 | D, X] = H(p)^2 / (p(1-p))
#   xi | D, X ~ Logistic(0,1) exactly, so E[xi^2 R^2] = (pi^2/3) J
#   E[e R (Y - p) z] = E[H(p) z R] = A - G,  A = E[z R],  so C = Sg (A - G)
#
# Run from repo root: Rscript R/theory_variance_terms.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
suppressPackageStartupMessages(library(parallel))

# ---- population quantities, by product-grid quadrature over (X, R) --------
# X and R = D - 0.8X are independent standard normals, so a tensor grid with
# normal weights integrates everything to machine precision and leaves no
# Monte-Carlo noise on the theory side.

pop_terms <- function(theta0, ngrid = 481L, span = 8) {
  node <- seq(-span, span, length.out = ngrid)
  wt <- dnorm(node) * (node[2] - node[1])
  X <- rep(node, times = ngrid)
  R <- rep(node, each = ngrid)
  w <- rep(wt, times = ngrid) * rep(wt, each = ngrid)
  w <- w / sum(w)                                   # renormalise the tails

  D <- 0.8 * X + R
  eta <- theta0 * D + X
  # log-scale probabilities: at strong signal |eta| reaches ~50 on this grid
  # and plogis() saturates at 0/1, which would take H and E[e^2] to NaN
  lp <- plogis(eta, log.p = TRUE)
  lq <- plogis(-eta, log.p = TRUE)
  p <- exp(lp); qq <- exp(lq)
  H <- -(p * lp + qq * lq)                          # binary entropy, nats
  cc <- 1 - H                                       # pass-through factor
  z <- cbind(1, D, X)

  J <- sum(w * R^2)
  G <- as.numeric(crossprod(z, w * cc * R))
  A <- as.numeric(crossprod(z, w * R))
  I <- crossprod(z, w * p * qq * z)
  Sg <- solve(I)
  C <- as.numeric(Sg %*% (A - G))

  # E[e^2 | D, X] = H^2 / (p q), computed as H^2 exp(-lp - lq) to stay finite
  sig_e <- sum(w * H^2 * exp(-lp - lq) * R^2)
  sig_psi <- pi^2 / 3 * J                           # E[xi^2 R^2], xi ~ logistic
  sig_u <- sig_psi - sig_e

  list(theta0 = theta0, J = J, G = G, A = A, Sg = Sg, C = C,
       GtSgG = drop(G %*% Sg %*% G), GtC = drop(G %*% C),
       sig_e = sig_e, sig_u = sig_u, sig_psi = sig_psi,
       V_B = function(B) (sig_e + drop(G %*% Sg %*% G) + 2 * drop(G %*% C) +
                            (drop(G %*% Sg %*% G) + sig_u) / B) / J^2,
       T_B = function(B) (sig_e + sig_u +
                            (1 + 1 / B) * (drop(G %*% Sg %*% G) + sig_u)) / J^2)
}

# ---- the two closed forms this rests on ----------------------------------
# Everything above reduces to one identity. With mu_+ = E[xi | xi > -eta] and
# mu_- = E[xi | xi <= -eta] under the logistic law,
#
#     p (1 - p) (mu_+ - mu_-) = H(p),      p = plogis(eta),
#
# which gives both closed forms at once:
#   c(eta) = 1 - f(-eta)(mu_+ - mu_-) = 1 - H(p)     (pass-through factor)
#   E[e (Y - p) | D, X] = p(1-p)(mu_+ - mu_-) = H(p) (hence C = Sg (A - G),
#     since E[e R (Y - p) z] = E[H z R] = E[(1 - c) z R] = A - G)
# Check it pointwise by quadrature; no Monte Carlo, so no tail trouble at
# strong signal, where e is heavy-tailed and a sample mean converges slowly.

check_identity <- function(eta_grid = c(-6, -3, -1, -0.3, 0, 0.5, 2, 5, 9)) {
  mu_p <- vapply(eta_grid, function(e)
    integrate(function(x) x * dlogis(x), -e, Inf, rel.tol = 1e-12)$value /
      plogis(e), numeric(1))
  mu_m <- vapply(eta_grid, function(e)
    integrate(function(x) x * dlogis(x), -Inf, -e, rel.tol = 1e-12)$value /
      plogis(-e), numeric(1))
  p <- plogis(eta_grid)
  lp <- plogis(eta_grid, log.p = TRUE); lq <- plogis(-eta_grid, log.p = TRUE)
  H <- -(p * lp + exp(lq) * lq)
  data.frame(eta = eta_grid, lhs = p * (1 - p) * (mu_p - mu_m), H = H,
             err = abs(p * (1 - p) * (mu_p - mu_m) - H))
}

# ---- simulation ----------------------------------------------------------

one_rep <- function(seed, n, theta0, B_max) {
  set.seed(seed)
  X <- rnorm(n)
  D <- 0.8 * X + rnorm(n)
  Y <- rbinom(n, 1, plogis(theta0 * D + X))
  q <- qr(cbind(1, X))
  rD <- qr.resid(q, D)
  fit <- suppressWarnings(glm(Y ~ D + X, family = binomial))
  bh <- coef(fit); Vc <- vcov(fit)
  mm <- cbind(1, D, X)
  th <- numeric(B_max); vr <- numeric(B_max)
  for (b in seq_len(B_max)) {
    beta_b <- MASS::mvrnorm(1, bh, Vc)
    S <- complete_surrogate(Y + 1L, as.numeric(mm %*% beta_b),
                            c(-Inf, 0, Inf), "logit")
    f <- fwl_theta(qr.resid(q, S), rD)
    th[b] <- f$theta; vr[b] <- f$var
  }
  c(th, vr)
}

run_sim <- function(n, theta0, B_set, R_reps, base_seed = 0) {
  B_max <- max(B_set)
  res <- mclapply(seq_len(R_reps),
                  function(r) one_rep(base_seed + r, n, theta0, B_max),
                  mc.cores = max(1, detectCores() - 1))
  M <- do.call(rbind, res)
  th <- M[, seq_len(B_max), drop = FALSE]
  vr <- M[, B_max + seq_len(B_max), drop = FALSE]
  do.call(rbind, lapply(B_set, function(B) {
    bar <- rowMeans(th[, seq_len(B), drop = FALSE])
    v_emp <- var(bar) * n
    if (B >= 2) {
      Tv <- n * (rowMeans(vr[, seq_len(B), drop = FALSE]) +
                   (1 + 1 / B) * apply(th[, seq_len(B), drop = FALSE], 1, var))
      t_emp <- mean(Tv)
      t_se <- sd(Tv) / sqrt(R_reps)
    } else {
      t_emp <- NA_real_; t_se <- NA_real_
    }
    data.frame(B = B, V_emp = v_emp,
               V_emp_se = v_emp * sqrt(2 / (R_reps - 1)),
               T_emp = t_emp, T_emp_se = t_se)
  }))
}

# ---- run -----------------------------------------------------------------

n <- 20000
R_reps <- 4000
B_set <- c(1, 2, 5, 20)

id <- check_identity()
cat("Identity p(1-p)(mu_+ - mu_-) = H(p), the basis of c = 1 - H and C:\n")
print(format(id, digits = 5), row.names = FALSE)
stopifnot(max(id$err) < 1e-9)
cat("PASS: identity holds to", format(max(id$err), digits = 2), "\n")

all_sim <- list()
for (theta0 in c(1, 3)) {
  pt <- pop_terms(theta0)
  cat("\n== Population terms (congenial binary-logit testbed), theta0 =",
      theta0, "==\n")
  cat(sprintf("  J_theta        %.5f\n", pt$J))
  cat(sprintf("  sigma_e^2      %.5f\n", pt$sig_e))
  cat(sprintf("  sigma_u^2      %.5f\n", pt$sig_u))
  cat(sprintf("  sigma_psi^2    %.5f   (= pi^2/3 * J, single draw)\n", pt$sig_psi))
  cat(sprintf("  G              (%s)\n",
              paste(sprintf("%.4f", pt$G), collapse = ", ")))
  cat(sprintf("  G' Sigma G     %.5f\n", pt$GtSgG))
  cat(sprintf("  G' C           %.5f\n", pt$GtC))
  cat(sprintf("  gap  sigma_u^2 - G'C          = %+.5f\n", pt$sig_u - pt$GtC))
  cat(sprintf("  predicted T - V = 2 J^-2 gap  = %+.5f  (%+.2f%% of V at B=20)\n",
              2 * (pt$sig_u - pt$GtC) / pt$J^2,
              100 * 2 * (pt$sig_u - pt$GtC) / pt$J^2 / pt$V_B(20)))

  cat("  simulation: n =", n, " reps =", R_reps, "\n")
  sim <- run_sim(n, theta0, B_set, R_reps)
  sim$V_pred <- vapply(sim$B, pt$V_B, numeric(1))
  sim$T_pred <- vapply(sim$B, pt$T_B, numeric(1))
  sim$V_z <- (sim$V_emp - sim$V_pred) / sim$V_emp_se
  sim$T_z <- (sim$T_emp - sim$T_pred) / sim$T_emp_se
  sim$ratio_pred <- sim$T_pred / sim$V_pred
  print(format(cbind(theta0 = theta0, sim), digits = 4), row.names = FALSE)

  stopifnot(max(abs(sim$V_z)) < 4)
  stopifnot(max(abs(sim$T_z), na.rm = TRUE) < 4)
  all_sim[[length(all_sim) + 1]] <- cbind(
    theta0 = theta0, sim,
    sig_e = pt$sig_e, sig_u = pt$sig_u, GtSgG = pt$GtSgG, GtC = pt$GtC)
}

cat("\nPASS: fixed-B variance formula and the Rubin T decomposition both match",
    "Monte Carlo at every B and both theta0 (|z| < 4)\n")

out <- do.call(rbind, all_sim)
dir.create("R/results", showWarnings = FALSE)
write.csv(out, "R/results/theory_variance_terms.csv", row.names = FALSE)
cat("wrote R/results/theory_variance_terms.csv\n")
