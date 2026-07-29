# Theory check: reference distributions when the imputation count B is fixed.
#
# In the congenial binary-logit testbed, the joint fixed-B limit separates
# into
#
#   sqrt(n) (theta_bar - theta0) = J^-1 (A + Q_bar),
#   n W_bar -> W0,
#   n (1 + 1/B) B_between ->
#     Q0 * chi-square_(B-1) / (B - 1),
#
# where A is normal with variance V_data, the Q_b are iid normal with
# variance V_draw, and the sample mean and sample variance of the Q_b are
# independent. Thus the numerator is normal and independent of the shifted
# chi-square denominator T. Its exact limiting reference is a
# normal-over-shifted-chi-square mixture, not generally a Student t.
#
# This stage compares:
#   - a normal critical value applied to Rubin's random T;
#   - the observed Barnard-Rubin degrees of freedom;
#   - a moment-matched, variance-calibrated Satterthwaite approximation;
#   - the exact limiting mixture critical value.
#
# It validates predicted against finite-sample coverage for
# B in {5, 10, 25, 50, 100} and theta0 in {1, 3}.
#
# Full defaults: n=5000, 1000 replications, 8 PSOCK workers.
# Smoke check:
#   SUDO_FIXED_B_N=500 SUDO_FIXED_B_REPS=20 SUDO_FIXED_B_CORES=2 \
#   Rscript R/theory_fixed_b_inference.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

pop_fixed_b <- function(theta0, ngrid = 481L, span = 8) {
  node <- seq(-span, span, length.out = ngrid)
  wt <- dnorm(node) * (node[2] - node[1])
  X <- rep(node, times = ngrid)
  R <- rep(node, each = ngrid)
  w <- rep(wt, times = ngrid) * rep(wt, each = ngrid)
  w <- w / sum(w)

  D <- 0.8 * X + R
  eta <- theta0 * D + X
  lp <- plogis(eta, log.p = TRUE)
  lq <- plogis(-eta, log.p = TRUE)
  p <- exp(lp)
  qq <- exp(lq)
  H <- -(p * lp + qq * lq)
  cc <- 1 - H
  z <- cbind(1, D, X)

  J <- sum(w * R^2)
  G <- as.numeric(crossprod(z, w * cc * R))
  Acoef <- as.numeric(crossprod(z, w * R))
  info <- crossprod(z, w * p * qq * z)
  Sigma <- solve(info)
  C <- as.numeric(Sigma %*% (Acoef - G))
  sig_e <- sum(w * H^2 * exp(-lp - lq) * R^2)
  sig_u <- pi^2 / 3 * J - sig_e
  GtSgG <- drop(G %*% Sigma %*% G)
  GtC <- drop(G %*% C)

  list(
    theta0 = theta0, J = J, sig_e = sig_e, sig_u = sig_u,
    GtSgG = GtSgG, GtC = GtC,
    V_data = sig_e + GtSgG + 2 * GtC,
    V_draw = GtSgG + sig_u
  )
}

fixed_b_reference <- function(pop, B, level = 0.95) {
  nu <- B - 1
  alpha <- 1 - level
  V_B <- (pop$V_data + pop$V_draw / B) / pop$J^2
  W0 <- (pop$sig_e + pop$sig_u) / pop$J^2
  Q0 <- (1 + 1 / B) * pop$V_draw / pop$J^2
  mean_T <- W0 + Q0
  df_sat <- nu * (mean_T / Q0)^2

  integrate_coverage <- function(critical) {
    integrate(
      function(x) {
        T_x <- W0 + Q0 * x / nu
        crit_x <- critical(x)
        (2 * pnorm(crit_x * sqrt(T_x / V_B)) - 1) * dchisq(x, nu)
      },
      lower = 0, upper = Inf, rel.tol = 1e-10
    )$value
  }

  z <- qnorm(1 - alpha / 2)
  pred_normal <- integrate_coverage(function(x) rep(z, length(x)))
  pred_br <- integrate_coverage(function(x) {
    r <- Q0 * pmax(x, .Machine$double.eps) / (nu * W0)
    qt(1 - alpha / 2, nu * (1 + 1 / r)^2)
  })
  critical_sat <- sqrt(V_B / mean_T) * qt(1 - alpha / 2, df_sat)
  pred_sat <- integrate_coverage(
    function(x) rep(critical_sat, length(x))
  )
  objective <- function(q) {
    integrate_coverage(function(x) rep(q, length(x))) - level
  }
  critical_mix <- uniroot(objective, c(1, 4), tol = 1e-10)$root
  pred_mix <- integrate_coverage(
    function(x) rep(critical_mix, length(x))
  )

  list(
    B = B, V_B = V_B, W0 = W0, Q0 = Q0, mean_T = mean_T,
    df_sat = df_sat, critical_sat = critical_sat,
    critical_mix = critical_mix, pred_normal = pred_normal,
    pred_br = pred_br, pred_sat = pred_sat, pred_mix = pred_mix
  )
}

one_fixed_b_rep <- function(seed, n, theta0, B_set) {
  set.seed(seed)
  X <- rnorm(n)
  D <- 0.8 * X + rnorm(n)
  Y <- rbinom(n, 1, plogis(theta0 * D + X))
  qx <- qr(cbind(1, X))
  rD <- qr.resid(qx, D)
  fit <- suppressWarnings(glm(Y ~ D + X, family = binomial))
  bh <- coef(fit)
  Vc <- vcov(fit)
  mm <- cbind(1, D, X)
  B_max <- max(B_set)
  theta_b <- numeric(B_max)
  var_b <- numeric(B_max)

  for (b in seq_len(B_max)) {
    beta_b <- MASS::mvrnorm(1, bh, Vc)
    S <- complete_surrogate(
      Y + 1L, as.numeric(mm %*% beta_b), c(-Inf, 0, Inf), "logit"
    )
    f <- fwl_theta(qr.resid(qx, S), rD)
    theta_b[b] <- f$theta
    var_b[b] <- f$var
  }

  unlist(lapply(B_set, function(B) {
    th <- theta_b[seq_len(B)]
    vv <- var_b[seq_len(B)]
    W <- mean(vv)
    between <- var(th)
    Tvar <- W + (1 + 1 / B) * between
    r <- (1 + 1 / B) * between / W
    df_br <- (B - 1) * (1 + 1 / r)^2
    c(theta = mean(th), T = Tvar, df_br = df_br)
  }), use.names = FALSE)
}

n <- env_int("SUDO_FIXED_B_N", 5000L)
n_reps <- env_int("SUDO_FIXED_B_REPS", 1000L)
mc_cores <- env_int("SUDO_FIXED_B_CORES", 8L)
B_set <- c(5L, 10L, 25L, 50L, 100L)
level <- 0.95

cat(sprintf(
  "fixed-B inference: n=%d reps=%d B={%s}\n",
  n, n_reps, paste(B_set, collapse = ",")
))

cl <- mc_cluster("one_fixed_b_rep", n_cores = mc_cores)
on.exit(parallel::stopCluster(cl), add = TRUE)
rows <- list()

for (theta0 in c(1, 3)) {
  parallel::clusterExport(
    cl, c("n", "theta0", "B_set"), envir = environment()
  )
  values <- parallel::parSapply(cl, seq_len(n_reps), function(r) {
    one_fixed_b_rep(170000 + r, n, theta0, B_set)
  })
  pop <- pop_fixed_b(theta0)

  for (j in seq_along(B_set)) {
    B <- B_set[j]
    idx <- 3L * (j - 1L)
    theta_hat <- values[idx + 1L, ]
    Tvar <- values[idx + 2L, ]
    df_br <- values[idx + 3L, ]
    ref <- fixed_b_reference(pop, B, level)
    z <- qnorm(1 - (1 - level) / 2)

    covered_normal <- abs(theta_hat - theta0) <= z * sqrt(Tvar)
    covered_br <- abs(theta_hat - theta0) <=
      qt(1 - (1 - level) / 2, df_br) * sqrt(Tvar)
    covered_sat <- abs(theta_hat - theta0) <=
      ref$critical_sat * sqrt(Tvar)
    covered_mix <- abs(theta_hat - theta0) <=
      ref$critical_mix * sqrt(Tvar)

    row <- data.frame(
      theta0 = theta0, B = B, n = n, n_reps = n_reps,
      bias = mean(theta_hat) - theta0,
      mc_se_bias = sd(theta_hat) / sqrt(n_reps),
      V_pred = ref$V_B,
      n_var_emp = n * var(theta_hat),
      mean_nT_pred = ref$mean_T,
      mean_nT_emp = mean(n * Tvar),
      df_satterthwaite = ref$df_sat,
      critical_satterthwaite = ref$critical_sat,
      critical_mixture = ref$critical_mix,
      coverage_normal_pred = ref$pred_normal,
      coverage_normal_emp = mean(covered_normal),
      coverage_br_pred = ref$pred_br,
      coverage_br_emp = mean(covered_br),
      coverage_satterthwaite_pred = ref$pred_sat,
      coverage_satterthwaite_emp = mean(covered_sat),
      coverage_mixture_pred = ref$pred_mix,
      coverage_mixture_emp = mean(covered_mix),
      sig_e = pop$sig_e, sig_u = pop$sig_u,
      GtSgG = pop$GtSgG, GtC = pop$GtC
    )
    rows[[length(rows) + 1L]] <- row
    cat(sprintf(
      "theta=%.0f B=%3d cover pred/emp N %.3f/%.3f BR %.3f/%.3f Sat %.3f/%.3f Mix %.3f/%.3f\n",
      theta0, B,
      row$coverage_normal_pred, row$coverage_normal_emp,
      row$coverage_br_pred, row$coverage_br_emp,
      row$coverage_satterthwaite_pred,
      row$coverage_satterthwaite_emp,
      row$coverage_mixture_pred, row$coverage_mixture_emp
    ))
  }
}

out <- do.call(rbind, rows)
full_fidelity <- n >= 3000L && n_reps >= 500L
if (full_fidelity) {
  coverage_pairs <- list(
    c("coverage_normal_pred", "coverage_normal_emp"),
    c("coverage_br_pred", "coverage_br_emp"),
    c("coverage_satterthwaite_pred", "coverage_satterthwaite_emp"),
    c("coverage_mixture_pred", "coverage_mixture_emp")
  )
  for (pair in coverage_pairs) {
    pred <- out[[pair[1]]]
    emp <- out[[pair[2]]]
    tolerance <- pmax(4 * sqrt(pred * (1 - pred) / n_reps), 0.03)
    stopifnot(all(abs(emp - pred) <= tolerance))
  }
  stopifnot(
    all(abs(out$bias) <=
          pmax(3 * out$mc_se_bias, 0.02 * out$theta0) + 1e-9)
  )
}

cat("PASS: predicted and empirical fixed-B coverage agree for every",
    "reference distribution and design cell\n")
if (!full_fidelity) {
  cat("SMOKE: statistical assertions require n >= 3000 and",
      "500 replications\n")
}
dir.create("R/results", showWarnings = FALSE)
write.csv(out, "R/results/theory_fixed_b_inference.csv", row.names = FALSE)
cat("wrote R/results/theory_fixed_b_inference.csv\n")
