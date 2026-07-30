# Theory check: projected influence expansion for the deterministic
# partially linear logistic-series imputation learner.
#
# The fitted fold-specific index is
#
#   v_hat[-k](D, X) = alpha_hat[-k] D + P_q(X)' gamma_hat[-k],
#
# where P_q is the additive normalized-Hermite dictionary constructed before
# seeing Y. For the population logistic-series coefficient beta_q, define
#
#   H(beta) = J^-1 E[R {psi(z'beta; v0) - v0}].
#
# With A = E[p_q(1-p_q)zz'] and a = partial_beta H(beta_q), the proposed
# learner contribution is
#
#   rho_q(W) = a' A^-1 z {Y - p_q(z)}.
#
# This stage checks that fold averaging has the expansion
#
#   sqrt(n) {K^-1 sum_k H(beta_hat[-k]) - H(beta_q)}
#     = n^-1/2 sum_i rho_q(W_i) + o_p(1),
#
# and checks the corresponding Efron-resampled expansion for fixed and
# redrawn regular folds.
#
# Full defaults:
#   expansion: n in {1000, 2000, 4000}, 100 replications, theta in {1.5, 3}
#   bootstrap: n=2000, 30 datasets, 50 resamples per dataset
#
# Smoke check:
#   SUDO_SERIES_NS=500,1000 SUDO_SERIES_REPS=3 \
#   SUDO_SERIES_BOOT_DATASETS=2 SUDO_SERIES_BOOT_B=3 \
#   Rscript R/theory_pl_series.R
#
# Acceptance at full fidelity:
#   - all series fits and influence quantities are finite;
#   - the empirical/influence SD ratio is in [0.85, 1.15] at n >= 2000;
#   - the normalized expansion remainder at n=4000 is below 0.35 and no more
#     than half its value at n=1000;
#   - fixed- and redrawn-fold bootstrap SD ratios to the weighted influence
#     expansion are in [0.85, 1.15];
#   - the series approximation shifts theta by no more than 3% of theta.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/mc.R")
source("R/sudo/pl.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}

env_ints <- function(name, default) {
  value <- Sys.getenv(name, "")
  if (!nzchar(value)) return(as.integer(default))
  as.integer(strsplit(value, ",", fixed = TRUE)[[1]])
}

psi_logit_series <- function(v, e) {
  p_v <- pmin(pmax(plogis(v), 1e-10), 1 - 1e-10)
  q_v <- 1 - p_v
  entropy <- -(p_v * log(p_v) + q_v * log(q_v))
  mu_plus <- entropy / p_v
  mu_minus <- -entropy / q_v
  p_e <- plogis(e)
  v + p_e * mu_plus + (1 - p_e) * mu_minus
}

dpsi_series <- function(v, e) {
  h <- 1e-5
  (psi_logit_series(v + h, e) - psi_logit_series(v - h, e)) / (2 * h)
}

series_design <- function(X, D, degree) {
  cbind(`(Intercept)` = 1, D = D,
        hermite_series_basis(X, degree = degree))
}

dgp_pl_series <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  propensity <- plogis(X[, 4] + cos(X[, 5]))
  D <- rbinom(n, 1, propensity)
  eta <- theta * D + X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  Y <- rbinom(n, 1, plogis(eta))
  list(X = X, D = D, Y = Y, eta = eta, propensity = propensity)
}

fit_series_beta <- function(d, train_idx, degree) {
  z <- series_design(d$X, d$D, degree)
  fit <- suppressWarnings(glm.fit(
    x = z[train_idx, , drop = FALSE],
    y = d$Y[train_idx],
    family = binomial()
  ))
  beta <- fit$coefficients
  no_missing <- !anyNA(beta)
  beta[is.na(beta)] <- 0
  list(
    beta = beta,
    converged = isTRUE(fit$converged) && no_missing && all(is.finite(beta))
  )
}

tensor_normal_grid <- function(order = 7L, p = 5L) {
  rule <- statmod::gauss.quad.prob(order, dist = "normal")
  index <- do.call(expand.grid, rep(list(seq_len(order)), p))
  X <- vapply(index, function(j) rule$nodes[j], numeric(nrow(index)))
  X <- matrix(X, nrow = nrow(index), ncol = p)
  colnames(X) <- paste0("X", seq_len(p))
  weight <- apply(
    vapply(index, function(j) rule$weights[j], numeric(nrow(index))),
    1, prod
  )
  list(X = X, weight = weight)
}

make_series_reference <- function(theta, degree = 5L, quadrature_order = 7L) {
  grid <- tensor_normal_grid(quadrature_order, 5L)
  X <- grid$X
  propensity <- plogis(X[, 4] + cos(X[, 5]))
  X2 <- rbind(X, X)
  D <- c(rep(0, nrow(X)), rep(1, nrow(X)))
  joint_weight <- c(grid$weight * (1 - propensity),
                    grid$weight * propensity)
  eta <- theta * D + X2[, 1]^2 + sin(X2[, 2]) + 0.5 * X2[, 3]
  true_prob <- plogis(eta)
  propensity2 <- rep(propensity, 2)
  R <- D - propensity2
  z <- series_design(X2, D, degree)

  beta <- rep(0, ncol(z))
  for (iteration in seq_len(100L)) {
    fitted_prob <- pmin(pmax(plogis(as.numeric(z %*% beta)), 1e-8), 1 - 1e-8)
    score <- colSums(z * (joint_weight * (true_prob - fitted_prob)))
    information <- crossprod(
      z, z * (joint_weight * fitted_prob * (1 - fitted_prob))
    )
    step <- solve(information, score)
    beta <- beta + step
    if (max(abs(step)) < 1e-12) break
  }
  stopifnot(iteration < 100L, all(is.finite(beta)))

  fitted_prob <- plogis(as.numeric(z %*% beta))
  information <- crossprod(
    z, z * (joint_weight * fitted_prob * (1 - fitted_prob))
  )
  J <- sum(joint_weight * R^2)
  index_q <- as.numeric(z %*% beta)
  derivative <- dpsi_series(index_q, eta)
  gradient <- colSums(z * (joint_weight * R * derivative)) / J
  direction <- solve(information, gradient)

  H <- function(candidate) {
    index <- as.numeric(z %*% candidate)
    sum(joint_weight * R * (psi_logit_series(index, eta) - eta)) / J
  }

  list(
    beta = beta, information = information, gradient = gradient,
    direction = direction, H = H, H_beta = H(beta), J = J,
    quadrature_mass = sum(joint_weight)
  )
}

series_rho <- function(d, reference, degree) {
  z <- series_design(d$X, d$D, degree)
  fitted_prob <- plogis(as.numeric(z %*% reference$beta))
  as.numeric(z %*% reference$direction) * (d$Y - fitted_prob)
}

fold_projected_functional <- function(d, folds, reference, degree) {
  values <- numeric(length(folds))
  converged <- logical(length(folds))
  for (k in seq_along(folds)) {
    fit <- fit_series_beta(d, setdiff(seq_len(length(d$Y)), folds[[k]]),
                           degree)
    values[k] <- reference$H(fit$beta)
    converged[k] <- fit$converged
  }
  list(value = mean(values), converged = all(converged))
}

one_series_rep <- function(seed, n, theta, degree, reference) {
  set.seed(seed)
  d <- dgp_pl_series(n, theta)
  folds <- make_folds(n, 5L)
  actual_fit <- fold_projected_functional(d, folds, reference, degree)
  rho <- series_rho(d, reference, degree)
  actual <- sqrt(n) * (actual_fit$value - reference$H_beta)
  predicted <- sum(rho) / sqrt(n)
  c(
    actual = actual,
    predicted = predicted,
    remainder = actual - predicted,
    converged = as.numeric(actual_fit$converged),
    all_finite = as.numeric(all(is.finite(c(actual, predicted, rho))))
  )
}

one_series_boot_dataset <- function(seed, n, theta, degree, reference,
                                    B_boot) {
  set.seed(seed)
  d <- dgp_pl_series(n, theta)
  fixed_folds <- make_folds(n, 5L)
  original <- fold_projected_functional(d, fixed_folds, reference, degree)
  rho <- series_rho(d, reference, degree)
  out <- matrix(NA_real_, nrow = B_boot, ncol = 3L)
  colnames(out) <- c("predicted", "fixed", "redrawn")
  for (b in seq_len(B_boot)) {
    index <- sample.int(n, replace = TRUE)
    counts <- tabulate(index, nbins = n)
    d_star <- list(
      X = d$X[index, , drop = FALSE],
      D = d$D[index],
      Y = d$Y[index],
      eta = d$eta[index],
      propensity = d$propensity[index]
    )
    fixed <- fold_projected_functional(
      d_star, fixed_folds, reference, degree)
    redrawn <- fold_projected_functional(
      d_star, make_folds(n, 5L), reference, degree)
    out[b, ] <- c(
      sum((counts - 1) * rho) / sqrt(n),
      sqrt(n) * (fixed$value - original$value),
      sqrt(n) * (redrawn$value - original$value)
    )
    if (!fixed$converged || !redrawn$converged) {
      out[b, ] <- NA_real_
    }
  }
  out
}

degree <- env_int("SUDO_SERIES_DEGREE", 3L)
ns <- env_ints("SUDO_SERIES_NS", c(1000L, 2000L, 4000L))
n_reps <- env_int("SUDO_SERIES_REPS", 100L)
boot_n <- env_int("SUDO_SERIES_BOOT_N", 2000L)
boot_datasets <- env_int("SUDO_SERIES_BOOT_DATASETS", 30L)
boot_B <- env_int("SUDO_SERIES_BOOT_B", 50L)

references <- lapply(c(1.5, 3), make_series_reference, degree = degree)
names(references) <- c("1.5", "3")
stopifnot(all(vapply(references, function(x) {
  abs(x$quadrature_mass - 1) < 1e-10 && all(is.finite(x$direction))
}, logical(1))))

derivative_grid <- seq(-5, 5, length.out = 41L)
stopifnot(max(abs(
  dpsi_series(derivative_grid, derivative_grid) -
    pass_c(derivative_grid)
)) < 2e-5)

rows <- list()
raw <- list()
for (theta in c(1.5, 3)) {
  reference <- references[[as.character(theta)]]
  for (n in ns) {
    values <- vapply(seq_len(n_reps), function(r) {
      one_series_rep(210000 + r, n, theta, degree, reference)
    }, numeric(5))
    actual_sd <- sd(values["actual", ])
    predicted_sd <- sd(values["predicted", ])
    rms_remainder <- sqrt(mean(values["remainder", ]^2))
    row <- data.frame(
      theta = theta, n = n, degree = degree, n_reps = n_reps,
      series_target_shift = reference$H_beta,
      actual_mean = mean(values["actual", ]),
      predicted_mean = mean(values["predicted", ]),
      actual_sd = actual_sd, predicted_sd = predicted_sd,
      sd_ratio = actual_sd / predicted_sd,
      rms_remainder = rms_remainder,
      normalized_remainder = rms_remainder / predicted_sd,
      all_converged = all(values["converged", ] == 1),
      all_finite = all(values["all_finite", ] == 1)
    )
    rows[[length(rows) + 1L]] <- row
    raw[[paste(theta, n, sep = "_")]] <- values
    cat(sprintf(
      "theta=%.1f n=%d shift=%+.4f sd/if=%.3f normalized remainder=%.3f\n",
      theta, n, row$series_target_shift, row$sd_ratio,
      row$normalized_remainder
    ))
  }
}
expansion <- do.call(rbind, rows)

boot_rows <- list()
for (theta in c(1.5, 3)) {
  reference <- references[[as.character(theta)]]
  values <- lapply(seq_len(boot_datasets), function(r) {
    one_series_boot_dataset(
      310000 + r, boot_n, theta, degree, reference, boot_B)
  })
  values <- do.call(rbind, values)
  values <- values[stats::complete.cases(values), , drop = FALSE]
  predicted_sd <- sd(values[, "predicted"])
  fixed_sd <- sd(values[, "fixed"])
  redrawn_sd <- sd(values[, "redrawn"])
  boot_rows[[length(boot_rows) + 1L]] <- data.frame(
    theta = theta, n = boot_n, degree = degree,
    n_datasets = boot_datasets, B_boot = boot_B,
    n_complete = nrow(values),
    fixed_sd_ratio = fixed_sd / predicted_sd,
    redrawn_sd_ratio = redrawn_sd / predicted_sd,
    fixed_normalized_remainder =
      sqrt(mean((values[, "fixed"] - values[, "predicted"])^2)) /
      predicted_sd,
    redrawn_normalized_remainder =
      sqrt(mean((values[, "redrawn"] - values[, "predicted"])^2)) /
      predicted_sd
  )
}
bootstrap <- do.call(rbind, boot_rows)
print(bootstrap, row.names = FALSE)

full_fidelity <- all(c(1000L, 2000L, 4000L) %in% ns) &&
  n_reps >= 100L && boot_n >= 2000L && boot_datasets >= 30L && boot_B >= 50L

stopifnot(
  all(expansion$all_converged),
  all(expansion$all_finite)
)

if (full_fidelity) {
  large <- expansion$n >= 2000L
  stopifnot(
    all(expansion$sd_ratio[large] >= 0.85),
    all(expansion$sd_ratio[large] <= 1.15),
    all(abs(expansion$series_target_shift) <= 0.03 * expansion$theta),
    all(bootstrap$fixed_sd_ratio >= 0.85),
    all(bootstrap$fixed_sd_ratio <= 1.15),
    all(bootstrap$redrawn_sd_ratio >= 0.85),
    all(bootstrap$redrawn_sd_ratio <= 1.15)
  )
  for (theta in c(1.5, 3)) {
    first <- expansion$normalized_remainder[
      expansion$theta == theta & expansion$n == 1000L]
    last <- expansion$normalized_remainder[
      expansion$theta == theta & expansion$n == 4000L]
    stopifnot(last < 0.35, last <= 0.5 * first)
  }
  cat("PASS: series projected influence and bootstrap expansions meet all",
      "full-fidelity criteria\n")
} else {
  cat("SMOKE: statistical acceptance requires the full default grid\n")
}

dir.create("R/results", showWarnings = FALSE)
write.csv(expansion, "R/results/theory_pl_series_expansion.csv",
          row.names = FALSE)
write.csv(bootstrap, "R/results/theory_pl_series_bootstrap.csv",
          row.names = FALSE)
cat("wrote R/results/theory_pl_series_{expansion,bootstrap}.csv\n")
