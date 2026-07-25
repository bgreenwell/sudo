# Rubin's rules pooling for surrogate-draw estimates.
#
# Each surrogate draw b = 1..B yields (theta_b, var_b) from the same PLR fit.
# Treating draws as multiple imputations of the latent utility:
#   theta = mean(theta_b)
#   W     = mean(var_b)                 within-draw variance
#   B_b   = var(theta_b)                between-draw variance
#   T     = W + (1 + 1/B) * B_b         total variance
# CI uses the Barnard-Rubin (1999) small-sample degrees of freedom; with the
# usual B of 10-50 a z-interval understates draw uncertainty.

pool_rubin <- function(theta_b, var_b, level = 0.95, n_obs = Inf, n_par = 1) {
  B <- length(theta_b)
  stopifnot(B >= 2, length(var_b) == B)
  theta <- mean(theta_b)
  W <- mean(var_b)
  B_between <- var(theta_b)
  T_var <- W + (1 + 1 / B) * B_between

  # relative increase in variance due to the missing latent utility
  r <- (1 + 1 / B) * B_between / W
  df_old <- (B - 1) * (1 + 1 / r)^2
  if (is.finite(n_obs)) {
    # Barnard-Rubin adjustment for finite samples
    df_com <- n_obs - n_par
    lambda <- (1 + 1 / B) * B_between / T_var
    df_obs <- (df_com + 1) / (df_com + 3) * df_com * (1 - lambda)
    df <- 1 / (1 / df_old + 1 / df_obs)
  } else {
    df <- df_old
  }

  q <- qt(1 - (1 - level) / 2, df)
  se <- sqrt(T_var)
  list(theta = theta, W = W, B_between = B_between, T = T_var, se = se,
       df = df, ci_lo = theta - q * se, ci_hi = theta + q * se, B = B)
}
