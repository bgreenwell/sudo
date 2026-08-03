# Shared design and acceptance helpers for the count validation stages.

count_cells <- function() {
  rbind(
    expand.grid(treatment = "continuous", family = c("poisson", "negbin"),
                theta = log(c(1.5, 2)), stringsAsFactors = FALSE),
    expand.grid(treatment = "binary", family = c("poisson", "negbin"),
                theta = log(2), stringsAsFactors = FALSE)
  )
}

dgp_count <- function(n, theta, family = c("poisson", "negbin"),
                      treatment = c("binary", "continuous")) {
  family <- match.arg(family)
  treatment <- match.arg(treatment)
  X <- matrix(rnorm(n * 5), n, 5,
              dimnames = list(NULL, paste0("X", seq_len(5))))
  assignment <- X[, 4] + cos(X[, 5])
  D <- if (treatment == "binary") rbinom(n, 1, plogis(assignment)) else
    0.7 * assignment + rnorm(n)
  eta <- 0.5 + theta * D + 0.35 * (X[, 1]^2 - 1) +
    0.5 * sin(X[, 2]) + 0.25 * X[, 3]
  mu <- exp(eta)
  Y <- if (family == "poisson") rpois(n, mu) else
    rnbinom(n, mu = mu, size = 2)
  list(X = X, D = D, Y = Y, eta = eta, mu = mu,
       family = family, treatment = treatment, theta = theta)
}

count_cell_id <- function(treatment, family, theta) sprintf(
  "%s_%s_t%s", treatment, family,
  if (abs(theta - log(2)) < 1e-10) "log2" else "log1p5"
)

count_seed <- function(stage, cell, replication) {
  treatment_code <- match(cell$treatment, c("continuous", "binary"))
  family_code <- match(cell$family, c("poisson", "negbin"))
  theta_code <- match(round(cell$theta, 8), round(log(c(1.5, 2)), 8))
  as.integer(stage * 1000000L + treatment_code * 100000L +
               family_code * 10000L + theta_code * 1000L + replication)
}

count_quality <- function(fit, d, family = d$family) {
  width <- pmax(fit$upper - fit$lower, 1e-300)
  oracle_bounds <- .count_cdf_bounds(
    d$Y, d$mu, family, if (family == "poisson") Inf else 2
  )
  null_mu <- rep(mean(d$Y), length(d$Y))
  null_size <- if (family == "poisson") Inf else
    .estimate_count_size(d$Y, null_mu)
  null_bounds <- .count_cdf_bounds(d$Y, null_mu, family, null_size)
  score <- mean(log(width))
  oracle_score <- mean(log(pmax(oracle_bounds$upper - oracle_bounds$lower,
                                1e-300)))
  null_score <- mean(log(pmax(null_bounds$upper - null_bounds$lower, 1e-300)))
  denominator <- oracle_score - null_score
  data.frame(
    normalized_index_rmse = sqrt(mean((fit$index - d$eta)^2)) / sd(d$eta),
    index_correlation = cor(fit$index, d$eta),
    log_score = score, oracle_log_score = oracle_score,
    null_log_score = null_score,
    oracle_log_score_fraction = (score - null_score) / denominator,
    dispersion = fit$diagnostics$dispersion,
    max_contrast_spread = fit$diagnostics$max_contrast_spread
  )
}

count_mc_summary <- function(replications, truth_column = "truth") {
  target_names <- c("rao_blackwell", "randomized", "index_only",
                    "structured_gam", "poisson_one_step", "raw_label")
  rows <- lapply(target_names, function(target) {
    estimate_name <- paste0(target, "_theta")
    if (!estimate_name %in% names(replications)) return(NULL)
    value <- replications[[estimate_name]]
    truth <- replications[[truth_column]]
    ok <- is.finite(value)
    value <- value[ok]
    truth <- truth[ok]
    sd_value <- sd(value)
    se_name <- paste0(target, "_se")
    has_se <- se_name %in% names(replications)
    se_value <- if (has_se) replications[[se_name]][ok] else rep(NA_real_, sum(ok))
    coverage <- if (has_se)
      mean(abs(value - truth) <= qnorm(0.975) * se_value) else NA_real_
    data.frame(
      estimator = target, n_reps = length(value),
      bias = mean(value - truth), mc_se_bias = sd(value - truth) / sqrt(length(value)),
      sd = sd_value, mean_se = if (has_se) mean(se_value) else NA_real_,
      coverage = coverage,
      mc_se_coverage = if (has_se)
        sqrt(coverage * (1 - coverage) / length(value)) else NA_real_,
      fail_rate = mean(!ok)
    )
  })
  do.call(rbind, rows)
}

count_primary_gates <- function(summary) {
  bias_ok <- abs(summary$bias) <=
    pmax(2 * summary$mc_se_bias, 0.03 * abs(summary$truth))
  coverage_ok <- abs(summary$coverage - 0.95) <=
    pmax(2 * summary$mc_se_coverage, 0.04)
  ratio <- summary$sd / summary$mean_se
  ratio_mc_se <- ratio * sqrt(1 / (2 * (summary$n_reps - 1)))
  data.frame(
    bias_ok = bias_ok, coverage_ok = coverage_ok,
    sd_to_se = ratio,
    sd_to_se_ok = abs(ratio - 1) <= 2 * ratio_mc_se,
    pass = bias_ok & coverage_ok &
      abs(ratio - 1) <= 2 * ratio_mc_se
  )
}
