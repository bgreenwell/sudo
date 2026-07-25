# Monte Carlo harness shared by all stage scripts.
#
# estimator(data) must return a list with at least theta and se; optional
# ci_lo/ci_hi (defaults to theta +- 1.96*se).
# run_mc() returns one row per rep; summarize_mc() the usual MC table with
# Monte Carlo standard errors so acceptance checks are principled.

run_mc <- function(n_reps, dgp, estimator, theta_true, seed = 1) {
  rows <- lapply(seq_len(n_reps), function(r) {
    set.seed(seed + r)
    est <- estimator(dgp())
    ci_lo <- if (!is.null(est$ci_lo)) est$ci_lo else est$theta - 1.96 * est$se
    ci_hi <- if (!is.null(est$ci_hi)) est$ci_hi else est$theta + 1.96 * est$se
    data.frame(rep = r, est = est$theta, se = est$se, ci_lo = ci_lo,
               ci_hi = ci_hi, covered = ci_lo <= theta_true & theta_true <= ci_hi)
  })
  df <- do.call(rbind, rows)
  attr(df, "theta_true") <- theta_true
  df
}

summarize_mc <- function(df, theta_true = attr(df, "theta_true")) {
  n <- nrow(df)
  bias <- mean(df$est) - theta_true
  sdv <- sd(df$est)
  cov <- mean(df$covered)
  data.frame(
    n_reps = n,
    mean_est = mean(df$est),
    bias = bias,
    mc_se_bias = sdv / sqrt(n),
    sd = sdv,
    rmse = sqrt(mean((df$est - theta_true)^2)),
    mean_se = mean(df$se),
    coverage = cov,
    mc_se_cov = sqrt(cov * (1 - cov) / n)
  )
}

# PSOCK parallel MC (mgcv is not fork-safe on macOS). Closures passed as dgp/
# estimator must not have free variables living in globalenv: build them with
# a factory that force()s its arguments so the call frame ships to workers.
mc_cluster <- function(exports = character(), n_cores = NULL) {
  if (is.null(n_cores)) n_cores <- max(1, parallel::detectCores() - 2)
  cl <- parallel::makeCluster(n_cores)
  invisible(parallel::clusterEvalQ(cl, {
    source("R/sudo/fwl.R"); source("R/sudo/surrogate.R")
    source("R/sudo/rubin.R"); source("R/sudo/mc.R")
    suppressPackageStartupMessages(library(mgcv))
  }))
  if (length(exports)) parallel::clusterExport(cl, exports)
  cl
}

run_mc_par <- function(cl, n_reps, dgp, estimator, theta_true, seed = 1) {
  parallel::clusterExport(cl, c("dgp", "estimator", "theta_true", "seed"),
                          envir = environment())
  rows <- parallel::parLapply(cl, seq_len(n_reps), function(r) {
    set.seed(seed + r)
    est <- estimator(dgp())
    ci_lo <- if (!is.null(est$ci_lo)) est$ci_lo else est$theta - 1.96 * est$se
    ci_hi <- if (!is.null(est$ci_hi)) est$ci_hi else est$theta + 1.96 * est$se
    out <- data.frame(rep = r, est = est$theta, se = est$se, ci_lo = ci_lo,
                      ci_hi = ci_hi,
                      covered = ci_lo <= theta_true & theta_true <= ci_hi)
    # carry through any extra scalar diagnostics the estimator returns
    # (W, B_between, df, max_abs_lp, ...) so drivers can decompose coverage
    extra <- setdiff(names(est), c("theta", "se", "ci_lo", "ci_hi"))
    for (nm in extra) {
      v <- est[[nm]]
      if (is.numeric(v) && length(v) == 1L) out[[nm]] <- v
    }
    out
  })
  df <- do.call(rbind, rows)
  attr(df, "theta_true") <- theta_true
  df
}

print_mc <- function(label, summ) {
  cat(sprintf(
    "%-28s bias %+ .4f (mc-se %.4f)  sd %.4f  rmse %.4f  mean-se %.4f  cover %.3f (mc-se %.3f)\n",
    label, summ$bias, summ$mc_se_bias, summ$sd, summ$rmse, summ$mean_se,
    summ$coverage, summ$mc_se_cov))
}
