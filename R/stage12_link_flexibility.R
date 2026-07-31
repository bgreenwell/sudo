# Stage 12: does link flexibility give SUDO a genuine advantage?
#
# Stage 11 showed that in the binary logit case SUDO loses to orthogonal
# debiasing: the PL orthogonal score removed essentially all regularisation
# bias (+0.029) while SUDO attenuated only part of it (-0.105). But that
# competitor, like Liu, Zhang and Zhou's logistic PL estimator, BUILDS THE
# LOGIT LINK IN. The paper's own framing says so.
#
# So generate binary data whose latent error is Gumbel-max, making cloglog
# the true link, and leave everything else correctly specified. Only the link
# is wrong for a logit-based method. That isolates link flexibility from
# every other source of error.
#
# Arms:
#   sudo_cloglog   SUDO, cloglog imputation model and cloglog completion   [correct]
#   sudo_logit     SUDO, logit throughout                                  [wrong link]
#   plos_cloglog   orthogonal PL score derived for cloglog                 [correct]
#   plos_logit     orthogonal PL score for logit (the liu2021lplr analogue) [wrong link]
#   glm_cloglog    direct coefficient from the correctly specified GLM     [correct]
#   glm_logit      direct coefficient from the logit GLM                   [wrong link]
#
# The cloglog orthogonal score is included deliberately. A referee will say
# the Liu et al. construction generalises to other links by the same route,
# and they are right. Including it means this stage measures what link
# flexibility is actually worth rather than beating a competitor that was
# never allowed to use the correct link. If plos_cloglog matches
# sudo_cloglog, the honest claim is that SUDO supplies arbitrary links
# through one interface, not that no alternative exists.
#
# The general orthogonal score, for link F with density f and index
# eta_d = g(X) + theta*d:
#   w_d = f(eta_d)^2 / [F(eta_d){1 - F(eta_d)}]
#   a   = m*w_1 / {m*w_1 + (1-m)*w_0}
#   score(theta) = sum (D - a) * f(eta_D) {Y - F(eta_D)} / [F(eta_D){1-F(eta_D)}]
# For the logit this reduces to codex's form, since f = p(1-p) there.
#
# Everything except the link is correctly specified: g_0 is linear and the
# GLMs include exactly the right terms. Regularisation plays no role here,
# unlike stage 11, so the two stages isolate different failure modes.
#
# Run from the repository root:
#   Rscript R/stage12_link_flexibility.R
#
# Smoke test:
#   SUDO_STAGE12_REPS=6 SUDO_STAGE12_N=600 Rscript R/stage12_link_flexibility.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))

env_int <- function(name, default) {
  v <- Sys.getenv(name, unset = "")
  if (nzchar(v)) as.integer(v) else as.integer(default)
}
N_OBS  <- env_int("SUDO_STAGE12_N", 2000)
N_REPS <- env_int("SUDO_STAGE12_REPS", 300)
B_DRAW <- env_int("SUDO_STAGE12_B", 25)
THETA0 <- 1.0
PROGRESS <- Sys.getenv("SUDO_PROGRESS", unset = "")

log_progress <- function(msg) {
  if (nzchar(PROGRESS)) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg),
        file = PROGRESS, append = TRUE)
  }
}

# ---- link machinery ---------------------------------------------------------
# cloglog: F(x) = 1 - exp(-exp(x)), f(x) = exp(x - exp(x)).
# Gumbel-max latent error gives exactly this (see AGENTS.md: binary glm
# cloglog is Gumbel-max; ordinal clm cloglog is Gumbel-min).
# Each link supplies p, the score weight f/{p(1-p)}, and the variance weight
# f^2/{p(1-p)}, computed in stable closed form rather than as a ratio of
# quantities that both underflow.
#
# For the logit, f = p(1-p), so the score weight is exactly 1.
# For cloglog with u = exp(eta): p = -expm1(-u), 1-p = exp(-u), f = u exp(-u),
# so the score weight is u / (1 - e^{-u}) and the variance weight is
# u^2 e^{-u} / (1 - e^{-u}). Note the score weight grows like exp(eta): the
# cloglog orthogonal score has an UNBOUNDED weight in the upper tail. That is
# a genuine property of the link, not an implementation artifact, and it is
# why this estimator is numerically delicate where the logit one is not.
LINKS <- list(
  logit = list(
    p = function(x) stats::plogis(x),
    score_w = function(x) rep(1, length(x)),
    var_w = function(x) { p <- stats::plogis(x); p * (1 - p) },
    surrogate_link = "logit"),
  cloglog = list(
    p = function(x) -expm1(-exp(pmin(x, 30))),
    score_w = function(x) {
      u <- exp(pmin(x, 30)); u / pmax(-expm1(-u), 1e-300)
    },
    var_w = function(x) {
      u <- exp(pmin(x, 30)); u^2 * exp(-u) / pmax(-expm1(-u), 1e-300)
    },
    surrogate_link = "cloglog")
)

dgp12 <- function(n) {
  X <- matrix(rnorm(n * 3), n, 3, dimnames = list(NULL, c("X1", "X2", "X3")))
  m0 <- stats::plogis(0.8 * X[, 1] + 0.5 * X[, 2])
  D <- rbinom(n, 1, m0)
  g0 <- X[, 1] + 0.5 * X[, 2] - 0.5 * X[, 3]
  gumbel_max <- -log(-log(runif(n)))
  list(X = X, D = D, Y = as.integer(THETA0 * D + g0 + gumbel_max > 0), m0 = m0)
}

# orthogonal partially-linear score for an arbitrary link
plos_estimate <- function(link, g_hat, D, Y, m_hat) {
  L <- LINKS[[link]]
  score <- function(th) {
    e1 <- g_hat + th; e0 <- g_hat
    w1 <- L$var_w(e1); w0 <- L$var_w(e0)
    a <- m_hat * w1 / pmax(m_hat * w1 + (1 - m_hat) * w0, 1e-300)
    eD <- ifelse(D == 1L, e1, e0)
    sum((D - a) * L$score_w(eD) * (Y - L$p(eD)))
  }
  out <- tryCatch(stats::uniroot(score, c(-4, 6))$root, error = function(e)
    tryCatch(stats::uniroot(score, c(-12, 12))$root,
             error = function(e2) NA_real_))
  out
}

one_rep <- function(seed, n, B) {
  set.seed(seed)
  d <- dgp12(n)
  X <- as.data.frame(d$X)
  dat <- data.frame(Y = d$Y, D = d$D, X)
  folds <- make_folds(n, 5L)
  m_hat <- crossfit(X, d$D, fit_gam_binomial, folds)
  d_res <- d$D - m_hat

  per_link <- function(link) {
    fam <- stats::binomial(link = if (link == "logit") "logit" else "cloglog")
    fit <- stats::glm(Y ~ D + X1 + X2 + X3, family = fam, data = dat)
    beta <- coef(fit)
    direct <- unname(beta["D"])
    mm <- model.matrix(fit)
    v_obs <- as.numeric(mm %*% beta)
    g_hat <- v_obs - direct * d$D          # index at D = 0
    vc <- vcov(fit)

    # SUDO: proper parameter draws, completion on the assumed link,
    # outcome nuisance fitted once from a plug-in completion (validated
    # binary default; stage 3b showed per-draw refitting changes nothing)
    S_plug <- complete_surrogate(d$Y + 1L, v_obs, c(-Inf, 0, Inf),
                                 LINKS[[link]]$surrogate_link)
    ell <- crossfit(X, S_plug, fit_gam, folds)
    thetas <- vapply(seq_len(B), function(b) {
      bd <- MASS::mvrnorm(1, beta, vc)
      vb <- as.numeric(mm %*% bd)
      Sb <- complete_surrogate(d$Y + 1L, vb, c(-Inf, 0, Inf),
                               LINKS[[link]]$surrogate_link)
      fwl_theta(Sb - ell, d_res)$theta
    }, numeric(1))

    c(sudo = mean(thetas), direct = direct,
      plos = plos_estimate(link, g_hat, d$D, d$Y, m_hat))
  }

  cl <- per_link("cloglog"); lg <- per_link("logit")
  data.frame(
    sudo_cloglog = cl["sudo"], plos_cloglog = cl["plos"],
    glm_cloglog  = cl["direct"],
    sudo_logit   = lg["sudo"], plos_logit = lg["plos"],
    glm_logit    = lg["direct"], row.names = NULL)
}

cl <- mc_cluster(c("dgp12", "one_rep", "plos_estimate",
                   "LINKS", "THETA0", "log_progress", "PROGRESS"))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages({
  library(MASS)
})))
# N_REPS is referenced inside the worker closure via log_progress(). It is
# only forced when progress logging is on, so omitting it here fails at run
# time but not in a smoke test with logging disabled.
parallel::clusterExport(cl, c("N_OBS", "B_DRAW", "N_REPS"),
                        envir = environment())

cat(sprintf("Stage 12: link flexibility, n=%d, %d reps, B=%d, theta0=%.1f\n",
            N_OBS, N_REPS, B_DRAW, THETA0))
cat("Truth is cloglog (Gumbel-max latent). Only the link is misspecified.\n\n")

res <- do.call(rbind, parallel::parLapplyLB(cl, seq_len(N_REPS), function(r) {
  out <- one_rep(12000 + r, N_OBS, B_DRAW)
  log_progress(sprintf("rep %d/%d", r, N_REPS))
  out
}))
parallel::stopCluster(cl)

EST <- c("sudo_cloglog", "plos_cloglog", "glm_cloglog",
         "sudo_logit", "plos_logit", "glm_logit")
summ <- do.call(rbind, lapply(EST, function(e) {
  x <- res[[e]]; ok <- is.finite(x)
  data.frame(estimator = e,
             link = if (grepl("cloglog", e)) "cloglog (correct)" else "logit (WRONG)",
             n_reps = sum(ok), fail_rate = mean(!ok),
             bias = mean(x[ok]) - THETA0,
             mc_se_bias = stats::sd(x[ok]) / sqrt(sum(ok)),
             sd = stats::sd(x[ok]),
             rmse = sqrt(mean((x[ok] - THETA0)^2)),
             std_bias = abs(mean(x[ok]) - THETA0) / stats::sd(x[ok]))
}))

cat(sprintf("%-13s %-18s %9s %8s %8s %9s %8s\n",
            "estimator", "assumed link", "bias", "mc_se", "sd", "rmse", "|b|/sd"))
for (i in seq_len(nrow(summ))) {
  cat(sprintf("%-13s %-18s %+9.4f %8.4f %8.4f %9.4f %8.3f\n",
              summ$estimator[i], summ$link[i], summ$bias[i],
              summ$mc_se_bias[i], summ$sd[i], summ$rmse[i], summ$std_bias[i]))
}

# Acceptance: computational invariants plus the design's own premise, namely
# that assuming the wrong link really does bias a logit-based method here.
# Nothing asserts that SUDO wins; the correct-link arms are compared openly.
wrong <- summ[summ$estimator == "plos_logit", ]
gate_ok <- abs(wrong$bias) >= 0.05 && abs(wrong$bias) > 4 * wrong$mc_se_bias
stopifnot(
  nrow(summ) == length(EST),
  all(is.finite(summ$bias)), all(summ$sd > 0),
  max(summ$fail_rate) < 0.02
)
cat(sprintf("\nPASS: all arms finite, failure rate %.1f%%\n",
            100 * max(summ$fail_rate)))
cat(sprintf("LINK-MISSPECIFICATION GATE (logit PL-OS bias >= 0.05): %s (%+.4f)\n",
            if (gate_ok) "PASSED" else "FAILED - wrong link did not bite",
            wrong$bias))

dir.create("R/results", showWarnings = FALSE)
write.csv(summ, "R/results/stage12_link_flexibility.csv", row.names = FALSE)
write.csv(res, "R/results/stage12_link_replications.csv", row.names = FALSE)
cat("wrote R/results/stage12_link_flexibility.csv\n")
