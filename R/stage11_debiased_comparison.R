# Stage 11: SUDO versus properly debiased alternatives, where debiasing matters.
#
# The question this settles. Stages 8 and 9 compared SUDO against reading the
# treatment coefficient off the imputation model, and found parity. But every
# one of those designs was benign: stage 8 used unpenalised clm, and stage 9's
# mgcv arm left D as an unpenalised parametric term. Neither exhibits the
# regularisation bias that double machine learning exists to remove, so
# neither could show a difference. Only stage 9's mboost arm regularised the
# treatment coefficient, and that is the only arm where SUDO clearly won.
#
# The naive plug-in is therefore a straw competitor: it carries regularisation
# bias, and repairing that requires debiasing machinery whichever route is
# taken. The real question is what SUDO's construction buys RELATIVE TO A
# PROPERLY DEBIASED ESTIMATOR of the same latent-scale effect.
#
# Design. Sparse high-dimensional logistic, p = n/2, fitted by cross-fitted
# lasso. Two paired learner arms differing ONLY in penalty.factor on D:
#
#   pen    D is penalised   -> treatment-direction regularisation present
#   unpen  D is unpenalised -> negative control, nuisance regularised only
#
# That switches the mechanism on and off while holding model class, features,
# loss, folds and tuning criterion fixed.
#
# Competitors, all from identical cross-fitted fits:
#   T_U_plug   naive unweighted index contrast
#   T_W_plug   naive overlap-weighted contrast (isolates weighting from truncation)
#   T_U_os     one-step / debiased T_U
#   T_W_os     one-step T_W, including the propensity-derivative term
#   PL_OS      orthogonal logistic partially-linear score  <- PRIMARY COMPETITOR
#   oracle     unpenalised MLE on the active support (harness check, not a rival)
#   sudo       analytic Rao-Blackwellised completion + FWL
#   sudo_mi    SUDO with the model-implied nuisance (implementation diagnostic)
#
# PL_OS is the primary competitor because it is exactly "fit a flexible
# partially-linear model and take the treatment coefficient", done with an
# orthogonal score. Beating a plug-in proves nothing; this is the bar.
#
# Nothing here asserts that SUDO wins. The acceptance block enforces
# computational invariants and an INFORMATIVENESS GATE: if the naive plug-in
# does not visibly fail, the run never entered the regime under study and is
# uninformative rather than evidence either way.
#
# Run from the repository root:
#   Rscript R/stage11_debiased_comparison.R
#
# Live progress (one line per completed replication):
#   SUDO_PROGRESS=/tmp/stage11.log Rscript R/stage11_debiased_comparison.R
#
# Smoke test:
#   SUDO_STAGE11_SMOKE=1 Rscript R/stage11_debiased_comparison.R

source("R/sudo/fwl.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(glmnet))

env_int <- function(name, default) {
  v <- Sys.getenv(name, unset = "")
  if (nzchar(v)) as.integer(v) else as.integer(default)
}
SMOKE <- nzchar(Sys.getenv("SUDO_STAGE11_SMOKE", unset = ""))
PROGRESS <- Sys.getenv("SUDO_PROGRESS", unset = "")
THETA0 <- 1.5

# Grid trimmed from codex's 3-cell spec. Measured throughput on this machine
# was 188 reps/hour at n=1000/p=500, and cv.glmnet cost scales roughly with
# n*p, so the original grid needed about 25 hours of uninterrupted compute.
# n=2000 is the designated primary decision cell, so the main verdict
# survives; the n=4000 no-reversal check is a separate follow-up run.
GRID <- if (SMOKE) {
  data.frame(n = c(300L, 400L), p = c(40L, 60L), reps = c(4L, 4L))
} else {
  data.frame(n = c(1000L, 2000L), p = c(500L, 1000L), reps = c(300L, 250L))
}

# Inner CV folds for every cv.glmnet call. Codex specified 5; 3 cuts roughly
# 40% of the dominant cost and is still a defensible tuning criterion.
INNER_FOLDS <- 3L

A_COEF <- c(0.4675, -0.3825, 0.3400, -0.2975, 0.2550)
B_COEF <- c(0.3600, -0.2925, 0.2475, -0.2250, 0.1800,
            0.2700, -0.2475, 0.2025, -0.1800, 0.1575)

log_progress <- function(msg) {
  if (nzchar(PROGRESS)) {
    cat(sprintf("[%s] %s\n", format(Sys.time(), "%H:%M:%S"), msg),
        file = PROGRESS, append = TRUE)
  }
}

# ---- stable logistic entropy pieces ----------------------------------------
# p and q are computed separately rather than as q = 1 - p, and the truncated
# means use log-scale forms, so nothing is clipped in the primary analysis.
ent_parts <- function(v) {
  lp <- stats::plogis(v, log.p = TRUE)
  lq <- stats::plogis(-v, log.p = TRUE)
  p <- exp(lp); q <- exp(lq)
  list(p = p, q = q, lp = lp, lq = lq,
       H = -(p * lp + q * lq),
       mu1 = -lp - exp(lq - lp) * lq,     # E[xi | Y=1, v]
       mu0 = exp(lp - lq) * lp + lq)      # E[xi | Y=0, v]
}
c_pass <- function(e) { z <- ent_parts(e); 1 - z$H }

dgp11 <- function(n, p) {
  X <- matrix(rnorm(n * p), n, p)
  m0 <- stats::plogis(-0.2 + as.numeric(X[, 1:5, drop = FALSE] %*% A_COEF))
  D <- rbinom(n, 1, m0)
  g0 <- as.numeric(X[, 1:10, drop = FALSE] %*% B_COEF)
  Y <- rbinom(n, 1, stats::plogis(THETA0 * D + g0))
  list(X = X, D = D, Y = Y, m0 = m0, g0 = g0)
}

cv_fit <- function(x, y, fam, pf = NULL, foldid = NULL) {
  args <- list(x = x, y = y, family = fam, alpha = 1, standardize = TRUE)
  if (!is.null(pf)) args$penalty.factor <- pf
  if (!is.null(foldid)) args$foldid <- foldid else args$nfolds <- INNER_FOLDS
  do.call(glmnet::cv.glmnet, args)
}

# ---- one replication --------------------------------------------------------
one_rep <- function(seed, n, p) {
  set.seed(seed)
  d <- dgp11(n, p)
  X <- d$X; D <- d$D; Y <- d$Y
  folds <- make_folds(n, 5L)
  inner <- sample(rep_len(seq_len(INNER_FOLDS), n))  # shared inner CV allocation

  # cross-fitted propensity, X only
  m_hat <- numeric(n)
  for (te in folds) {
    tr <- setdiff(seq_len(n), te)
    f <- cv_fit(X[tr, , drop = FALSE], D[tr], "binomial",
                foldid = inner[tr])
    m_hat[te] <- as.numeric(predict(f, newx = X[te, , drop = FALSE],
                                    s = "lambda.min", type = "response"))
  }
  R <- D - m_hat
  w_hat <- m_hat * (1 - m_hat)

  arm_result <- function(penalise_D) {
    pf <- c(if (penalise_D) 1 else 0, rep(1, p))
    v0 <- v1 <- ell <- ell_mi <- numeric(n)
    nz <- dsel <- lam <- numeric(length(folds))
    for (k in seq_along(folds)) {
      te <- folds[[k]]; tr <- setdiff(seq_len(n), te)
      xt <- cbind(D = D[tr], X[tr, , drop = FALSE])
      f <- cv_fit(xt, Y[tr], "binomial", pf = pf, foldid = inner[tr])
      cf <- as.numeric(coef(f, s = "lambda.min"))
      lam[k] <- f$lambda.min; nz[k] <- sum(cf[-1] != 0); dsel[k] <- cf[2] != 0
      pred <- function(dv, idx) as.numeric(predict(
        f, newx = cbind(D = dv, X[idx, , drop = FALSE]), s = "lambda.min",
        type = "link"))
      v0[te] <- pred(0, te); v1[te] <- pred(1, te)
      # SUDO outcome nuisance, fitted ENTIRELY inside the training sample
      vtr <- pred(D[tr], tr)
      ztr <- ent_parts(vtr)
      Str <- vtr + ifelse(Y[tr] == 1L, ztr$mu1, ztr$mu0)
      Ztr <- cbind(m = m_hat[tr], X[tr, , drop = FALSE])
      fl <- cv_fit(Ztr, Str, "gaussian", foldid = inner[tr])
      ell[te] <- as.numeric(predict(
        fl, newx = cbind(m = m_hat[te], X[te, , drop = FALSE]),
        s = "lambda.min"))
      ell_mi[te] <- pred(0, te) + (pred(1, te) - pred(0, te)) * m_hat[te]
    }
    v <- ifelse(D == 1L, v1, v0)
    z <- ent_parts(v); z0 <- ent_parts(v0); z1 <- ent_parts(v1)
    S <- v + ifelse(Y == 1L, z$mu1, z$mu0)
    delta <- v1 - v0

    sudo    <- sum(R * (S - ell)) / sum(R^2)
    sudo_mi <- sum(R * (S - ell_mi)) / sum(R^2)
    T_U_plug <- mean(delta)
    T_W_plug <- sum(w_hat * delta) / sum(w_hat)
    T_U_os <- mean(delta + D / m_hat * (Y - z1$p) / (z1$p * z1$q) -
                     (1 - D) / (1 - m_hat) * (Y - z0$p) / (z0$p * z0$q))
    Ai <- w_hat * delta + (1 - 2 * m_hat) * delta * R +
      D * (1 - m_hat) * (Y - z1$p) / (z1$p * z1$q) -
      (1 - D) * m_hat * (Y - z0$p) / (z0$p * z0$q)
    Bi <- w_hat + (1 - 2 * m_hat) * R
    T_W_os <- sum(Ai) / sum(Bi)

    # orthogonal logistic partially-linear score
    score <- function(th) {
      q1 <- ent_parts(v0 + th); q0 <- ent_parts(v0)
      a <- m_hat * q1$p * q1$q /
        (m_hat * q1$p * q1$q + (1 - m_hat) * q0$p * q0$q)
      pD <- ifelse(D == 1L, q1$p, q0$p)
      sum((D - a) * (Y - pD))
    }
    pl_os <- tryCatch(stats::uniroot(score, c(-4, 6))$root, error = function(e)
      tryCatch(stats::uniroot(score, c(-10, 10))$root,
               error = function(e2) NA_real_))

    # exact channel decomposition, path fixed: X-error first, then treatment
    eta0 <- d$g0; eta1 <- THETA0 + d$g0
    w0 <- d$m0 * (1 - d$m0)
    h <- v0 - eta0
    dl <- delta - THETA0
    psi <- function(vv, ee) {
      zz <- ent_parts(vv); pe <- ent_parts(ee)$p
      vv + pe * zz$mu1 + (1 - pe) * zz$mu0
    }
    Btot <- sum(w0 * ((psi(eta1 + h + dl, eta1) - eta1) -
                        (psi(eta0 + h, eta0) - eta0))) / sum(w0)
    BX <- sum(w0 * ((psi(eta1 + h, eta1) - eta1) -
                      (psi(eta0 + h, eta0) - eta0))) / sum(w0)
    data.frame(
      sudo = sudo, sudo_mi = sudo_mi, T_U_plug = T_U_plug,
      T_W_plug = T_W_plug, T_U_os = T_U_os, T_W_os = T_W_os, PL_OS = pl_os,
      B_total = Btot, B_X = BX, B_D = Btot - BX,
      B_D_first = sum(w0 * c_pass(eta1) * dl) / sum(w0),
      B_X_first = sum(w0 * h * (c_pass(eta1) - c_pass(eta0))) / sum(w0),
      direct_treat_err = sum(w0 * dl) / sum(w0),
      mean_lambda = mean(lam), mean_nz = mean(nz), d_selected = mean(dsel),
      min_p = min(c(z0$p, z1$p)), min_q = min(c(z0$q, z1$q)))
  }

  pen <- cbind(arm = "pen", arm_result(TRUE))
  unp <- cbind(arm = "unpen", arm_result(FALSE))
  ox <- cbind(D = D, X[, 1:10, drop = FALSE])
  oracle <- unname(coef(stats::glm.fit(cbind(1, ox), Y,
                                       family = binomial()))[2])
  out <- rbind(pen, unp, make.row.names = FALSE)
  out$oracle <- oracle
  out$n <- n; out$p <- p; out$seed <- seed
  out
}

# ---- driver -----------------------------------------------------------------
if (nzchar(PROGRESS)) cat("", file = PROGRESS)
cat(sprintf("Stage 11: SUDO vs debiased competitors%s\n", if (SMOKE) " [SMOKE]" else ""))
cat(sprintf("theta0 = %.1f, grid: %s\n\n", THETA0,
            paste(sprintf("n=%d/p=%d/%dreps", GRID$n, GRID$p, GRID$reps),
                  collapse = ", ")))

cl <- mc_cluster(c("dgp11", "one_rep", "ent_parts", "c_pass", "cv_fit",
                   "log_progress", "A_COEF", "B_COEF", "THETA0", "PROGRESS",
                   "INNER_FOLDS"))
invisible(parallel::clusterEvalQ(cl, suppressPackageStartupMessages(
  library(glmnet))))

all_reps <- list()
for (g in seq_len(nrow(GRID))) {
  n <- GRID$n[g]; p <- GRID$p[g]; reps <- GRID$reps[g]
  parallel::clusterExport(cl, c("n", "p", "reps"), envir = environment())
  t0 <- Sys.time()
  rows <- parallel::parLapplyLB(cl, seq_len(reps), function(r) {
    out <- one_rep(11000 + 1000 * n + r, n, p)
    log_progress(sprintf("n=%d p=%d rep %d/%d", n, p, r, reps))
    out
  })
  all_reps[[g]] <- do.call(rbind, rows)
  cat(sprintf("  n=%4d p=%4d: %d reps in %.1f min\n", n, p, reps,
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  # checkpoint after every cell: a kill or a sleeping laptop must not throw
  # away completed work, which is exactly what happened on the first attempt
  dir.create("R/results", showWarnings = FALSE)
  utils::write.csv(do.call(rbind, all_reps),
                   "R/results/stage11_debiased_replications.csv",
                   row.names = FALSE)
  cat(sprintf("    checkpointed %d replications\n",
              nrow(do.call(rbind, all_reps))))
}
parallel::stopCluster(cl)
reps_df <- do.call(rbind, all_reps)

EST <- c("sudo", "sudo_mi", "T_U_plug", "T_W_plug", "T_U_os", "T_W_os", "PL_OS")
summ <- do.call(rbind, lapply(split(reps_df, list(reps_df$n, reps_df$arm),
                                    drop = TRUE), function(b) {
  do.call(rbind, lapply(EST, function(e) {
    x <- b[[e]]; ok <- is.finite(x)
    data.frame(n = b$n[1], p = b$p[1], arm = b$arm[1], estimator = e,
               n_reps = sum(ok), fail_rate = mean(!ok),
               bias = mean(x[ok]) - THETA0,
               mc_se_bias = stats::sd(x[ok]) / sqrt(sum(ok)),
               sd = stats::sd(x[ok]),
               rmse = sqrt(mean((x[ok] - THETA0)^2)),
               std_bias = abs(mean(x[ok]) - THETA0) / stats::sd(x[ok]),
               rmse_vs_PL_OS = sqrt(mean((x[ok] - THETA0)^2)) /
                 sqrt(mean((b$PL_OS[is.finite(b$PL_OS)] - THETA0)^2)))
  }))
}))
rownames(summ) <- NULL
summ <- summ[order(summ$n, summ$arm, summ$estimator), ]

for (a in c("pen", "unpen")) {
  cat(sprintf("\n=== arm: %s (D %s) ===\n", a,
              if (a == "pen") "penalised" else "UNpenalised"))
  cat(sprintf("%6s %-9s %9s %9s %8s %9s %10s\n",
              "n", "estimator", "bias", "mc_se", "sd", "rmse", "rmse/PLOS"))
  s <- summ[summ$arm == a, ]
  for (i in seq_len(nrow(s))) {
    cat(sprintf("%6d %-9s %+9.4f %9.4f %8.4f %9.4f %10.3f\n",
                s$n[i], s$estimator[i], s$bias[i], s$mc_se_bias[i],
                s$sd[i], s$rmse[i], s$rmse_vs_PL_OS[i]))
  }
}

dec <- do.call(rbind, lapply(split(reps_df, list(reps_df$n, reps_df$arm),
                                   drop = TRUE), function(b) data.frame(
  n = b$n[1], arm = b$arm[1],
  B_total = mean(b$B_total), B_X = mean(b$B_X), B_D = mean(b$B_D),
  B_D_first = mean(b$B_D_first), B_X_first = mean(b$B_X_first),
  direct_treat_err = mean(b$direct_treat_err),
  decomp_resid = max(abs(b$B_total - b$B_X - b$B_D)),
  d_selected = mean(b$d_selected), mean_nz = mean(b$mean_nz),
  min_p = min(b$min_p))))
rownames(dec) <- NULL
cat("\n=== channel decomposition (X-error first, then treatment) ===\n")
cat(sprintf("%6s %-6s %9s %9s %9s %12s %10s\n",
            "n", "arm", "B_total", "B_X", "B_D", "treat_err", "d_selected"))
for (i in seq_len(nrow(dec))) {
  cat(sprintf("%6d %-6s %+9.4f %+9.4f %+9.4f %+12.4f %10.2f\n",
              dec$n[i], dec$arm[i], dec$B_total[i], dec$B_X[i], dec$B_D[i],
              dec$direct_treat_err[i], dec$d_selected[i]))
}

# ---- acceptance: invariants and the informativeness gate --------------------
# The oracle does not depend on the arm, so it is duplicated across the two
# arm rows; de-duplicate before computing its Monte Carlo SE or the check is
# too strict by a factor of sqrt(2). Check per cell rather than pooled: the
# logistic MLE's finite-sample bias is O(1/n), so pooling different n mixes
# genuinely different bias levels and fails for the wrong reason.
oracle_by_cell <- lapply(split(reps_df[reps_df$arm == "pen", ],
                               reps_df$n[reps_df$arm == "pen"]),
                         function(b) {
  data.frame(n = b$n[1], bias = abs(mean(b$oracle) - THETA0),
             mcse = stats::sd(b$oracle) / sqrt(nrow(b)))
})
oracle_tab <- do.call(rbind, oracle_by_cell)
oracle_bias <- max(oracle_tab$bias)
oracle_mcse <- min(oracle_tab$mcse)
oracle_ok <- all(oracle_tab$bias < 3 * oracle_tab$mcse)
big_n <- max(GRID$n)
gate_row <- summ[summ$n == big_n & summ$arm == "pen" &
                   summ$estimator == "T_W_plug", ]
gate_ok <- abs(gate_row$bias) >= 0.05 &&
  abs(gate_row$bias) > 4 * gate_row$mc_se_bias
plos_fail <- max(summ$fail_rate[summ$estimator == "PL_OS"])

stopifnot(
  nrow(summ) == nrow(GRID) * 2L * length(EST),
  all(is.finite(summ$bias)), all(summ$sd > 0),
  max(dec$decomp_resid) < 1e-10,
  oracle_ok,
  plos_fail < 0.02
)
cat(sprintf("\nPASS: invariants hold (oracle bias %.4f < 3 MCSE %.4f; ",
            oracle_bias, 3 * oracle_mcse))
cat(sprintf("decomp resid %.1e; PL_OS failures %.1f%%)\n",
            max(dec$decomp_resid), 100 * plos_fail))
cat(sprintf("INFORMATIVENESS GATE (naive T_W_plug bias >= 0.05 at n=%d): %s\n",
            big_n, if (gate_ok) "PASSED" else
              "FAILED - run did not enter the regime under study"))

dir.create("R/results", showWarnings = FALSE)
write.csv(summ, "R/results/stage11_debiased_comparison.csv", row.names = FALSE)
write.csv(reps_df, "R/results/stage11_debiased_replications.csv",
          row.names = FALSE)
cat("wrote R/results/stage11_debiased_comparison.csv and _replications.csv\n")
