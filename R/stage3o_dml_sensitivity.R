# Stage 3o: how does SUDO's nuisance sensitivity compare to standard DML's?
#
# The natural objection to the whole black-box program is "wouldn't
# ordinary DML also struggle with inaccurate nuisances?" It would — but
# only at SECOND order, and that difference is the point.
#
# In DML PLR the outcome is observed, and nuisance error enters theta only
# through the PRODUCT of the outcome-nuisance and treatment-nuisance
# errors: get either one right and the bias vanishes. SUDO adds a third
# nuisance, the imputation model, whose error is baked into the
# constructed outcome S rather than differenced away, and enters LINEARLY.
#
# This stage measures both, on a common axis. The fairness anchor matters:
# rather than "use a worse learner" (incomparable across methods), each
# arm is perturbed by a KNOWN function scaled by tau, and bias is plotted
# against latent-scale RMSE = tau * sd(h). DML's l_0 and SUDO's index both
# live on the latent scale, so the x-axes are genuinely comparable.
#
# Arms (oracle nuisances everywhere except where perturbed):
#   DML-1  plr_crossfit on OBSERVED U; l perturbed, m oracle
#            -> predicted bias EXACTLY 0 at every tau (one exact nuisance
#               is enough; this is the arm that makes the point)
#   DML-2  both nuisances perturbed      -> predicted pure QUADRATIC in tau
#   SUDO-X oracle index + tau*h(X)       -> predicted LINEAR, small slope
#   SUDO-D oracle index + kappa*D        -> predicted LINEAR, slope cbar_1
#
# SUDO-D doubles as an independent measurement of the pass-through
# constant derived in the paper appendix (Prop A3): its fitted slope equals
# cbar_1 (0.669 at theta=1.5, 0.861 at theta=3). That is the headline
# assertion — it confirms the theory by a completely separate route from
# the retro-prediction table.
#
# Run from repo root: Rscript R/stage3o_dml_sensitivity.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")

dgp_eta <- function(n, theta) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  eta <- theta * D + g
  U <- eta + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0), eta = eta, U = U, g = g)
}

# perturbation directions (centred so tau only changes shape, not level)
h_adv  <- function(X) X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3] - 1  # E[g] = 1
h_m    <- function(X) X[, 4] + cos(X[, 5]) - exp(-0.5)           # E[cos X5]

oracle_m <- function(X, y) function(Xnew) plogis(Xnew$X4 + cos(Xnew$X5))
make_oracle_l <- function(theta) {
  force(theta)
  function(X, y) function(Xnew)
    Xnew$X1^2 + sin(Xnew$X2) + 0.5 * Xnew$X3 +
      theta * plogis(Xnew$X4 + cos(Xnew$X5))
}
# oracle l perturbed by tau*h_adv, on the latent scale
make_pert_l <- function(theta, tau) {
  force(theta); force(tau)
  function(X, y) function(Xnew)
    Xnew$X1^2 + sin(Xnew$X2) + 0.5 * Xnew$X3 +
      theta * plogis(Xnew$X4 + cos(Xnew$X5)) +
      tau * (Xnew$X1^2 + sin(Xnew$X2) + 0.5 * Xnew$X3 - 1)
}
# oracle m perturbed on the probability scale, clipped to stay a probability
make_pert_m <- function(tau) {
  force(tau)
  function(X, y) function(Xnew)
    pmin(pmax(plogis(Xnew$X4 + cos(Xnew$X5)) +
                tau * 0.1 * (Xnew$X4 + cos(Xnew$X5) - exp(-0.5)), 0.01), 0.99)
}

cl <- mc_cluster(c("dgp_eta", "h_adv", "h_m", "oracle_m", "make_oracle_l",
                  "make_pert_l", "make_pert_m"))
invisible(parallel::clusterEvalQ(cl, source("R/sudo/estimator.R")))

make_dgp <- function(n, theta) {
  force(n); force(theta)
  function() dgp_eta(n, theta)
}

make_est <- function(arm, theta, amt) {
  force(arm); force(theta); force(amt)
  ol <- make_oracle_l(theta)
  switch(arm,
    "DML-1" = function(d) {
      r <- plr_crossfit(d$U, d$D, as.data.frame(d$X),
                        fit_l = make_pert_l(theta, amt), fit_m = oracle_m)
      list(theta = r$theta, se = r$se)
    },
    "DML-2" = function(d) {
      r <- plr_crossfit(d$U, d$D, as.data.frame(d$X),
                        fit_l = make_pert_l(theta, amt),
                        fit_m = make_pert_m(amt))
      list(theta = r$theta, se = r$se)
    },
    "SUDO-X" = function(d) {
      fm <- function(dd, folds)
        list(lp_hat = dd$eta + amt * h_adv(dd$X), fit_fold = NULL,
             to_lp = identity)
      sudo_binary(d, full_model = fm, fit_l = ol, fit_m = oracle_m, B = 25)
    },
    "SUDO-D" = function(d) {
      fm <- function(dd, folds)
        list(lp_hat = dd$eta + amt * dd$D, fit_fold = NULL, to_lp = identity)
      sudo_binary(d, full_model = fm, fit_l = ol, fit_m = oracle_m, B = 25)
    })
}

taus <- c(0, 0.1, 0.2, 0.4, 0.8)
kappas <- c(-0.4, -0.2, 0, 0.2, 0.4)
n_reps <- 200

# theoretical predictions from the pass-through derivation
set.seed(7); Xbig <- matrix(rnorm(2e5 * 5), 2e5, 5,
                            dimnames = list(NULL, paste0("X", 1:5)))
gbig <- Xbig[, 1]^2 + sin(Xbig[, 2]) + 0.5 * Xbig[, 3]
mbig <- plogis(Xbig[, 4] + cos(Xbig[, 5])); wbig <- mbig * (1 - mbig)
cbar <- sapply(c(1.5, 3), function(th) pass_cbar1(th + gbig, mbig))
names(cbar) <- c("1.5", "3")
xleak <- sapply(c(1.5, 3), function(th)
  sum(wbig * (pass_c(th + gbig) - pass_c(gbig)) * h_adv(Xbig)) / sum(wbig))
names(xleak) <- c("1.5", "3")
cat(sprintf("theory: cbar_1 = %.3f (t1.5), %.3f (t3);  X-leak slope = %+.3f, %+.3f\n\n",
            cbar["1.5"], cbar["3"], xleak["1.5"], xleak["3"]))

rows <- list()
for (th in c(1.5, 3)) {
  cat(sprintf("--- theta = %.1f ---\n", th))
  for (arm in c("DML-1", "DML-2", "SUDO-X", "SUDO-D")) {
    amts <- if (arm == "SUDO-D") kappas else taus
    biases <- numeric(length(amts))
    for (i in seq_along(amts)) {
      df <- run_mc_par(cl, n_reps, make_dgp(2000, th),
                       make_est(arm, th, amts[i]), th, seed = 7000)
      s <- summarize_mc(df)
      biases[i] <- s$bias
      rows[[length(rows) + 1]] <- data.frame(
        stage = "3o", theta = th, arm = arm, amount = amts[i],
        latent_rmse = abs(amts[i]) * ifelse(arm == "SUDO-D", 0.5, sd(h_adv(Xbig))),
        bias = s$bias, mc_se_bias = s$mc_se_bias, coverage = s$coverage,
        n_reps = n_reps)
    }
    fit <- lm(biases ~ amts)
    cat(sprintf("  %-7s biases: %s | slope %+.3f\n", arm,
                paste(sprintf("%+.3f", biases), collapse = " "),
                unname(coef(fit)[2])))
  }
  cat("\n")
}
parallel::stopCluster(cl)
sm <- do.call(rbind, rows)

slope_of <- function(arm, th) {
  s <- sm[sm$arm == arm & sm$theta == th, ]
  unname(coef(lm(s$bias ~ s$amount))[2])
}
cat("== headline: SUDO-D slope should equal cbar_1 (independent route to the theory) ==\n")
for (th in c(1.5, 3)) {
  sl <- slope_of("SUDO-D", th); cb <- cbar[[as.character(th)]]
  cat(sprintf("theta=%.1f  measured %.3f  theory %.3f  diff %.3f\n",
              th, sl, cb, abs(sl - cb)))
}
cat("\n== DML arms ==\n")
for (th in c(1.5, 3)) {
  d1 <- sm[sm$arm == "DML-1" & sm$theta == th, ]
  cat(sprintf("theta=%.1f  DML-1 max|bias| over the whole tau ladder = %.4f (mc_se ~%.4f)\n",
              th, max(abs(d1$bias)), mean(d1$mc_se_bias)))
}

# DML-1 is a harness self-test: with one exact nuisance, orthogonality
# makes the bias exactly zero regardless of how wrong the other one is.
# If this fails, the perturbation plumbing is broken, not the theory.
for (th in c(1.5, 3)) {
  d1 <- sm[sm$arm == "DML-1" & sm$theta == th, ]
  stopifnot(max(abs(d1$bias)) < 4 * mean(d1$mc_se_bias))
}
cat("\nPASS: DML-1 unbiased at every perturbation level (orthogonality self-test).\n")
for (th in c(1.5, 3)) {
  stopifnot(abs(slope_of("SUDO-D", th) - cbar[[as.character(th)]]) < 0.08)
}
cat("PASS: SUDO-D slope matches the derived pass-through constant at both thetas.\n")

write.csv(sm, "R/results/stage3o_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3o_summary.csv\n")
