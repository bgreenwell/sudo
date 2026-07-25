# Stage 3c: "why not just read theta off the full model of Y on (D, X)?"
# Confounded DGP — the propensity uses the SAME covariates that drive g, so
# errors in modeling g leak into any non-orthogonal estimate of theta:
#   g(X) = X1^2 + sin(X2) + 0.5*X3,   P(D=1|X) = plogis(X1 + cos(X2))
# Three estimators on identical data (theta=1.5, n=2000, 200 reps):
#   direct_glm  coefficient on D from glm(Y ~ D + X1..X5, binomial) — the
#               critic with a misspecified (linear-in-X) parametric model
#   direct_gam  coefficient on D from gam(Y ~ D + s(X1)+..+s(X5), binomial)
#               — the critic's best case: correct additive form, penalized
#   sudo        full SUDO (gam full model, proper draws, Rubin)
# Run from repo root: Rscript R/stage3c_direct_comparison.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")
suppressPackageStartupMessages(library(mgcv))
source("R/sudo/estimator.R")

theta0 <- 1.5
n_obs <- 2000
n_reps <- 200

dgp3c <- function(n = n_obs, theta = theta0) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 1] + cos(X[, 2])))
  U <- theta * D + g + rlogis(n)
  list(X = X, D = D, Y = as.integer(U > 0))
}

est_direct_glm <- function(d) {
  fit <- glm(d$Y ~ d$D + d$X, family = binomial)
  co <- summary(fit)$coefficients["d$D", ]
  list(theta = unname(co["Estimate"]), se = unname(co["Std. Error"]),
       ci_lo = co["Estimate"] - 1.96 * co["Std. Error"],
       ci_hi = co["Estimate"] + 1.96 * co["Std. Error"])
}

est_direct_gam <- function(d) {
  dat <- data.frame(Y = d$Y, D = d$D, d$X)
  rhs <- paste(sprintf("s(%s)", colnames(d$X)), collapse = " + ")
  fit <- gam(as.formula(paste("Y ~ D +", rhs)), family = binomial,
             data = dat)
  th <- unname(coef(fit)["D"])
  se <- sqrt(vcov(fit)["D", "D"])
  list(theta = th, se = se, ci_lo = th - 1.96 * se, ci_hi = th + 1.96 * se)
}

cl <- mc_cluster(c("dgp3c", "est_direct_glm", "est_direct_gam",
                   "sudo_binary", "crossfit_fullmodel_gam",
                   "n_obs", "theta0"))

estimators <- list(direct_glm = est_direct_glm,
                   direct_gam = est_direct_gam,
                   sudo = function(d) sudo_binary(d))

cat(sprintf("confounded DGP, true theta = %.1f, n = %d, %d reps\n\n",
            theta0, n_obs, n_reps))
all_summ <- list()
for (nm in names(estimators)) {
  df <- run_mc_par(cl, n_reps, dgp3c, estimators[[nm]], theta0, seed = 7000)
  s <- summarize_mc(df)
  print_mc(nm, s)
  all_summ[[nm]] <- cbind(stage = "3c", estimator = nm, s)
}
parallel::stopCluster(cl)
sm <- do.call(rbind, all_summ)

g <- function(nm) sm[sm$estimator == nm, ]
# the misspecified parametric read-off must be visibly biased; SUDO must be
# unbiased with honest coverage on the same data
stopifnot(abs(g("direct_glm")$bias) > 4 * g("direct_glm")$mc_se_bias)
stopifnot(abs(g("sudo")$bias) <
          max(2 * g("sudo")$mc_se_bias, 0.02 * theta0) + 1e-9)
stopifnot(g("sudo")$coverage >= 0.925)
cat("\nPASS: misspecified direct read-off biased; SUDO unbiased with",
    "honest coverage\n")
cat("(direct_gam is reported, not asserted: the critic's best case)\n")

write.csv(sm, "R/results/stage3c_summary.csv", row.names = FALSE)
cat("wrote R/results/stage3c_summary.csv\n")
