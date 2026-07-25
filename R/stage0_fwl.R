# Stage 0: FWL is the backbone.
#   0a  exact FWL theorem: residual-on-residual == lm coefficient on D
#   0b  PLR with nonlinear nuisances: cross-fit gam partialling-out is
#       unbiased with nominal coverage; one-seed cross-check vs DoubleML.
# Run from repo root: Rscript R/stage0_fwl.R

source("R/sudo/fwl.R")
source("R/sudo/mc.R")

cat("== 0a: exact FWL on a linear model ==\n")
set.seed(1)
n <- 500
X <- matrix(rnorm(n * 3), n, 3, dimnames = list(NULL, paste0("X", 1:3)))
D <- 0.5 * X[, 1] + rnorm(n)
Y <- 1.5 * D + X %*% c(1, -1, 0.5) + rnorm(n)
lm_coef <- unname(coef(lm(Y ~ D + X))["D"])
S_res <- resid(lm(Y ~ X))
D_res <- resid(lm(D ~ X))
fwl <- fwl_theta(S_res, D_res)$theta
cat(sprintf("lm coef %.10f  FWL %.10f  |diff| %.2e\n", lm_coef, fwl,
            abs(lm_coef - fwl)))
stopifnot(abs(lm_coef - fwl) < 1e-10)
cat("PASS: FWL matches lm to <1e-10\n\n")

# nonlinear PLR DGP (v1 functional forms), continuous outcome
dgp_plr <- function(n = 500, theta = 1.0) {
  X <- matrix(rnorm(n * 5), n, 5, dimnames = list(NULL, paste0("X", 1:5)))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  m <- X[, 4] + cos(X[, 5])
  D <- m + rnorm(n)
  Y <- theta * D + g + rnorm(n)
  list(X = X, D = D, Y = Y)
}

cat("== 0b: cross-fit PLR (gam nuisances), 200-rep MC ==\n")
theta0 <- 1.0
est_plr <- function(d) plr_crossfit(d$Y, d$D, d$X, fit_l = fit_gam, fit_m = fit_gam)
df <- run_mc(200, function() dgp_plr(1000, theta0), est_plr, theta0, seed = 100)
summ <- summarize_mc(df)
print_mc("PLR gam crossfit", summ)
stopifnot(abs(summ$bias) < 2 * summ$mc_se_bias + 1e-9)
stopifnot(summ$coverage >= 0.925, summ$coverage <= 0.975)
cat("PASS: |bias| < 2*MC-SE and coverage in [0.925, 0.975]\n\n")

cat("== 0b cross-check: one seed vs DoubleML-R (ranger nuisances) ==\n")
suppressMessages({library(DoubleML); library(mlr3); library(mlr3learners)})
lgr::get_logger("mlr3")$set_threshold("warn")
set.seed(42)
d <- dgp_plr(2000, theta0)
mine <- plr_crossfit(d$Y, d$D, d$X, fit_l = fit_ranger, fit_m = fit_ranger)
dml_data <- double_ml_data_from_matrix(X = d$X, y = d$Y, d = d$D)
set.seed(42)
dml <- DoubleMLPLR$new(dml_data,
                       ml_l = lrn("regr.ranger", num.trees = 200, min.node.size = 5),
                       ml_m = lrn("regr.ranger", num.trees = 200, min.node.size = 5),
                       n_folds = 5)
dml$fit()
cat(sprintf("plr_crossfit %.4f (se %.4f)   DoubleMLPLR %.4f (se %.4f)\n",
            mine$theta, mine$se, dml$coef, dml$se))
stopifnot(abs(mine$theta - dml$coef) < 2 * sqrt(mine$se^2 + dml$se^2))
cat("PASS: estimates agree within 2*combined SE\n")

dir.create("R/results", showWarnings = FALSE)
write.csv(cbind(stage = 0, estimator = "plr_gam", summ),
          "R/results/stage0_summary.csv", row.names = FALSE)
cat("wrote R/results/stage0_summary.csv\n")
