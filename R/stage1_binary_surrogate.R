# Stage 1: binary surrogate completion, single draw, parametric everything.
# Logistic DGP eta = theta*D + beta*X; the full model glm(Y ~ D + X) is the
# imputation model for the latent utility. Three numbers should agree:
#   (a) glm coefficient on D (parametric benchmark)
#   (b) FWL on the TRUE latent index (oracle)
#   (c) FWL on one truncated-logistic surrogate draw (the method)
# Run from repo root: Rscript R/stage1_binary_surrogate.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/mc.R")

theta0 <- 1.5

dgp_bin <- function(n = 5000, theta = theta0, beta = 1.0) {
  X <- rnorm(n)
  D <- rbinom(n, 1, plogis(0.8 * X))
  eta <- theta * D + beta * X
  Y <- rbinom(n, 1, plogis(eta))
  list(X = X, D = D, Y = Y, eta = eta)
}

est_all <- function(d) {
  rD <- resid(lm(d$D ~ d$X))
  glm_fit <- glm(d$Y ~ d$D + d$X, family = binomial)
  # oracle: FWL on the true latent index
  oracle <- fwl_theta(resid(lm(d$eta ~ d$X)), rD)$theta
  # surrogate: complete Y to S at the fitted index, then FWL
  S <- complete_surrogate(d$Y + 1L, predict(glm_fit), c(-Inf, 0, Inf), "logit")
  surr <- fwl_theta(resid(lm(S ~ d$X)), rD)
  list(glm = unname(coef(glm_fit)["d$D"]), oracle = oracle,
       theta = surr$theta, se = surr$se)
}

n_reps <- 200
res <- t(sapply(seq_len(n_reps), function(r) {
  set.seed(r)
  e <- est_all(dgp_bin())
  c(glm = e$glm, oracle = e$oracle, surrogate = e$theta, se = e$se)
}))

cat(sprintf("true theta = %.1f over %d reps:\n", theta0, n_reps))
for (k in c("glm", "oracle", "surrogate")) {
  m <- mean(res[, k]); s <- sd(res[, k])
  cat(sprintf("  %-10s mean %.4f  sd %.4f  bias %+.4f (mc-se %.4f)\n",
              k, m, s, m - theta0, s / sqrt(n_reps)))
}

for (k in c("glm", "oracle", "surrogate")) {
  bias <- mean(res[, k]) - theta0
  stopifnot(abs(bias) < 2 * sd(res[, k]) / sqrt(n_reps) + 1e-9)
}
cat("PASS: glm, oracle, and surrogate FWL all within 2*MC-SE of theta\n")

write.csv(data.frame(stage = 1, estimator = c("glm", "oracle", "surrogate"),
                     mean_est = colMeans(res[, 1:3]),
                     sd = apply(res[, 1:3], 2, sd), n_reps = n_reps),
          "R/results/stage1_summary.csv", row.names = FALSE)
cat("wrote R/results/stage1_summary.csv\n")
