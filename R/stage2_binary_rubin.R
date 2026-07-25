# Stage 2: surrogate draws as multiple imputations of the latent utility.
# Same DGP/estimator as stage 1, but B draws pooled with Rubin's rules.
# Three inference schemes:
#   naive     single-draw sandwich CI (ignores draw uncertainty)
#   improper  B draws from the SAME fitted index; Rubin-t pooling
#             (misses imputation-model parameter uncertainty)
#   proper    per draw, full-model coefficients are redrawn from their
#             asymptotic posterior N(beta_hat, vcov) before sampling S;
#             Rubin-t pooling. This is proper MI in Rubin's sense.
# Run from repo root: Rscript R/stage2_binary_rubin.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/mc.R")

theta0 <- 1.5
n_obs <- 5000
Bs <- c(10, 25, 50)
n_reps <- 500

dgp_bin <- function(n = n_obs, theta = theta0, beta = 1.0) {
  X <- rnorm(n)
  D <- rbinom(n, 1, plogis(0.8 * X))
  Y <- rbinom(n, 1, plogis(theta * D + beta * X))
  list(X = X, D = D, Y = Y)
}

one_rep <- function(d) {
  rD <- resid(lm(d$D ~ d$X))
  fit <- glm(d$Y ~ d$D + d$X, family = binomial)
  mm <- model.matrix(fit)
  lp <- predict(fit)
  beta_hat <- coef(fit)
  V <- vcov(fit)

  draw_one <- function(lp_b) {
    S <- complete_surrogate(d$Y + 1L, lp_b, c(-Inf, 0, Inf), "logit")
    f <- fwl_theta(resid(lm(S ~ d$X)), rD)
    c(theta = f$theta, var = f$var)
  }
  improper <- sapply(seq_len(max(Bs)), function(b) draw_one(lp))
  proper <- sapply(seq_len(max(Bs)), function(b) {
    beta_b <- MASS::mvrnorm(1, beta_hat, V)
    draw_one(as.numeric(mm %*% beta_b))
  })

  out <- list(naive = list(theta = improper["theta", 1],
                           se = sqrt(improper["var", 1])))
  for (B in Bs) {
    for (kind in c("improper", "proper")) {
      dr <- get(kind)
      p <- pool_rubin(dr["theta", 1:B], dr["var", 1:B], n_obs = n_obs)
      out[[sprintf("%s_B%d", kind, B)]] <-
        list(theta = p$theta, se = p$se, ci_lo = p$ci_lo, ci_hi = p$ci_hi)
    }
  }
  out
}

labels <- c("naive", as.vector(outer(c("improper_B", "proper_B"), Bs, paste0)))
rows <- list()
for (r in seq_len(n_reps)) {
  set.seed(r)
  reps <- one_rep(dgp_bin())
  for (lab in labels) {
    e <- reps[[lab]]
    ci_lo <- if (!is.null(e$ci_lo)) e$ci_lo else e$theta - 1.96 * e$se
    ci_hi <- if (!is.null(e$ci_hi)) e$ci_hi else e$theta + 1.96 * e$se
    rows[[length(rows) + 1]] <- data.frame(
      estimator = lab, rep = r, est = e$theta, se = e$se,
      covered = ci_lo <= theta0 & theta0 <= ci_hi)
  }
}
df <- do.call(rbind, rows)

cat(sprintf("true theta = %.1f, n = %d, %d reps\n\n", theta0, n_obs, n_reps))
summ <- do.call(rbind, lapply(labels, function(lab) {
  s <- df[df$estimator == lab, ]
  out <- summarize_mc(s, theta0)
  print_mc(lab, out)
  cbind(stage = 2, estimator = lab, out)
}))

cov_of <- function(lab) summ$coverage[summ$estimator == lab]
stopifnot(cov_of("naive") < 0.925)
for (B in Bs) stopifnot(cov_of(sprintf("proper_B%d", B)) >= 0.925,
                        cov_of(sprintf("proper_B%d", B)) <= 0.975)
cat("\nPASS: proper-MI Rubin-t coverage in [0.925, 0.975] for all B;",
    "naive under-covers\n")

write.csv(summ, "R/results/stage2_summary.csv", row.names = FALSE)
cat("wrote R/results/stage2_summary.csv\n")
