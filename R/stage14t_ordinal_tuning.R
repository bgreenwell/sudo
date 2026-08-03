# Stage 14t: target-level tuning for the flexible ordinal learner.
#
# This stage uses dedicated seeds that do not overlap stage 14 confirmation.
# It searches mstop and learning rate by absolute SUDO bias, not predictive
# deviance. The selected configuration is frozen in a CSV for inspection.
# Stage 14 defaults match the current selected configuration.
#
# Full screen: 20 replications per J and configuration at n=1000, B=5.
# Smoke example:
#   SUDO_STAGE14T_N=400 SUDO_STAGE14T_REPS=2 SUDO_STAGE14T_B=2 \
#   SUDO_STAGE14T_MSTOP=100,300 Rscript R/stage14t_ordinal_tuning.R

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/discrete.R")

env_int <- function(name, default) {
  value <- Sys.getenv(name, unset = "")
  if (nzchar(value)) as.integer(value) else as.integer(default)
}
n <- env_int("SUDO_STAGE14T_N", 1000L)
n_reps <- env_int("SUDO_STAGE14T_REPS", 20L)
B <- env_int("SUDO_STAGE14T_B", 5L)
mstop_value <- Sys.getenv("SUDO_STAGE14T_MSTOP", unset = "")
mstop_grid <- if (nzchar(mstop_value)) {
  as.integer(strsplit(mstop_value, ",")[[1]])
} else {
  c(500L, 1000L, 2000L)
}
nu_grid <- c(0.2, 0.5)

dgp14t <- function(n, J) {
  X <- matrix(rnorm(n * 5), n, 5,
              dimnames = list(NULL, paste0("X", seq_len(5))))
  g <- X[, 1]^2 + sin(X[, 2]) + 0.5 * X[, 3]
  D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
  cuts <- if (J == 3L) c(0, 2) else c(-1, 0.5, 1.75, 3)
  Y <- 1L + findInterval(D + g + rlogis(n), cuts)
  list(X = X, D = D, Y = Y)
}

rows <- list()
for (mstop in mstop_grid) {
  for (nu in nu_grid) {
    for (J in c(3L, 5L)) {
      estimates <- vapply(seq_len(n_reps), function(r) {
        set.seed(114000L + J * 1000L + r)
        d <- dgp14t(n, J)
        fit <- sudo_ordinal(
          d, adapter = make_ordinal_propodds_adapter(mstop, nu),
          B = B, n_folds = 5L
        )
        fit$theta
      }, numeric(1))
      rows[[length(rows) + 1L]] <- data.frame(
        mstop = mstop, nu = nu, J = J, n_reps = n_reps,
        mean_estimate = mean(estimates), bias = mean(estimates) - 1,
        mc_se_bias = sd(estimates) / sqrt(n_reps), sd = sd(estimates)
      )
    }
  }
}
screen <- do.call(rbind, rows)
aggregate_score <- aggregate(abs(bias) ~ mstop + nu, screen, max)
selected <- aggregate_score[which.min(aggregate_score$`abs(bias)`), ]
selected$criterion <- "minimize maximum absolute target-level bias across J"
selected$tuning_seed_start <- 114000L
selected$screen_n <- n
selected$screen_reps <- n_reps
selected$confirmatory_ready <- n >= 1000L && n_reps >= 20L && B >= 5L
print(screen, row.names = FALSE)
cat("\nselected\n")
print(selected, row.names = FALSE)
dir.create("R/results", showWarnings = FALSE)
suffix <- if (selected$confirmatory_ready) "" else "_smoke"
write.csv(screen, paste0("R/results/stage14t_ordinal_tuning", suffix, ".csv"),
          row.names = FALSE)
write.csv(selected, paste0("R/results/stage14t_ordinal_config", suffix, ".csv"),
          row.names = FALSE)
cat("\nTUNING PASS: configuration selected only from target-level bias on",
    "dedicated seeds\n")
