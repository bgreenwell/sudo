# Theory and unit checks for coefficientless count completion.
#
# Run from the repository root. The checks are deterministic and cover the
# analytic normal-reference completion, numerical edge cases, count
# dispersion, and the constrained learner's additive treatment block.

source("R/sudo/fwl.R")
source("R/sudo/surrogate.R")
source("R/sudo/rubin.R")
source("R/sudo/discrete.R")

mc_n <- as.integer(Sys.getenv("SUDO_THEORY_COUNT_N", unset = "300000"))
stopifnot(mc_n >= 10000L)

lower <- c(0, 0.1, 0.4, 0.9, 1 - 1e-14, 1, 0)
upper <- c(0.05, 0.3, 0.6, 1, 1, 1, 0)
analytic <- normal_jump_mean(lower, upper)

set.seed(151001)
mc <- vapply(seq_along(lower), function(j) {
  value <- complete_discrete(
    rep(lower[j], mc_n), rep(upper[j], mc_n), rep(0, mc_n),
    reference = "normal"
  )
  c(mean = mean(value), mc_se = sd(value) / sqrt(mc_n), finite = all(is.finite(value)))
}, numeric(3))
difference <- mc["mean", ] - analytic
stopifnot(
  all(mc["finite", ] == 1), all(is.finite(analytic)),
  all(abs(difference) <= pmax(5 * mc["mc_se", ], 0.001)),
  identical(normal_jump_mean(c(0, 1), c(0, 1)),
            qnorm(c(1e-12, 1 - 1e-12)))
)

# The randomized completion must be reproducible under an explicit seed.
set.seed(151002)
draw_a <- complete_discrete(lower, upper, seq_along(lower), "normal")
set.seed(151002)
draw_b <- complete_discrete(lower, upper, seq_along(lower), "normal")
stopifnot(identical(draw_a, draw_b), all(is.finite(draw_a)))

# A single training-fold negative-binomial size estimate is positive and
# recovers the generating value closely in a large oracle-mean check.
set.seed(151003)
mu <- exp(runif(10000, -1, 2))
y <- rnbinom(length(mu), mu = mu, size = 2)
size_hat <- .estimate_count_size(y, mu)
stopifnot(is.finite(size_hat), size_hat > 0, abs(size_hat / 2 - 1) < 0.1)

# The constrained XGBoost interaction groups make a fixed D contrast
# invariant to X. This is a structural check, not a predictive-quality gate.
set.seed(151004)
n <- as.integer(Sys.getenv("SUDO_THEORY_COUNT_XGB_N", unset = "400"))
X <- matrix(rnorm(n * 5), n, 5,
            dimnames = list(NULL, paste0("X", seq_len(5))))
D <- rbinom(n, 1, plogis(X[, 4] + cos(X[, 5])))
eta <- 0.5 + log(2) * D + 0.35 * (X[, 1]^2 - 1) +
  0.5 * sin(X[, 2]) + 0.25 * X[, 3]
d <- list(X = X, D = D, Y = rpois(n, exp(eta)))
folds <- make_folds(n, 3L)
fit <- fit_count_xgboost_adapter(
  d, folds, family = "poisson", constrained = TRUE,
  max_depth = 2L, min_child_weight = 5, nrounds = 30L
)
stopifnot(fit$diagnostics$max_contrast_spread < 1e-6)

# Reusing a seed reproduces both samples and folds, which is the mechanism
# used by the comparator stages to guarantee common experimental inputs.
make_seeded <- function(seed) {
  set.seed(seed)
  value <- list(sample = rnorm(50), folds = make_folds(50, 5L))
  value
}
stopifnot(identical(make_seeded(151005), make_seeded(151005)))

result <- data.frame(
  interval = seq_along(lower), lower = lower, upper = upper,
  analytic = analytic, randomized_mean = mc["mean", ],
  mc_se = mc["mc_se", ], difference = difference,
  size_hat = size_hat,
  constrained_max_contrast_spread = fit$diagnostics$max_contrast_spread
)
dir.create("R/results", showWarnings = FALSE)
write.csv(result, "R/results/theory_count_completion.csv", row.names = FALSE)
print(format(result, digits = 5), row.names = FALSE)
cat("\nPASS: analytic and randomized count completions agree; edge, seed,",
    "dispersion, and constrained-treatment checks pass\n")
