# Theory check: randomized-PIT completion for arbitrary discrete outcomes.
#
# Under the correct conditional distribution, drawing U uniformly inside the
# observed CDF jump makes U | D,X marginally Uniform(0,1). A mean-zero
# reference quantile therefore gives E[index + H^-1(U) | D,X] = index.
# This script checks that identity for Bernoulli, ordinal, Poisson, and
# negative-binomial laws, including zeros and tail CDF bounds.

source("R/sudo/surrogate.R")

set.seed(16001)
n <- as.integer(Sys.getenv("SUDO_THEORY_DISCRETE_N", unset = "200000"))
eta <- runif(n, -2, 2)

check <- function(name, y, lower, upper, reference, tolerance = 0.012) {
  s <- complete_discrete(lower, upper, eta, reference)
  err <- s - eta
  se <- sd(err) / sqrt(n)
  bias <- mean(err)
  stopifnot(all(is.finite(s)), abs(bias) <= max(4 * se, tolerance))
  data.frame(family = name, mean_reference_error = bias, mc_se = se,
             min_width = min(upper - lower), max_abs_surrogate = max(abs(s)))
}

p <- plogis(eta)
y_binary <- rbinom(n, 1, p)
binary <- check(
  "binary-logit", y_binary,
  ifelse(y_binary == 0L, 0, 1 - p),
  ifelse(y_binary == 0L, 1 - p, 1), "logistic"
)

p_cloglog <- -expm1(-exp(eta))
y_cloglog <- rbinom(n, 1, p_cloglog)
binary_cloglog <- check(
  "binary-cloglog", y_cloglog,
  ifelse(y_cloglog == 0L, 0, 1 - p_cloglog),
  ifelse(y_cloglog == 0L, 1 - p_cloglog, 1), "gumbel_max"
)

cuts <- c(-1, 0.5, 2)
latent <- eta + rlogis(n)
y_ordinal <- 1L + findInterval(latent, cuts)
lo_cut <- c(-Inf, cuts)[y_ordinal]
hi_cut <- c(cuts, Inf)[y_ordinal]
ordinal <- check("ordinal-logit", y_ordinal, plogis(lo_cut - eta),
                 plogis(hi_cut - eta), "logistic")

latent_min <- eta + qgumbel_min(runif(n))
y_ordinal_min <- 1L + findInterval(latent_min, cuts)
lo_min <- c(-Inf, cuts)[y_ordinal_min]
hi_min <- c(cuts, Inf)[y_ordinal_min]
ordinal_cloglog <- check(
  "ordinal-cloglog", y_ordinal_min, pgumbel_min(lo_min - eta),
  pgumbel_min(hi_min - eta), "gumbel_min"
)

mu <- exp(eta)
y_poisson <- rpois(n, mu)
poisson <- check("poisson", y_poisson, ppois(y_poisson - 1L, mu),
                 ppois(y_poisson, mu), "normal")

size <- 2
y_negbin <- rnbinom(n, mu = mu, size = size)
negbin <- check("negative-binomial", y_negbin,
                pnbinom(y_negbin - 1L, mu = mu, size = size),
                pnbinom(y_negbin, mu = mu, size = size), "normal")

# Mechanical edge cases: zero counts, large counts, zero-width intervals from
# floating-point saturation, and exact boundary probabilities must stay finite.
edge <- complete_discrete(
  lower = c(0, 1 - 1e-15, 1, 0),
  upper = c(1e-300, 1, 1, 0),
  index = c(-30, 30, 0, 0), reference = "normal",
  u = c(0, 1, 0.5, 0.5)
)
stopifnot(all(is.finite(edge)))

result <- rbind(binary, binary_cloglog, ordinal, ordinal_cloglog,
                poisson, negbin)
dir.create("R/results", showWarnings = FALSE)
write.csv(result, "R/results/theory_discrete_completion.csv", row.names = FALSE)
print(format(result, digits = 4), row.names = FALSE)
cat("\nPASS: randomized-PIT completion is centered for all tested laws and",
    "all edge draws are finite\n")
