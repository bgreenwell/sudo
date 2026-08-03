# Surrogate completion: draw a continuous latent utility S consistent with the
# observed discrete Y, by inverse-transform sampling from the assumed error
# distribution truncated to the interval Y implies.
#
# Binary Y in {0,1} is the J=2 case with cutpoints (-Inf, 0, Inf).
# Ordinal Y in {1..J} uses cutpoints (-Inf, c_1, ..., c_{J-1}, Inf) and the
# interval (c_{Y-1}, c_Y].

# clamp cumulative probabilities away from {0, 1}: when the observed category
# has ~zero mass under a (perturbed) index, the truncated quantile would
# otherwise return +-Inf
.clamp_p <- function(p) pmin(pmax(p, 1e-12), 1 - 1e-12)

# General discrete-outcome completion. A fitted family supplies the
# randomized probability-integral-transform interval
#   [P(Y < y | D, X), P(Y <= y | D, X)].
# The reference law only sets the completed outcome's noise scale. Under a
# correct fitted conditional distribution the randomized PIT is Uniform(0,1),
# so any mean-zero reference law preserves E[S | D, X] = index.
complete_discrete <- function(lower, upper, index,
                              reference = c("logistic", "normal",
                                            "gumbel_max", "gumbel_min"),
                              u = NULL) {
  reference <- match.arg(reference)
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  index <- as.numeric(index)
  n <- length(index)
  stopifnot(length(lower) == n, length(upper) == n,
            all(is.finite(index)), all(is.finite(lower)),
            all(is.finite(upper)), all(lower >= 0), all(upper <= 1),
            all(lower <= upper))
  if (is.null(u)) u <- runif(n)
  stopifnot(length(u) == n, all(is.finite(u)), all(u >= 0), all(u <= 1))
  p <- .clamp_p(lower + u * (upper - lower))
  # Extreme-value link errors are centered before use. Their conventional
  # location-zero laws have means +gamma (max) and -gamma (min); subtracting
  # those constants preserves every CDF jump and variance while making the
  # reference contribution mean zero as required by the common interface.
  euler_gamma <- -digamma(1)
  error <- switch(
    reference,
    logistic = qlogis(p),
    normal = qnorm(p),
    gumbel_max = qgumbel(p) - euler_gamma,
    gumbel_min = qgumbel_min(p) + euler_gamma
  )
  out <- index + error
  if (!all(is.finite(out))) stop("non-finite discrete surrogate draw")
  out
}

# Conditional mean of the standard-normal reference quantile over one CDF
# jump. This is the Rao-Blackwell counterpart of complete_discrete(...,
# reference = "normal"). The boundary terms reproduce .clamp_p() exactly,
# including point masses induced when an interval reaches 0 or 1.
normal_jump_mean <- function(lower, upper, clamp = 1e-12) {
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  stopifnot(length(lower) == length(upper), all(is.finite(lower)),
            all(is.finite(upper)), all(lower >= 0), all(upper <= 1),
            all(lower <= upper), clamp > 0, clamp < 0.5)
  width <- upper - lower
  midpoint <- (lower + upper) / 2
  out <- qnorm(pmin(pmax(midpoint, clamp), 1 - clamp))
  # The midpoint is the continuous limit and avoids cancellation when a CDF
  # jump is positive but narrower than floating-point quadrature can resolve.
  regular <- width > sqrt(.Machine$double.eps)
  if (any(regular)) {
    lo <- pmax(lower[regular], clamp)
    hi <- pmin(upper[regular], 1 - clamp)
    interior <- pmax(hi - lo, 0)
    integral <- numeric(length(lo))
    has_interior <- interior > 0
    integral[has_interior] <-
      dnorm(qnorm(lo[has_interior])) - dnorm(qnorm(hi[has_interior]))
    q_lo <- qnorm(clamp)
    q_hi <- qnorm(1 - clamp)
    lower_mass <- pmax(pmin(upper[regular], clamp) - lower[regular], 0)
    upper_mass <- pmax(upper[regular] - pmax(lower[regular], 1 - clamp), 0)
    out[regular] <- (lower_mass * q_lo + integral + upper_mass * q_hi) /
      width[regular]
  }
  if (!all(is.finite(out))) stop("non-finite analytic normal completion")
  out
}

# Convert category probabilities to randomized-PIT bounds. `y` uses integer
# codes 1,...,J and `prob` has one column per category.
categorical_cdf_bounds <- function(y, prob) {
  y <- as.integer(y)
  prob <- as.matrix(prob)
  stopifnot(nrow(prob) == length(y), min(y) >= 1L, max(y) <= ncol(prob),
            all(is.finite(prob)), all(prob >= 0))
  prob <- prob / rowSums(prob)
  cumulative <- t(apply(prob, 1L, cumsum))
  row <- seq_along(y)
  upper <- cumulative[cbind(row, y)]
  lower <- upper - prob[cbind(row, y)]
  list(lower = pmax(0, lower), upper = pmin(1, upper))
}

# truncated logistic(location, 1) on (lo, hi]
rlogis_trunc <- function(loc, lo, hi) {
  u <- runif(length(loc))
  qlogis(.clamp_p(u * (plogis(hi, location = loc) - plogis(lo, location = loc)) +
                  plogis(lo, location = loc)), location = loc)
}

# truncated standard normal (error law behind the probit link). This is also
# the Albert-Chib data-augmentation draw, so for a probit imputation model the
# Gibbs augmentation variable and the SUDO surrogate are the same object.
rnorm_trunc <- function(loc, lo, hi) {
  u <- runif(length(loc))
  qnorm(.clamp_p(u * (pnorm(hi, mean = loc) - pnorm(lo, mean = loc)) +
                 pnorm(lo, mean = loc)), mean = loc)
}

# truncated Gumbel (max convention; error law behind the cloglog link)
# cdf F(x) = exp(-exp(-(x - loc)))
pgumbel <- function(q, loc = 0) exp(-exp(-(q - loc)))
qgumbel <- function(p, loc = 0) loc - log(-log(p))

rgumbel_trunc <- function(loc, lo, hi) {
  u <- runif(length(loc))
  qgumbel(.clamp_p(u * (pgumbel(hi, loc) - pgumbel(lo, loc)) +
                   pgumbel(lo, loc)), loc)
}

# truncated Gumbel (min convention). Sign matters: binary glm cloglog implies
# a Gumbel-MAX latent error (P(Y=1) = 1 - exp(-exp(eta))), while ordinal clm
# link="cloglog" implies Gumbel-MIN (P(Y<=j) = 1 - exp(-exp(alpha_j - eta))).
# cdf F(x) = 1 - exp(-exp(x - loc))
pgumbel_min <- function(q, loc = 0) 1 - exp(-exp(q - loc))
qgumbel_min <- function(p, loc = 0) loc + log(-log(1 - p))

rgumbel_min_trunc <- function(loc, lo, hi) {
  u <- runif(length(loc))
  qgumbel_min(.clamp_p(u * (pgumbel_min(hi, loc) - pgumbel_min(lo, loc)) +
                       pgumbel_min(lo, loc)), loc)
}

# y: integer codes 1..J (binary: use y = Y + 1 with J = 2)
# v_hat: latent index estimate (link scale), cutpoints: length J+1 incl. +-Inf
complete_surrogate <- function(y, v_hat, cutpoints = c(-Inf, 0, Inf),
                               link = c("logit", "probit", "cloglog",
                                        "cloglog_min")) {
  link <- match.arg(link)
  stopifnot(min(y) >= 1, max(y) <= length(cutpoints) - 1)
  lo <- cutpoints[y]
  hi <- cutpoints[y + 1]
  S <- switch(link,
              logit       = rlogis_trunc(v_hat, lo, hi),
              probit      = rnorm_trunc(v_hat, lo, hi),
              cloglog     = rgumbel_trunc(v_hat, lo, hi),
              cloglog_min = rgumbel_min_trunc(v_hat, lo, hi))
  # if both cumulative bounds underflow past the clamp the draw can land
  # outside its interval; clip back (the observed category has ~zero mass
  # under the index there)
  pmin(pmax(S, lo + 1e-9), hi)
}
