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

# truncated logistic(location, 1) on (lo, hi]
rlogis_trunc <- function(loc, lo, hi) {
  u <- runif(length(loc))
  qlogis(.clamp_p(u * (plogis(hi, location = loc) - plogis(lo, location = loc)) +
                  plogis(lo, location = loc)), location = loc)
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
                               link = c("logit", "cloglog", "cloglog_min")) {
  link <- match.arg(link)
  stopifnot(min(y) >= 1, max(y) <= length(cutpoints) - 1)
  lo <- cutpoints[y]
  hi <- cutpoints[y + 1]
  S <- switch(link,
              logit       = rlogis_trunc(v_hat, lo, hi),
              cloglog     = rgumbel_trunc(v_hat, lo, hi),
              cloglog_min = rgumbel_min_trunc(v_hat, lo, hi))
  # if both cumulative bounds underflow past the clamp the draw can land
  # outside its interval; clip back (the observed category has ~zero mass
  # under the index there)
  pmin(pmax(S, lo + 1e-9), hi)
}
