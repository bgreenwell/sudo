"""Surrogate completion: draw a continuous latent utility consistent with the
observed discrete outcome by inverse-transform sampling from the assumed
error law truncated to the interval the category implies.

Binary y in {0,1} maps to codes {1,2} with cutpoints (-inf, 0, inf); ordinal
y uses codes 1..J with cutpoints (-inf, c_1, .., c_{J-1}, inf).
"""

import numpy as np

from .links import get_link

# clamp cumulative probabilities away from {0,1}: a perturbed index can give
# the observed category ~zero mass, and the quantile would return +-inf
_P_EPS = 1e-12


def complete_surrogate(y, v_hat, cutpoints=(-np.inf, 0.0, np.inf),
                       link="logit", rng=None):
    """y: integer codes 1..J; v_hat: latent index on the link scale."""
    rng = np.random.default_rng(rng)
    link = get_link(link)
    y = np.asarray(y, dtype=int)
    v_hat = np.asarray(v_hat, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    if y.min() < 1 or y.max() > len(cutpoints) - 1:
        raise ValueError("y codes must lie in 1..J matching cutpoints")
    lo = cutpoints[y - 1] - v_hat
    hi = cutpoints[y] - v_hat
    p_lo = link.cdf(lo)
    p_hi = link.cdf(hi)
    u = rng.uniform(size=len(y))
    p = np.clip(u * (p_hi - p_lo) + p_lo, _P_EPS, 1 - _P_EPS)
    s = v_hat + link.ppf(p)
    # if both cumulative bounds underflow past the clamp, the draw can land
    # outside its interval; clip back (the observed category has ~zero mass
    # under the index there, so the boundary is the honest completion)
    return np.clip(s, cutpoints[y - 1] + 1e-9, cutpoints[y])
