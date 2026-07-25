"""Rubin's rules pooling for surrogate-draw estimates.

Mirrors R/sudo/rubin.R exactly; tests pin both implementations to a shared
numeric fixture. CI uses the Barnard-Rubin (1999) small-sample degrees of
freedom.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist


@dataclass
class PoolResult:
    theta: float
    W: float
    B_between: float
    T: float
    se: float
    df: float
    ci_lo: float
    ci_hi: float
    B: int


def pool_rubin(theta_b, var_b, level=0.95, n_obs=np.inf, n_par=1):
    theta_b = np.asarray(theta_b, dtype=float)
    var_b = np.asarray(var_b, dtype=float)
    B = len(theta_b)
    if B < 2 or len(var_b) != B:
        raise ValueError("need B >= 2 draws with matching variances")
    theta = theta_b.mean()
    W = var_b.mean()
    B_between = theta_b.var(ddof=1)
    T = W + (1 + 1 / B) * B_between

    r = (1 + 1 / B) * B_between / W
    df_old = (B - 1) * (1 + 1 / r) ** 2
    if np.isfinite(n_obs):
        df_com = n_obs - n_par
        lam = (1 + 1 / B) * B_between / T
        df_obs = (df_com + 1) / (df_com + 3) * df_com * (1 - lam)
        df = 1 / (1 / df_old + 1 / df_obs)
    else:
        df = df_old

    q = t_dist.ppf(1 - (1 - level) / 2, df)
    se = np.sqrt(T)
    return PoolResult(theta=theta, W=W, B_between=B_between, T=T, se=se,
                      df=df, ci_lo=theta - q * se, ci_hi=theta + q * se, B=B)
