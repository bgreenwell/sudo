"""Monte Carlo harness mirroring R/sudo/mc.R."""

import numpy as np
import pandas as pd


def run_mc(n_reps, dgp, estimate, theta_true, seed=1):
    """dgp(seed) -> data tuple; estimate(data) -> object with theta_, se_,
    ci_. Returns one row per rep."""
    rows = []
    for r in range(1, n_reps + 1):
        est = estimate(dgp(seed + r))
        lo, hi = est.ci_
        rows.append(dict(rep=r, est=est.theta_, se=est.se_, ci_lo=lo,
                         ci_hi=hi, covered=lo <= theta_true <= hi))
    df = pd.DataFrame(rows)
    df.attrs["theta_true"] = theta_true
    return df


def summarize_mc(df, theta_true=None):
    if theta_true is None:
        theta_true = df.attrs["theta_true"]
    n = len(df)
    bias = df["est"].mean() - theta_true
    sd = df["est"].std(ddof=1)
    cov = df["covered"].mean()
    return pd.Series(dict(
        n_reps=n, mean_est=df["est"].mean(), bias=bias,
        mc_se_bias=sd / np.sqrt(n), sd=sd,
        rmse=np.sqrt(((df["est"] - theta_true) ** 2).mean()),
        mean_se=df["se"].mean(), coverage=cov,
        mc_se_cov=np.sqrt(cov * (1 - cov) / n)))
