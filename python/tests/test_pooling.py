"""Pin pool_rubin to the shared R fixture (R/sudo/rubin.R, same inputs)."""

import numpy as np
import pytest

from sudo.pooling import pool_rubin

THETA_B = [1.02, 0.97, 1.05, 0.99, 1.11, 0.93, 1.04, 1.00, 0.96, 1.08]
VAR_B = [0.010, 0.012, 0.011, 0.009, 0.013, 0.010, 0.011, 0.012, 0.009,
         0.010]

# values produced by R: pool_rubin(theta_b, var_b, n_obs = 5000, n_par = 2)
R_FIXTURE = dict(
    theta=1.015,
    W=0.0107,
    B_between=0.00313888888888889,
    T=0.0141527777777778,
    se=0.118965447831619,
    df=145.392258674338,
    ci_lo=0.779874940956283,
    ci_hi=1.25012505904372,
)


def test_matches_r_fixture():
    p = pool_rubin(THETA_B, VAR_B, n_obs=5000, n_par=2)
    for key, val in R_FIXTURE.items():
        tol = 1e-9 if key.startswith("ci") else 1e-12
        assert getattr(p, key) == pytest.approx(val, rel=tol), key


def test_total_variance_decomposition():
    p = pool_rubin(THETA_B, VAR_B)
    assert p.T == pytest.approx(p.W + (1 + 1 / p.B) * p.B_between)
    assert p.ci_lo < p.theta < p.ci_hi


def test_rejects_single_draw():
    with pytest.raises(ValueError):
        pool_rubin([1.0], [0.1])


def test_infinite_n_obs_uses_old_df():
    p_inf = pool_rubin(THETA_B, VAR_B)
    p_fin = pool_rubin(THETA_B, VAR_B, n_obs=5000)
    assert p_fin.df < p_inf.df
