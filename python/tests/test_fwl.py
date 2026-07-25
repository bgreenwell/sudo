"""Stage-0a exactness: FWL residual-on-residual equals the OLS coefficient."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from sudo.fwl import crossfit, fwl_theta, make_folds


def test_fwl_matches_ols():
    rng = np.random.default_rng(1)
    n = 500
    X = rng.normal(size=(n, 3))
    D = 0.5 * X[:, 0] + rng.normal(size=n)
    Y = 1.5 * D + X @ np.array([1.0, -1.0, 0.5]) + rng.normal(size=n)

    Z = np.column_stack([np.ones(n), D, X])
    beta = np.linalg.lstsq(Z, Y, rcond=None)[0]

    lr = LinearRegression()
    s_res = Y - lr.fit(X, Y).predict(X)
    d_res = D - lr.fit(X, D).predict(X)
    theta, _ = fwl_theta(s_res, d_res)
    assert theta == pytest.approx(beta[1], abs=1e-10)


def test_crossfit_out_of_fold():
    rng = np.random.default_rng(2)
    n = 200
    X = rng.normal(size=(n, 2))
    y = X @ np.array([1.0, 2.0]) + rng.normal(size=n)
    folds = make_folds(n, 5, rng)
    assert sorted(np.concatenate(folds).tolist()) == list(range(n))
    pred = crossfit(X, y, LinearRegression(), folds)
    assert np.corrcoef(pred, y)[0, 1] > 0.8
