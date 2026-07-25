"""FWL partialling-out and K-fold cross-fitting (mirrors R/sudo/fwl.R)."""

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold


def fwl_theta(s_res, d_res):
    """Residual-on-residual estimate with sandwich variance."""
    denom = np.sum(d_res ** 2)
    theta = np.sum(s_res * d_res) / denom
    resid = s_res - theta * d_res
    var = np.sum((d_res * resid) ** 2) / denom ** 2
    return theta, var


def make_folds(n, k, rng=None):
    rng = np.random.default_rng(rng)
    kf = KFold(n_splits=k, shuffle=True,
               random_state=int(rng.integers(2 ** 31)))
    return [test for _, test in kf.split(np.empty((n, 1)))]


def crossfit(X, y, estimator, folds, proba=False):
    """Out-of-fold predictions from a sklearn-style estimator."""
    pred = np.empty(len(y), dtype=float)
    mask = np.ones(len(y), dtype=bool)
    for test in folds:
        mask[:] = True
        mask[test] = False
        model = clone(estimator)
        model.fit(X[mask], y[mask])
        if proba:
            pred[test] = model.predict_proba(X[test])[:, 1]
        else:
            pred[test] = model.predict(X[test])
    return pred
