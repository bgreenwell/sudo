"""The SUDO estimator: surrogate completion + FWL partialling-out + proper-MI
Rubin pooling. Mirrors the validated R ladder (stages 2-5)."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from . import fullmodel
from .fwl import crossfit, fwl_theta, make_folds
from .pooling import pool_rubin
from .surrogate import complete_surrogate


class SudoDML:
    """Surrogate-Assisted DML for binary (y in {0,1}) or ordinal (y in 1..J,
    J >= 3) outcomes.

    The full (imputation) model of y on (D, X) is parametric: GLM logit or
    cloglog for binary, OrderedModel logit for ordinal. Nuisances E[S|X] and
    E[D|X] are pluggable sklearn regressors, cross-fitted. Each of the B
    surrogate draws redraws the full-model parameters (thresholds included)
    from their asymptotic posterior — proper multiple imputation — and the
    draws are pooled with Rubin's rules (Barnard-Rubin t CI).

    Attributes after fit: theta_, se_, ci_, df_, theta_b_.
    """

    def __init__(self, link="logit", n_folds=5, B=25, learner_s=None,
                 learner_d=None, level=0.95, random_state=None):
        self.link = link
        self.n_folds = n_folds
        self.B = B
        self.learner_s = learner_s
        self.learner_d = learner_d
        self.level = level
        self.random_state = random_state

    def fit(self, X, D, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        D = np.asarray(D, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        folds = make_folds(n, self.n_folds, rng)

        binary = y.max() <= 1
        codes = y + 1 if binary else y
        if binary:
            fm = fullmodel.fit_binary(y, D, X, link=self.link)
        else:
            if self.link != "logit":
                raise NotImplementedError(
                    "ordinal full model supports logit only; use the R "
                    "prototype (ordinal::clm) for cloglog")
            fm = fullmodel.fit_ordinal(y, D, X)

        # `x or default` would call __len__/__bool__ on a passed sklearn
        # estimator (raises when unfitted); use an explicit None check
        learner_s = self.learner_s if self.learner_s is not None else \
            RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                  n_jobs=-1,
                                  random_state=int(rng.integers(2 ** 31)))
        learner_d = self.learner_d if self.learner_d is not None else \
            RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                  n_jobs=-1,
                                  random_state=int(rng.integers(2 ** 31)))

        d_res = D - crossfit(X, D, learner_d, folds)
        s_init = complete_surrogate(codes, fm.v_hat, fm.cutpoints,
                                    self.link, rng)
        s_hat = crossfit(X, s_init, learner_s, folds)

        theta_b = np.empty(self.B)
        var_b = np.empty(self.B)
        for b in range(self.B):
            v_b, cuts_b = fm.draw(rng)
            s_b = complete_surrogate(codes, v_b, cuts_b, self.link, rng)
            theta_b[b], var_b[b] = fwl_theta(s_b - s_hat, d_res)

        pool = pool_rubin(theta_b, var_b, level=self.level, n_obs=n)
        self.theta_ = pool.theta
        self.se_ = pool.se
        self.ci_ = (pool.ci_lo, pool.ci_hi)
        self.df_ = pool.df
        self.theta_b_ = theta_b
        return self
