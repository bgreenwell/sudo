"""Parametric full (imputation) models of Y on (D, X).

Fit once on all data — the validated stage-2/4 design: a low-dimensional
parametric model does not overfit, and proper-MI draws must perturb the
parameter vector as a whole (fold-wise independent perturbations average
the between-draw variance down and under-cover; see parity_check.py).
Flexible full models require cross-fitting (R stage 3) and are not ported
yet. Binary uses statsmodels GLM (logit or cloglog); ordinal uses
OrderedModel (logit), whose covariance includes the thresholds.
"""

import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel


class FullModel:
    """Fitted full model with proper-draw support.

    v_hat      latent index at the fitted parameters
    cutpoints  (J+1)-vector including +-inf, at the fitted parameters
    draw(rng)  (v_b, cutpoints_b) with parameters redrawn from
               N(params, cov)
    """

    def __init__(self, design, params, cov, index_fn, cutpoint_fn):
        self._design = design
        self._params = params
        self._cov = cov
        self._index_fn = index_fn
        self._cutpoint_fn = cutpoint_fn
        self.v_hat = index_fn(design, params)
        self.cutpoints = cutpoint_fn(params)

    def draw(self, rng):
        par = rng.multivariate_normal(self._params, self._cov)
        return self._index_fn(self._design, par), self._cutpoint_fn(par)


def fit_binary(y, D, X, link="logit"):
    """GLM full model for binary y; latent cutpoints are (-inf, 0, inf)."""
    fam_link = (sm.families.links.CLogLog() if link == "cloglog"
                else sm.families.links.Logit())
    design = np.column_stack([np.ones(len(y)), D, X])
    res = sm.GLM(y, design, family=sm.families.Binomial(link=fam_link)).fit()
    cut = np.array([-np.inf, 0.0, np.inf])
    return FullModel(design, res.params, res.cov_params(),
                     lambda d, p: d @ p, lambda p: cut)


def fit_ordinal(y, D, X):
    """OrderedModel (logit) full model for ordinal y coded 1..J.

    Parameterization: P(Y<=j) = F(c_j - x'beta), latent U = x'beta + e.
    The threshold transform is applied per parameter draw, so proper draws
    perturb thresholds and coefficients jointly.
    """
    y = np.asarray(y, dtype=int)
    exog = np.column_stack([D, X])
    k = exog.shape[1]
    model = OrderedModel(y, exog, distr="logit")
    res = model.fit(method="bfgs", disp=False)

    def index_fn(design, params):
        return design @ params[:k]

    def cutpoint_fn(params):
        return model.transform_threshold_params(params)

    return FullModel(exog, res.params, res.cov_params(), index_fn,
                     cutpoint_fn)
