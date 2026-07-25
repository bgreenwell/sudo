"""Latent-PLR data generators mirroring the R stage DGPs."""

import numpy as np
from scipy.special import expit


def _gm(X):
    g = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.5 * X[:, 2]
    m = X[:, 3] + np.cos(X[:, 4])
    return g, m


def generate_binary(n, theta=1.5, beta=1.0, random_seed=None):
    """Stage 1/2 DGP: one covariate, logistic errors, linear index."""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=n)
    D = rng.binomial(1, expit(0.8 * X)).astype(float)
    U = theta * D + beta * X + rng.logistic(size=n)
    return X[:, None], D, (U > 0).astype(int), U


def generate_binary_nonlinear(n, theta=1.0, random_seed=None):
    """Stage 3 DGP: p=5, nonlinear g and propensity, logistic errors."""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=(n, 5))
    g, m = _gm(X)
    D = rng.binomial(1, expit(m)).astype(float)
    U = theta * D + g + rng.logistic(size=n)
    return X, D, (U > 0).astype(int), U


def generate_ordinal(n, theta=1.0, beta=1.0, cuts=(-1.0, 1.0),
                     random_seed=None):
    """Stage 4 DGP: ordinal J=3, logistic errors, linear index; y in 1..3."""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=n)
    D = rng.binomial(1, expit(0.8 * X)).astype(float)
    U = theta * D + beta * X + rng.logistic(size=n)
    y = 1 + np.searchsorted(np.asarray(cuts), U)
    return X[:, None], D, y, U


def generate_binary_cloglog(n, theta=1.0, beta=1.0, random_seed=None):
    """Stage 6A DGP: Gumbel-max errors — glm cloglog is the correct link."""
    rng = np.random.default_rng(random_seed)
    X = rng.normal(size=n)
    D = rng.binomial(1, expit(0.8 * X)).astype(float)
    U = theta * D + beta * X - np.log(-np.log(rng.uniform(size=n)))
    return X[:, None], D, (U > 0).astype(int), U
