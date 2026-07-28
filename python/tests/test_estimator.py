"""Seeded smoke MCs for SudoDML: binary and ordinal, linear nuisances."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from sudo import SudoDML, generate_binary, generate_ordinal
from sudo.mc import run_mc, summarize_mc

LIN = dict(learner_s=LinearRegression(), learner_d=LinearRegression())


def test_binary_smoke_mc():
    theta = 1.5
    df = run_mc(
        20,
        lambda seed: generate_binary(2000, theta=theta, random_seed=seed),
        lambda d: SudoDML(B=15, random_state=7, **LIN).fit(d[0], d[1], d[2]),
        theta, seed=10)
    s = summarize_mc(df)
    assert abs(s["bias"]) < 3 * s["mc_se_bias"]
    assert s["coverage"] >= 0.8


def test_ordinal_smoke_mc():
    theta = 1.0
    df = run_mc(
        10,
        lambda seed: generate_ordinal(1500, theta=theta, random_seed=seed),
        lambda d: SudoDML(B=10, random_state=7, **LIN).fit(d[0], d[1], d[2]),
        theta, seed=20)
    s = summarize_mc(df)
    assert abs(s["bias"]) < 3 * s["mc_se_bias"]
    assert s["coverage"] >= 0.7


def test_ordinal_cloglog_unsupported():
    X, D, y, _ = generate_ordinal(300, random_seed=1)
    with pytest.raises(NotImplementedError):
        SudoDML(link="cloglog", B=2, **LIN).fit(X, D, y)


def test_ordinal_refits_outcome_nuisance_by_default(monkeypatch):
    import sudo.estimator as estimator_module

    calls = 0
    original = estimator_module.crossfit

    def counted_crossfit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator_module, "crossfit", counted_crossfit)
    X, D, y, _ = generate_ordinal(300, random_seed=1)
    SudoDML(B=3, n_folds=3, random_state=1, **LIN).fit(X, D, y)

    # One treatment nuisance, one initial outcome nuisance, and one outcome
    # nuisance for each of the three ordinal completions.
    assert calls == 5


def test_custom_sklearn_learner_accepted():
    # a passed unfitted sklearn estimator that defines __len__ (e.g. RF)
    # must not be truthiness-tested; regression guard for the `or default`
    # bug that raised AttributeError on estimators_
    from sklearn.ensemble import RandomForestRegressor
    X, D, y, _ = generate_binary(400, random_seed=1)
    m = SudoDML(B=5, random_state=1,
                learner_s=RandomForestRegressor(n_estimators=20),
                learner_d=RandomForestRegressor(n_estimators=20)).fit(X, D, y)
    assert np.isfinite(m.theta_) and np.isfinite(m.se_)
