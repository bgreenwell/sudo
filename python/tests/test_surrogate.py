import numpy as np
import pytest

from sudo.surrogate import complete_surrogate


def test_binary_truncation_bounds():
    rng = np.random.default_rng(1)
    n = 5000
    y = rng.integers(0, 2, size=n)
    v = rng.normal(size=n)
    s = complete_surrogate(y + 1, v, (-np.inf, 0.0, np.inf), "logit", rng)
    assert np.all(s[y == 1] > 0)
    assert np.all(s[y == 0] <= 0)
    assert np.all(np.isfinite(s))


def test_ordinal_interval_containment():
    rng = np.random.default_rng(2)
    n = 5000
    cuts = np.array([-np.inf, -1.0, 1.0, np.inf])
    y = rng.integers(1, 4, size=n)
    v = rng.normal(size=n)
    for link in ("logit", "cloglog", "cloglog_min"):
        s = complete_surrogate(y, v, cuts, link, rng)
        assert np.all(s > cuts[y - 1])
        assert np.all(s <= cuts[y])
        assert np.all(np.isfinite(s))


def test_extreme_index_stays_finite():
    # observed category has ~zero mass under the index: clamp must hold
    rng = np.random.default_rng(3)
    y = np.array([2, 1])
    v = np.array([-40.0, 40.0])
    s = complete_surrogate(y, v, (-np.inf, 0.0, np.inf), "logit", rng)
    assert np.all(np.isfinite(s))


def test_marginal_law_recovered():
    # untruncated check: with y always the only category, draws follow the
    # link law located at v
    rng = np.random.default_rng(4)
    n = 200_000
    v = np.full(n, 0.7)
    s = complete_surrogate(np.ones(n, dtype=int), v,
                           (-np.inf, np.inf), "logit", rng)
    assert s.mean() == pytest.approx(0.7, abs=0.02)
    assert s.std() == pytest.approx(np.pi / np.sqrt(3), abs=0.02)


def test_bad_codes_rejected():
    with pytest.raises(ValueError):
        complete_surrogate([0, 1], [0.0, 0.0], (-np.inf, 0, np.inf))
