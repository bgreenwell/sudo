import numpy as np

from sudo import (generate_binary, generate_binary_cloglog,
                  generate_binary_nonlinear, generate_ordinal)


def test_shapes_and_codes():
    X, D, y, U = generate_binary(500, random_seed=1)
    assert X.shape == (500, 1) and set(np.unique(y)) <= {0, 1}
    X, D, y, U = generate_binary_nonlinear(500, random_seed=1)
    assert X.shape == (500, 5)
    X, D, y, U = generate_ordinal(500, random_seed=1)
    assert set(np.unique(y)) <= {1, 2, 3}
    X, D, y, U = generate_binary_cloglog(500, random_seed=1)
    assert set(np.unique(y)) <= {0, 1}


def test_latent_consistency():
    X, D, y, U = generate_binary(2000, random_seed=2)
    assert np.array_equal(y, (U > 0).astype(int))
    X, D, y, U = generate_ordinal(2000, random_seed=2)
    assert np.array_equal(y, 1 + np.searchsorted([-1.0, 1.0], U))


def test_reproducible():
    a = generate_binary(100, random_seed=42)
    b = generate_binary(100, random_seed=42)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
