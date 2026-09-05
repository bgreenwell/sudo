import numpy as np

from binary_link_experiment import (
    METHODS,
    ExperimentConfig,
    generate_data,
    make_tables,
    run_replication,
    summarize,
)


def test_binary_link_dgp_is_deterministic_and_non_degenerate():
    first = generate_data(500, 0.8, "cloglog", 17)
    second = generate_data(500, 0.8, "cloglog", 17)
    for left, right in zip(first, second):
        assert np.array_equal(left, right)
    assert 0.2 < first[2].mean() < 0.8


def test_binary_link_replication_and_tables():
    config = ExperimentConfig(n=300, reps=1, B=3, folds=3, seed=41)
    replications = run_replication(config, "logit", 0)
    assert tuple(replications["method"]) == METHODS
    assert np.all(np.isfinite(replications[["estimate", "se", "covered"]]))
    assert np.all(replications["se"] > 0)

    summary = summarize(replications, config.theta)
    bias, coverage = make_tables(summary)
    assert len(bias) == len(METHODS)
    assert len(coverage) == len(METHODS)
    assert coverage["coverage_of_latent_theta"].between(0, 1).all()
