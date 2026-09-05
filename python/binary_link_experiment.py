"""Binary-link comparison for a latent partially linear coefficient.

The experiment changes only the response link.  It compares raw-label PLR,
DoubleML's partially logistic estimator, logit and cloglog SUDO, and the
correctly specified GLM.  Run from ``python/`` with ``uv``.

Full run:
    uv run python binary_link_experiment.py

Smoke run:
    uv run python binary_link_experiment.py --n 500 --reps 4 --B 3
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import warnings

import doubleml as dml
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from sudo import SudoDML


METHODS = (
    "raw_label_plr",
    "logistic_dml",
    "sudo_logit",
    "sudo_cloglog",
    "correct_link_glm",
)

METHOD_LABELS = {
    "raw_label_plr": "Raw-label DML PLR",
    "logistic_dml": "Partially logistic DML",
    "sudo_logit": "SUDO (logit)",
    "sudo_cloglog": "SUDO (cloglog)",
    "correct_link_glm": "Correct-link GLM",
}

ASSUMPTIONS = {
    "raw_label_plr": "continuous-outcome PLR",
    "logistic_dml": "logit",
    "sudo_logit": "logit",
    "sudo_cloglog": "cloglog",
    "correct_link_glm": "true link",
}


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 2000
    reps: int = 300
    B: int = 25
    folds: int = 5
    theta: float = 0.8
    seed: int = 260817


def generate_data(n: int, theta: float, truth: str, seed: int):
    """Generate a low-dimensional, correctly specified latent PL model."""
    if truth not in {"logit", "cloglog"}:
        raise ValueError("truth must be 'logit' or 'cloglog'")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 1))
    D = 0.5 * X[:, 0] + rng.normal(scale=0.7, size=n)
    intercept = 0.0 if truth == "logit" else np.log(np.log(2.0))
    eta = intercept + theta * D + 0.5 * X[:, 0]
    if truth == "logit":
        probability = expit(eta)
    else:
        probability = -np.expm1(-np.exp(np.minimum(eta, 30.0)))
    y = rng.binomial(1, probability)
    return X, D, y


def _smooth_regressor():
    return make_pipeline(
        PolynomialFeatures(degree=5, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1e-4),
    )


def _smooth_classifier():
    return make_pipeline(
        PolynomialFeatures(degree=5, include_bias=False),
        StandardScaler(),
        LogisticRegression(C=100.0, max_iter=2000),
    )


def _normal_result(method: str, estimate: float, se: float, theta: float):
    lo = estimate - 1.959963984540054 * se
    hi = estimate + 1.959963984540054 * se
    return {
        "method": method,
        "estimate": estimate,
        "se": se,
        "covered": float(lo <= theta <= hi),
    }


def _fit_raw_plr(X, D, y, theta, folds, split_seed):
    data = dml.DoubleMLData.from_arrays(X, y, D)
    np.random.seed(split_seed)
    fit = dml.DoubleMLPLR(
        data,
        ml_l=_smooth_regressor(),
        ml_m=LinearRegression(),
        n_folds=folds,
    ).fit()
    return _normal_result("raw_label_plr", fit.coef[0], fit.se[0], theta)


def _fit_logistic_dml(X, D, y, theta, folds, split_seed):
    data = dml.DoubleMLData.from_arrays(X, y, D)
    np.random.seed(split_seed)
    fit = dml.DoubleMLLPLR(
        data,
        ml_M=_smooth_classifier(),
        ml_t=_smooth_regressor(),
        ml_m=LinearRegression(),
        n_folds=folds,
        n_folds_inner=min(3, folds),
    ).fit()
    return _normal_result("logistic_dml", fit.coef[0], fit.se[0], theta)


def _fit_sudo(X, D, y, theta, link, folds, B, seed):
    fit = SudoDML(
        link=link,
        n_folds=folds,
        B=B,
        learner_s=LinearRegression(),
        learner_d=LinearRegression(),
        random_state=seed,
    ).fit(X, D, y)
    return {
        "method": f"sudo_{link}",
        "estimate": fit.theta_,
        "se": fit.se_,
        "covered": float(fit.ci_[0] <= theta <= fit.ci_[1]),
    }


def _fit_correct_glm(X, D, y, theta, truth):
    design = sm.add_constant(np.column_stack([D, X]), has_constant="add")
    link = (
        sm.families.links.Logit()
        if truth == "logit"
        else sm.families.links.CLogLog()
    )
    fit = sm.GLM(y, design, family=sm.families.Binomial(link=link)).fit()
    return _normal_result(
        "correct_link_glm", float(fit.params[1]), float(fit.bse[1]), theta
    )


def run_replication(config: ExperimentConfig, truth: str, rep: int):
    data_seed = config.seed + 100_000 * (truth == "cloglog") + rep
    split_seed = config.seed + 200_000 + rep
    X, D, y = generate_data(config.n, config.theta, truth, data_seed)
    rows = []
    fitters = (
        lambda: _fit_raw_plr(
            X, D, y, config.theta, config.folds, split_seed
        ),
        lambda: _fit_logistic_dml(
            X, D, y, config.theta, config.folds, split_seed
        ),
        lambda: _fit_sudo(
            X, D, y, config.theta, "logit", config.folds, config.B,
            split_seed,
        ),
        lambda: _fit_sudo(
            X, D, y, config.theta, "cloglog", config.folds, config.B,
            split_seed,
        ),
        lambda: _fit_correct_glm(X, D, y, config.theta, truth),
    )
    for fitter, method in zip(fitters, METHODS):
        try:
            row = fitter()
        except Exception as error:  # keep failures visible in the table
            warnings.warn(f"{truth} rep {rep}, {method}: {error}")
            row = {
                "method": method,
                "estimate": np.nan,
                "se": np.nan,
                "covered": np.nan,
            }
        row.update(truth=truth, replication=rep)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(replications: pd.DataFrame, theta: float):
    rows = []
    for (truth, method), group in replications.groupby(
        ["truth", "method"], sort=False
    ):
        valid = group.dropna(subset=["estimate", "se", "covered"])
        estimates = valid["estimate"].to_numpy()
        ses = valid["se"].to_numpy()
        coverage = valid["covered"].mean()
        n_valid = len(valid)
        empirical_sd = estimates.std(ddof=1) if n_valid > 1 else np.nan
        rows.append({
            "truth": truth,
            "method": method,
            "n_valid": n_valid,
            "failure_rate": 1.0 - n_valid / len(group),
            "mean_estimate": estimates.mean() if n_valid else np.nan,
            "bias_vs_latent_theta": (
                estimates.mean() - theta if n_valid else np.nan
            ),
            "mc_se_bias": (
                empirical_sd / np.sqrt(n_valid) if n_valid > 1 else np.nan
            ),
            "empirical_sd": empirical_sd,
            "mean_se": ses.mean() if n_valid else np.nan,
            "coverage_of_latent_theta": coverage,
            "mc_se_coverage": (
                np.sqrt(coverage * (1.0 - coverage) / n_valid)
                if n_valid else np.nan
            ),
        })
    summary = pd.DataFrame(rows)
    order = {method: i for i, method in enumerate(METHODS)}
    summary["method_order"] = summary["method"].map(order)
    return summary.sort_values(["truth", "method_order"]).drop(
        columns="method_order"
    )


def make_tables(summary: pd.DataFrame):
    summary = summary.copy()
    summary["estimator"] = summary["method"].map(METHOD_LABELS)
    summary["assumption"] = summary["method"].map(ASSUMPTIONS)
    bias = summary[[
        "truth", "estimator", "assumption", "n_valid", "failure_rate",
        "mean_estimate",
        "bias_vs_latent_theta", "mc_se_bias",
    ]].copy()
    coverage = summary[[
        "truth", "estimator", "assumption", "n_valid", "failure_rate",
        "empirical_sd", "mean_se", "coverage_of_latent_theta",
        "mc_se_coverage",
    ]].copy()
    return bias, coverage


def _print_table(title: str, table: pd.DataFrame):
    print(f"\n{title}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--B", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--theta", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=260817)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = ExperimentConfig(
        n=args.n,
        reps=args.reps,
        B=args.B,
        folds=args.folds,
        theta=args.theta,
        seed=args.seed,
    )
    if config.n < 50 or config.reps < 1 or config.B < 2 or config.folds < 2:
        parser.error("require n >= 50, reps >= 1, B >= 2, and folds >= 2")

    tasks = [(truth, rep) for truth in ("logit", "cloglog")
             for rep in range(config.reps)]
    frames = []
    if args.jobs <= 1:
        for completed, (truth, rep) in enumerate(tasks, start=1):
            frames.append(run_replication(config, truth, rep))
            if completed % max(1, min(25, len(tasks))) == 0:
                print(f"Completed {completed}/{len(tasks)}")
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(run_replication, config, truth, rep):
                (truth, rep)
                for truth, rep in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                frames.append(future.result())
                if completed % max(1, min(25, len(tasks))) == 0:
                    print(f"Completed {completed}/{len(tasks)}")
    summary = summarize(pd.concat(frames, ignore_index=True), config.theta)
    bias, coverage = make_tables(summary)
    _print_table("BIAS AGAINST LATENT THETA", bias)
    _print_table("95% COVERAGE OF LATENT THETA", coverage)

    full_run = config.n >= 2000 and config.reps >= 300 and config.B >= 25
    suffix = "" if full_run else "_smoke"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bias.to_csv(args.output_dir / f"binary_link_bias{suffix}.csv", index=False)
    coverage.to_csv(
        args.output_dir / f"binary_link_coverage{suffix}.csv", index=False
    )
    print(f"\nWrote two tables to {args.output_dir}")


if __name__ == "__main__":
    main()
