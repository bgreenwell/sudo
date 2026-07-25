"""Plan B replication -- Liu, Zhang & Zhou (2021) Section 6, on the rebuilt
SUDO package.

Effect of emergency-contraceptive (EC) pill availability on two outcomes,
using municipality-level Chilean data:
  Y^(1): early-gestation fetal death (abortion proxy), pregnant women 15-24
  Y^(2): new birth, all women

Treatment A: EC pill accessible in municipality (binary).
Covariates Z: 16 municipality-level features.

Both SUDO and the logistic-PLR benchmark (Liu et al.'s own estimator, via
DoubleMLLPLR) target the same latent log-odds effect and share a
logistic-linear index, so this is an apples-to-apples comparison; SUDO adds
the surrogate + proper multiple-imputation inference. Published targets
(Liu et al. Tables 3-4, RF column):
  beta^(1) = -0.215  [-0.378, -0.052]  p = 0.007
  beta^(2) = -0.112  [-0.224,  0.000]  p = 0.033

Run from python/:
    uv run ../manuscript/simulations/planb_replication.py
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

from sudo import SudoDML

warnings.filterwarnings("ignore")

MANUSCRIPT = Path(__file__).parent.parent
DATA_DIR = MANUSCRIPT / "replication_AssessingPlanB" / "data"
RESULTS = MANUSCRIPT / "results"

# 16 municipality-level covariates used by Liu et al. (Section 6)
Z_COLS = [
    "outofschool", "healthspend", "healthstaff", "healthtraining",
    "educationspend", "educationmunicip", "femalepoverty", "urban",
    "condom", "usingcont", "femaleworkers", "poverty",
    "mujer", "votop", "pilldistance", "year",
]

TARGETS = {
    "Y^(1)": "beta=-0.215  CI=[-0.378,-0.052]  p=0.007",
    "Y^(2)": "beta=-0.112  CI=[-0.224, 0.000]  p=0.033",
}


# -- data construction (unchanged from the v1 replication) -----------------

def _preprocess(df):
    df = df.copy()
    df.loc[df["pill"] == 1, "pilldistance"] = 0.0
    return df


def build_abortion_dataset(seed=42):
    """Y^(1): early-gestation fetal death, pregnant women 15-24, Y=0
    down-sampled to 25% prevalence."""
    df = pd.read_csv(DATA_DIR / "S1Data_deaths_covars.csv", low_memory=False)
    df = _preprocess(df)
    df = df[(df["pregnant"] == 1) & df["age"].between(15, 24)].copy()
    df["earlyP"] = df["earlyP"].clip(lower=0).astype(int)
    df["n"] = df["n"].astype(int)
    df["n_safe"] = (df["n"] - df["earlyP"]).clip(lower=0)
    df = df.dropna(subset=Z_COLS + ["pill"])

    y1_rows = df[df["earlyP"] > 0]
    y1 = y1_rows.loc[y1_rows.index.repeat(y1_rows["earlyP"])][
        Z_COLS + ["pill"]].copy()
    y1["Y"] = 1

    y0_pool = df[df["n_safe"] > 0]
    n_neg = 3 * len(y1)  # prevalence 0.25
    w = y0_pool["n_safe"] / y0_pool["n_safe"].sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y0_pool), size=n_neg, p=w.values, replace=False)
    y0 = y0_pool.iloc[idx][Z_COLS + ["pill"]].copy()
    y0["Y"] = 0

    data = pd.concat([y1, y0], ignore_index=True)
    print(f"Y^(1) [abortion]: n = {len(data):,}  "
          f"prevalence = {data['Y'].mean():.3f}")
    return data


def build_birth_dataset(seed=42, n_target=10_000):
    """Y^(2): new birth (pregnant vs not), down-sampled to n=10,000 at 40%
    prevalence."""
    df = pd.read_csv(DATA_DIR / "S1Data_granular_covars.csv", low_memory=False)
    df = _preprocess(df)
    df = df.dropna(subset=Z_COLS + ["pill", "n", "pregnant"])
    df["n"] = df["n"].astype(int)

    preg = df[df["pregnant"] == 1]
    nonpreg = df[df["pregnant"] == 0]
    n_pos = int(n_target * 0.4)
    n_neg = n_target - n_pos
    rng = np.random.default_rng(seed)

    idx_pos = rng.choice(len(preg), size=n_pos,
                         p=(preg["n"] / preg["n"].sum()).values, replace=False)
    idx_neg = rng.choice(len(nonpreg), size=n_neg,
                         p=(nonpreg["n"] / nonpreg["n"].sum()).values,
                         replace=False)

    y1 = preg.iloc[idx_pos][Z_COLS + ["pill"]].copy()
    y1["Y"] = 1
    y0 = nonpreg.iloc[idx_neg][Z_COLS + ["pill"]].copy()
    y0["Y"] = 0

    data = pd.concat([y1, y0], ignore_index=True)
    print(f"Y^(2) [birth]:    n = {len(data):,}  "
          f"prevalence = {data['Y'].mean():.3f}")
    return data


# -- estimation ------------------------------------------------------------

def _doubleml_lplr(X, A, Y, n_splits, seed):
    from doubleml import DoubleMLData, DoubleMLLPLR
    from sklearn.ensemble import (RandomForestClassifier,
                                  RandomForestRegressor)

    dml = DoubleMLLPLR(
        DoubleMLData.from_arrays(X, Y, A),
        ml_M=RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    n_jobs=-1, random_state=seed),
        ml_t=RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed),
        ml_m=RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                    n_jobs=-1, random_state=seed),
        n_folds=n_splits)
    dml.fit()
    theta = float(dml.coef.squeeze())
    se = float(dml.se.squeeze())
    ci = dml.confint().iloc[0]
    return theta, se, (float(ci.iloc[0]), float(ci.iloc[1])), \
        float(dml.pval.squeeze())


def _overlap_auc(X, A):
    """How predictable is treatment from covariates? AUC near 1 signals a
    positivity/overlap violation, under which any partialling-out estimator
    loses identification (the treatment residual is annihilated)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    p = LogisticRegression(max_iter=1000).fit(X, A.astype(int)) \
        .predict_proba(X)[:, 1]
    return roc_auc_score(A, p)


def run_analysis(data, n_splits=5, B=30, seed=42):
    Y = data["Y"].to_numpy(dtype=int)
    A = data["pill"].to_numpy(dtype=float)
    X = data[Z_COLS].fillna(data[Z_COLS].median()).to_numpy(dtype=float)
    X = (X - X.mean(0)) / X.std(0)  # standardize covariates

    auc = _overlap_auc(X, A)
    print(f"  overlap check: AUC(treatment ~ covariates) = {auc:.3f}"
          + ("   [severe overlap violation]" if auc > 0.9 else ""))

    print("  DoubleMLLPLR (Liu et al.) ...", end=" ", flush=True)
    tl, sl, cil, pl = _doubleml_lplr(X, A, Y, n_splits, seed)
    print(f"beta = {tl:+.3f}  SE = {sl:.3f}  "
          f"CI = [{cil[0]:.3f}, {cil[1]:.3f}]  p = {pl:.3f}")

    print("  SUDO (proper MI)          ...", end=" ", flush=True)
    m = SudoDML(link="logit", n_folds=n_splits, B=B,
                random_state=seed).fit(X, A, Y)
    ts, ss, cis = m.theta_, m.se_, m.ci_
    ps = 2 * (1 - scipy_norm.cdf(abs(ts / ss)))
    print(f"beta = {ts:+.3f}  SE = {ss:.3f}  "
          f"CI = [{cis[0]:.3f}, {cis[1]:.3f}]  p = {ps:.3f}")

    return {"logistic": (tl, sl, cil, pl), "sudo": (ts, ss, cis, ps)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--B", type=int, default=30,
                    help="SUDO surrogate draws")
    args = ap.parse_args()

    print("Plan B replication -- Liu, Zhang & Zhou (2021) Section 6")
    print("=" * 62)

    print("\nBuilding Y^(1) [early fetal death] ...")
    res1 = run_analysis(build_abortion_dataset(args.seed),
                        args.n_splits, args.B, args.seed)
    print("\nBuilding Y^(2) [new birth] ...")
    res2 = run_analysis(build_birth_dataset(args.seed),
                        args.n_splits, args.B, args.seed)

    print("\n-- comparison with Liu et al. (RF column) " + "-" * 20)
    hdr = f"{'Outcome':<8} {'Method':<16} {'beta':>7} {'CI LB':>8} " \
          f"{'CI UB':>8} {'p':>7}  Liu et al. target"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for label, res in [("Y^(1)", res1), ("Y^(2)", res2)]:
        for method, key in [("DoubleMLLPLR", "logistic"), ("SUDO", "sudo")]:
            b, s, ci, p = res[key]
            tgt = TARGETS[label] if key == "logistic" else ""
            print(f"{label:<8} {method:<16} {b:+7.3f} {ci[0]:8.3f} "
                  f"{ci[1]:8.3f} {p:7.3f}  {tgt}")
            rows.append(dict(outcome=label, method=method, beta=b, se=s,
                             ci_lo=ci[0], ci_hi=ci[1], pval=p))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "planb_replication.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
