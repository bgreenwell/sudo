"""Binary SUDO Monte Carlo (stage-2/3 configs).

Run from python/: uv run ../manuscript/simulations/binary_mc.py
"""

import argparse
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sudo import SudoDML, generate_binary, generate_binary_nonlinear
from sudo.mc import run_mc, summarize_mc

MANUSCRIPT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=5000)
    ap.add_argument("--theta", type=float, default=1.5)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--B", type=int, default=25)
    ap.add_argument("--dgp", choices=["linear", "nonlinear"],
                    default="linear")
    args = ap.parse_args()

    if args.dgp == "linear":
        dgp = lambda seed: generate_binary(args.n_obs, theta=args.theta,
                                           random_seed=seed)
        learners = dict(learner_s=LinearRegression(),
                        learner_d=LinearRegression())
    else:
        dgp = lambda seed: generate_binary_nonlinear(
            args.n_obs, theta=args.theta, random_seed=seed)
        learners = dict(learner_s=RandomForestRegressor(
                            200, min_samples_leaf=5, n_jobs=-1),
                        learner_d=RandomForestRegressor(
                            200, min_samples_leaf=5, n_jobs=-1))

    df = run_mc(args.iterations, dgp,
                lambda d: SudoDML(B=args.B, random_state=1, **learners)
                .fit(d[0], d[1], d[2]),
                args.theta, seed=1)
    s = summarize_mc(df)
    print(s.to_string(float_format="%.4f"))
    out = MANUSCRIPT / "results" / f"binary_mc_{args.dgp}.csv"
    s.to_frame().T.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
