"""Ordinal SUDO Monte Carlo (stage-4 config).

Run from python/: uv run ../manuscript/simulations/ordinal_mc.py
"""

import argparse
from pathlib import Path

from sklearn.linear_model import LinearRegression

from sudo import SudoDML, generate_ordinal
from sudo.mc import run_mc, summarize_mc

MANUSCRIPT = Path(__file__).parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-obs", type=int, default=3000)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--B", type=int, default=25)
    args = ap.parse_args()

    df = run_mc(
        args.iterations,
        lambda seed: generate_ordinal(args.n_obs, theta=args.theta,
                                      random_seed=seed),
        lambda d: SudoDML(B=args.B, random_state=1,
                          learner_s=LinearRegression(),
                          learner_d=LinearRegression())
        .fit(d[0], d[1], d[2]),
        args.theta, seed=1)
    s = summarize_mc(df)
    print(s.to_string(float_format="%.4f"))
    out = MANUSCRIPT / "results" / "ordinal_mc.csv"
    s.to_frame().T.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
