"""R/Python parity on the stage-2 config: binary, linear nuisances, proper MI.

Runs the Python estimator on the stage-2 DGP and compares the MC summary to
the committed R results (R/results/stage2_summary.csv, estimator proper_B25).
Run from python/: uv run python parity_check.py
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from sudo import SudoDML, generate_binary
from sudo.mc import run_mc, summarize_mc

THETA = 1.5
N_REPS = 100

r_summ = pd.read_csv(Path(__file__).parent.parent
                     / "R/results/stage2_summary.csv")
r_row = r_summ[r_summ["estimator"] == "proper_B25"].iloc[0]

df = run_mc(
    N_REPS,
    lambda seed: generate_binary(5000, theta=THETA, random_seed=seed),
    lambda d: SudoDML(B=25, random_state=11,
                      learner_s=LinearRegression(),
                      learner_d=LinearRegression()).fit(d[0], d[1], d[2]),
    THETA, seed=123)
py = summarize_mc(df)

print(f"{'':10} {'bias':>9} {'sd':>8} {'mean_se':>8} {'coverage':>9}")
print(f"{'R':10} {r_row['bias']:+9.4f} {r_row['sd']:8.4f} "
      f"{r_row['mean_se']:8.4f} {r_row['coverage']:9.3f}")
print(f"{'Python':10} {py['bias']:+9.4f} {py['sd']:8.4f} "
      f"{py['mean_se']:8.4f} {py['coverage']:9.3f}")

mc_se = (py["mc_se_bias"] ** 2 + float(r_row["mc_se_bias"]) ** 2) ** 0.5
assert abs(py["bias"] - float(r_row["bias"])) < 3 * mc_se, "bias mismatch"
assert abs(py["sd"] - float(r_row["sd"])) < 0.25 * float(r_row["sd"]), \
    "sampling sd mismatch"
assert py["coverage"] >= 0.925, "python coverage below band"
print("PASS: R and Python stage-2 summaries agree within MC error")
