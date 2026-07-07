"""
Train the xT zone-value grid using Karun Singh's Markov-chain formulation
(https://karun.in/blog/expected-threat.html), adapted to our StatsBomb data.

Usage:
    python3 train_xt_grid.py                   # combined (period 1 + 2)
    python3 train_xt_grid.py --period p1       # first half only
    python3 train_xt_grid.py --period p2       # second half only
    python3 train_xt_grid.py --period et       # extra time (3 + 4)
    python3 train_xt_grid.py --period all      # run all four sequentially

Per-zone recurrence (one equation per zone z):

    xT(z) = P(shoot|z)        * xG(z)
          + P(move_success|z) * sum_{z'} T(z, z') * xT(z')

where:
    P(shoot|z)        -- shots / all actions originating in z
    P(move_success|z) -- successful moves / all actions originating in z
    P(turnover|z)     -- 1 - P(shoot|z) - P(move_success|z) (implicit, value 0)
    xG(z)             -- mean of calibrated xG over shots from zone z
                         (using xt_xg_model.pkl - see train_xg_model.py)
    T(z, z')          -- successful-move transition probability z -> z'

Turnovers (failed moves) act as an absorbing state with payoff 0, so they
drop out of the recurrence. This is what keeps defensive zones at low xT and
mirrors Karun's published grid behaviour.

Solved by value iteration (xT_0 = 0, iterate until ||xT_{k+1} - xT_k||_inf < eps).

Outputs:
    xt_zone_values.pkl  -- (12, 8) numpy array, indexed [x_zone, y_zone]

Convention matches predict_server.py / match-analyze route: pitch is 105 x 68,
12 zones along the length (x), 8 across the width (y).

Run:
    python3 train_xt_grid.py
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import the xG feature builder so we can score every shot consistently.
from train_xg_model import build_xg_features as build_xg_features  # noqa: E402

CSV_PATH = "statsbomb_xt_enhanced.csv.gz"
NX, NY = 12, 8
PITCH_X, PITCH_Y = 105.0, 68.0
MOVE_TYPES = {"pass", "carry", "dribble"}
SHOT_TYPES = {"shot"}
MAX_ITERS = 200
TOL = 1e-7

PERIOD_FILTERS = {
    "combined": [1, 2],
    "p1": [1],
    "p2": [2],
    "et": [3, 4],
}
PERIOD_OUTPUTS = {
    "combined": "xt_zone_values.pkl",
    "p1": "xt_zone_values_p1.pkl",
    "p2": "xt_zone_values_p2.pkl",
    "et": "xt_zone_values_et.pkl",
}


def zone_xy(x, y):
    """Map (x, y) in pitch coordinates to (zx, zy) zone indices."""
    zx = np.clip((x / PITCH_X * NX).astype(int), 0, NX - 1)
    zy = np.clip((y / PITCH_Y * NY).astype(int), 0, NY - 1)
    return zx, zy


def train_one(period_key: str, df_all: pd.DataFrame) -> None:
    print("=" * 72)
    print(f"PERIOD: {period_key.upper()}    -> {PERIOD_OUTPUTS[period_key]}")
    print("=" * 72)

    print("Step 1: Filtering events for period ...")
    periods = PERIOD_FILTERS[period_key]
    df = df_all[df_all["period"].isin(periods)].copy()
    df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y", "event_type"]).reset_index(drop=True)
    df["success"] = df["success"].fillna(True).astype(bool)
    print(f"  {len(df):,} events from period {periods}, {df['match_id'].nunique()} matches")

    print("\nStep 2: Binning into 12 x 8 zones ...")
    sx, sy = zone_xy(df["start_x"].values.astype(float), df["start_y"].values.astype(float))
    ex, ey = zone_xy(df["end_x"].values.astype(float),   df["end_y"].values.astype(float))
    df["sx"], df["sy"], df["ex"], df["ey"] = sx, sy, ex, ey

    is_shot = df["event_type"].isin(SHOT_TYPES).values
    is_move = df["event_type"].isin(MOVE_TYPES).values
    succ = df["success"].values

    # ---- per-zone action statistics ------------------------------------
    print("\nStep 3: Computing shoot / successful-move / turnover ratios per zone ...")
    n_shots      = np.zeros((NX, NY))
    n_move_succ  = np.zeros((NX, NY))
    n_move_fail  = np.zeros((NX, NY))
    for i in range(len(df)):
        zx, zy = sx[i], sy[i]
        if is_shot[i]:
            n_shots[zx, zy] += 1
        elif is_move[i]:
            if succ[i]:
                n_move_succ[zx, zy] += 1
            else:
                n_move_fail[zx, zy] += 1

    n_actions = n_shots + n_move_succ + n_move_fail
    safe = np.where(n_actions > 0, n_actions, 1)
    p_shoot     = n_shots     / safe
    p_move_succ = n_move_succ / safe
    p_turnover  = n_move_fail / safe

    # ---- xG per zone via the calibrated xG model -----------------------
    # Score every shot with xt_xg_model.pkl, then average per zone. This
    # replaces the previous empirical n_goals / n_shots estimator, which
    # was noisy in low-volume zones and ignored shot context.
    print("\nStep 3b: Scoring every shot with the calibrated xG model ...")
    xg_model  = pickle.load(open("xt_xg_model.pkl",  "rb"))
    xg_scaler = pickle.load(open("xt_xg_scaler.pkl", "rb"))

    shots_df = df[is_shot].reset_index(drop=True)
    shot_X = build_xg_features(shots_df)
    shot_X_s = xg_scaler.transform(shot_X)
    shot_xg = xg_model.predict_proba(shot_X_s)[:, 1]

    xg_sum = np.zeros((NX, NY))
    shot_zx = sx[is_shot]
    shot_zy = sy[is_shot]
    for i in range(len(shots_df)):
        xg_sum[shot_zx[i], shot_zy[i]] += shot_xg[i]

    # Per-zone xG = mean(shot_xg) for shots originating in that zone.
    xg_zone = np.where(n_shots > 0, xg_sum / np.where(n_shots > 0, n_shots, 1), 0.0)

    # Save the per-zone xG grid for the lab's "Show xG Grid" overlay (combined
    # period only; per-period grids would need their own xG aggregation).
    # NOTE: we deliberately do NOT mirror-symmetrise this grid. The xG model
    # is trained on raw coordinates (distance to goal, angle, defender
    # geometry, pressure) and its outputs reflect what the data actually
    # contains. Imposing left/right symmetry on the displayed grid would
    # hide real model variance. Asymmetries are mostly sample-size noise
    # in our 236-match dataset; the user can see them directly.
    if period_key == "combined":
        with open("xt_xg_zone_means.pkl", "wb") as f:
            pickle.dump(xg_zone, f)
        asym = np.abs(xg_zone - xg_zone[:, ::-1])
        print(f"  wrote xt_xg_zone_means.pkl  shape={xg_zone.shape}  peak={xg_zone.max():.4f}")
        print(f"  max left/right asymmetry: {asym.max()*100:.3f}%   mean asymmetry: {asym.mean()*100:.3f}%")

    print(f"  total shots: {int(n_shots.sum()):,}")
    print(f"  successful moves: {int(n_move_succ.sum()):,}    failed moves: {int(n_move_fail.sum()):,}")
    print(f"  peak P(shoot)={p_shoot.max():.3f}   peak P(move_succ)={p_move_succ.max():.3f}   peak P(turnover)={p_turnover.max():.3f}")
    print(f"  peak xG(z)={xg_zone.max():.3f}    mean shot xG={shot_xg.mean():.3f}")

    # ---- successful-move transition matrix T(z -> z') ------------------
    print("\nStep 4: Building successful-move transition tensor T(z -> z') ...")
    # T has shape (NX, NY, NX, NY): from (zx, zy) to (zx', zy').
    # Use only successful move actions, since failed moves break possession
    # and would not propagate threat under Karun's simplified Markov model.
    T = np.zeros((NX, NY, NX, NY))
    move_succ = is_move & succ
    for i in np.where(move_succ)[0]:
        T[sx[i], sy[i], ex[i], ey[i]] += 1.0

    # Row-normalise: for each starting zone, sum over destination zones = 1
    row_sums = T.reshape(NX * NY, NX * NY).sum(axis=1)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    T = (T.reshape(NX * NY, NX * NY) / row_sums[:, None]).reshape(NX, NY, NX, NY)
    print(f"  successful moves used: {int(move_succ.sum()):,}")

    # ---- value iteration -----------------------------------------------
    print("\nStep 5: Solving recurrence by value iteration ...")
    xT = np.zeros((NX, NY))
    shoot_term = p_shoot * xg_zone   # xT contribution from shooting
    for it in range(1, MAX_ITERS + 1):
        # Expected continuation value: for each starting zone, weighted avg of xT
        # over successful-move destinations.
        # einsum: 'ijkl,kl->ij' = sum_{kl} T[i,j,k,l] * xT[k,l]
        cont = np.einsum("ijkl,kl->ij", T, xT)
        new_xT = shoot_term + p_move_succ * cont
        delta = np.abs(new_xT - xT).max()
        xT = new_xT
        if it <= 5 or it % 10 == 0:
            print(f"  iter {it:3d}   max|delta| = {delta:.2e}   xT.max = {xT.max():.4f}")
        if delta < TOL:
            print(f"  converged at iter {it} (tolerance {TOL})")
            break
    else:
        print(f"  WARNING: hit MAX_ITERS={MAX_ITERS} without convergence")

    # ---- mirror-symmetrise across the y axis (Karun's convention) ------
    print("\nStep 6: Mirror-symmetrising about the centre line ...")
    # y is across the width; reflect zy <-> NY-1-zy.
    xT = (xT + xT[:, ::-1]) / 2.0

    # ---- save -----------------------------------------------------------
    out_path = PERIOD_OUTPUTS[period_key]
    print(f"\nStep 7: Saving {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump(xT, f)
    print(f"  wrote {out_path}    shape={xT.shape}    min={xT.min():.5f}    max={xT.max():.5f}")

    # ---- print grid in (y, x) orientation for human inspection ---------
    print("\nFinal grid (rows=y, cols=x; x=11 = attacking goal):")
    for y in range(NY):
        row = "  " + "  ".join(f"{xT[x, y]:.4f}" for x in range(NX))
        print(row)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--period",
        choices=["combined", "p1", "p2", "et", "all"],
        default="combined",
        help="Which period to train (default: combined). Use 'all' to train all four sequentially.",
    )
    args = parser.parse_args()

    print("Loading event CSV once ...")
    df_all = pd.read_csv(CSV_PATH, low_memory=False)
    print(f"  {len(df_all):,} events total")

    if args.period == "all":
        for k in ["combined", "p1", "p2", "et"]:
            train_one(k, df_all)
    else:
        train_one(args.period, df_all)
    print("[Done]")


if __name__ == "__main__":
    main()
