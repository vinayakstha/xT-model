"""
Run all three threat metrics on a single match, with TWO VAEP variants:

  1. Canonical xT (Markov)
  2. PV (XGBoost score model, per-action)
  3a. VAEP (per-action):    PV_score - PV_concede.
                             Lab uses this; consistent with the per-action
                             pedagogy. Has a documented goalkeeper / CB bias
                             because end-coordinate concede is structurally
                             elevated for actions in own defensive third.
  3b. VAEP (state-delta):   Canonical Decroos. State-conditioned classifiers
                             (last 2 actions of context) + state-delta with
                             possession-change perspective flips. Resolves
                             the goalkeeper paradox by crediting defensive
                             actions through the perspective flip on
                             possession changes.

The per-action variant is what the lab shows (single-action UX cannot do
state-delta without synthetic lag features). The state-delta variant uses
real action sequences and is canonical Decroos. Day 7's match analyzer
shows BOTH side by side so the contrast is visible.

Run:
  python3 compare_three_metrics.py
  python3 compare_three_metrics.py --match MATCH_3753999_Watford_vs_Arsenal.csv
"""

import argparse
import os
import pickle
import warnings
import numpy as np
import pandas as pd

# Harmless: models were pickled with scikit-learn 1.3.2; silence the
# cross-version scaler warning (StandardScaler transform is unaffected).
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

import xgboost as _xgb
_xgb.set_config(verbosity=0)  # silence the harmless older-version load notice

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train_xt_model import build_features  # noqa: E402
from train_state_models import build_state_features  # noqa: E402

NX, NY = 12, 8
PITCH_X, PITCH_Y = 105.0, 68.0
DEFAULT_MATCH = "MATCH_3753999_Watford_vs_Arsenal.csv"

# Dataset baselines (from train_state_models.py output) used for match-start
# events that have no real previous action.
BASELINE_STATE_SCORE = 0.00546
BASELINE_STATE_CONCEDE = 0.00165


def state_delta_vaep(m: pd.DataFrame, p_score: np.ndarray, p_concede: np.ndarray):
    """
    Canonical Decroos state-delta VAEP.

    For each consecutive event pair (a_{i-1}, a_i) in the match:
      same-team continuation:  delta_score   = p_score(s_i)   - p_score(s_{i-1})
                                delta_concede = p_concede(s_i) - p_concede(s_{i-1})
      possession change:       delta_score   = p_score(s_i)   - p_concede(s_{i-1})  [perspective flip]
                                delta_concede = p_concede(s_i) - p_score(s_{i-1})    [perspective flip]
      match start:             compare to dataset baselines.

    VAEP(a_i) = delta_score - delta_concede.
    """
    teams = m["team"].values
    match_ids = m["match_id"].values if "match_id" in m.columns else np.zeros(len(m), dtype=int)
    n = len(m)
    delta_score = np.zeros(n)
    delta_concede = np.zeros(n)
    for i in range(n):
        if i == 0 or match_ids[i] != match_ids[i - 1]:
            delta_score[i] = p_score[i] - BASELINE_STATE_SCORE
            delta_concede[i] = p_concede[i] - BASELINE_STATE_CONCEDE
        elif teams[i] == teams[i - 1]:
            delta_score[i] = p_score[i] - p_score[i - 1]
            delta_concede[i] = p_concede[i] - p_concede[i - 1]
        else:
            # possession change: flip perspective on a_{i-1}
            delta_score[i] = p_score[i] - p_concede[i - 1]
            delta_concede[i] = p_concede[i] - p_score[i - 1]
    return delta_score - delta_concede, delta_score, delta_concede


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", default=DEFAULT_MATCH)
    args = parser.parse_args()

    print("Loading models ...")
    grid = np.array(pickle.load(open("xt_zone_values.pkl", "rb")))
    pv_model = pickle.load(open("xt_model.pkl", "rb"))
    pv_scaler = pickle.load(open("xt_scaler.pkl", "rb"))
    co_model = pickle.load(open("xt_concede_model.pkl", "rb"))
    co_scaler = pickle.load(open("xt_concede_scaler.pkl", "rb"))
    state_score_model = pickle.load(open("xt_state_score_model.pkl", "rb"))
    state_score_scaler = pickle.load(open("xt_state_score_scaler.pkl", "rb"))
    state_concede_model = pickle.load(open("xt_state_concede_model.pkl", "rb"))
    state_concede_scaler = pickle.load(open("xt_state_concede_scaler.pkl", "rb"))

    print(f"Loading match: {args.match}")
    m = pd.read_csv(f"match_datasets/{args.match}", low_memory=False)
    m = m.dropna(subset=["start_x", "start_y", "end_x", "end_y", "team", "event_type"]).reset_index(drop=True)
    m["success"] = m["success"].fillna(True).astype(bool)
    if "match_id" not in m.columns:
        m["match_id"] = 0
    m = m.sort_values(["match_id", "period", "minute", "second"], kind="stable").reset_index(drop=True)
    print(f"  {len(m):,} events, teams: {sorted(m['team'].unique())}")

    # Pipeline A: canonical xT (Markov grid lookup)
    sx = np.clip((m["start_x"].values / PITCH_X * NX).astype(int), 0, NX - 1)
    sy = np.clip((m["start_y"].values / PITCH_Y * NY).astype(int), 0, NY - 1)
    ex = np.clip((m["end_x"].values / PITCH_X * NX).astype(int), 0, NX - 1)
    ey = np.clip((m["end_y"].values / PITCH_Y * NY).astype(int), 0, NY - 1)
    succ = m["success"].values
    m["xt"] = np.where(succ, grid[ex, ey] - grid[sx, sy], 0.0)

    # Pipeline B: PV (XGBoost, per-action)
    X = build_features(m)
    m["pv"] = pv_model.predict_proba(pv_scaler.transform(X))[:, 1]

    # Pipeline C variant 1: per-action VAEP (lab-consistent)
    m["pv_concede"] = co_model.predict_proba(co_scaler.transform(X))[:, 1]
    m["vaep_per_action"] = m["pv"] - m["pv_concede"]

    # Pipeline C variant 2: canonical state-delta VAEP
    Xs = build_state_features(m)
    state_p_score = state_score_model.predict_proba(state_score_scaler.transform(Xs))[:, 1]
    state_p_concede = state_concede_model.predict_proba(state_concede_scaler.transform(Xs))[:, 1]
    m["state_score"] = state_p_score
    m["state_concede"] = state_p_concede
    vaep_state, dsc, dco = state_delta_vaep(m, state_p_score, state_p_concede)
    m["vaep_state_delta"] = vaep_state

    # Team totals
    print("\n=== Team totals ===")
    cols_team = ["xt", "pv", "vaep_per_action", "vaep_state_delta", "pv_concede"]
    team_totals = m.groupby("team")[cols_team].sum().round(4)
    print(team_totals.to_string())

    teams = sorted(m["team"].unique())
    if len(teams) == 2:
        a, b = teams
        print(f"\nMargin (xT, PV, VAEP per-action, VAEP state-delta):  {a} - {b}")
        for col in ["xt", "pv", "vaep_per_action", "vaep_state_delta"]:
            print(f"  {col:>20s}: {team_totals.at[a, col] - team_totals.at[b, col]:+.4f}")

    # Top players by xT, by per-action VAEP, by state-delta VAEP
    cols_player = ["xt", "pv", "vaep_per_action", "vaep_state_delta", "pv_concede"]
    by_player = m.groupby(["team", "player"])[cols_player].sum().round(4)

    print("\n=== Top 10 players by xT ===")
    print(by_player.sort_values("xt", ascending=False).head(10).to_string())

    print("\n=== Top 10 players by VAEP (per-action, lab-consistent) ===")
    print(by_player.sort_values("vaep_per_action", ascending=False).head(10).to_string())

    print("\n=== Top 10 players by VAEP (state-delta, canonical Decroos) ===")
    print(by_player.sort_values("vaep_state_delta", ascending=False).head(10).to_string())

    print("\n=== Bottom 10 by VAEP per-action (the goalkeeper / CB paradox) ===")
    print(by_player.sort_values("vaep_per_action", ascending=True).head(10).to_string())

    print("\n=== Bottom 10 by VAEP state-delta (real attacking failures) ===")
    print(by_player.sort_values("vaep_state_delta", ascending=True).head(10).to_string())

    print("\n=== Per-event distributions (this match) ===")
    print(f"  xT          mean: {m['xt'].mean():+.5f}    max: {m['xt'].max():+.4f}    min: {m['xt'].min():+.4f}")
    print(f"  PV          mean: {m['pv'].mean():.5f}     max: {m['pv'].max():.4f}    min: {m['pv'].min():.6f}")
    print(f"  VAEP_per    mean: {m['vaep_per_action'].mean():+.5f}    max: {m['vaep_per_action'].max():+.4f}    min: {m['vaep_per_action'].min():+.4f}")
    print(f"  VAEP_state  mean: {m['vaep_state_delta'].mean():+.5f}    max: {m['vaep_state_delta'].max():+.4f}    min: {m['vaep_state_delta'].min():+.4f}")

    print("\n[Done]")


if __name__ == "__main__":
    main()
