"""
Train state-conditioned PV models for canonical Decroos state-delta VAEP.

This produces TWO XGBoost classifiers that predict, given a state representation
(current action + last 2 actions of context, 63 features), the probability of
a same-team goal (state_score) or opposing-team goal (state_concede) within the
next 10 SECONDS of game time. Both use the same 10-second labelling as the
per-action PV / concede models.

Used by Day 7 match analyzer (compare_three_metrics.py) to compute canonical
state-delta VAEP alongside the per-action variant. The lab does NOT use these
models — the lab is single-action UX and would have to zero-pad lag features
(synthetic), which we deliberately avoid. State-delta is reserved for the
offline pipeline where real action sequences are available.

Run:
  python3 train_state_models.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from train_xt_model import build_features, FEATURE_NAMES, label_leads_to_goal, LOOKAHEAD_SECONDS  # noqa: E402
from train_concede_model import label_leads_to_concede  # noqa: E402

CSV_PATH = "statsbomb_xt_enhanced.csv.gz"
RANDOM_STATE = 42

# 13 features per previous action: position, distance-to-goal, event-type
# one-hots, success, same-team flag.
PREV_FEATURE_NAMES = [
    "prev_start_x", "prev_start_y", "prev_end_x", "prev_end_y",
    "prev_distance_to_goal",
    "prev_is_pass", "prev_is_shot", "prev_is_cross", "prev_is_through_ball",
    "prev_is_dribble", "prev_is_carry",
    "prev_success", "prev_same_team",
]

STATE_FEATURE_NAMES = (
    list(FEATURE_NAMES)
    + [f"{name}_lag1" for name in PREV_FEATURE_NAMES]
    + [f"{name}_lag2" for name in PREV_FEATURE_NAMES]
)


def previous_action_block(df: pd.DataFrame, lag: int) -> np.ndarray:
    """Return a (n, 13) array describing the action `lag` events earlier within
    the same match. Rows where no previous action exists (start of match, or
    fewer than `lag` events back) are zero-padded."""
    n = len(df)
    out = np.zeros((n, len(PREV_FEATURE_NAMES)), dtype=float)
    sx = df["start_x"].astype(float).values
    sy = df["start_y"].astype(float).values
    ex = df["end_x"].astype(float).values
    ey = df["end_y"].astype(float).values
    teams = df["team"].values
    match_ids = df["match_id"].values
    success = df["success"].fillna(True).astype(bool).values
    etypes = df["event_type"].fillna("pass").values

    idx = np.arange(n)
    prev_idx = idx - lag
    valid = prev_idx >= 0
    same_match = np.zeros(n, dtype=bool)
    same_match[valid] = match_ids[idx[valid]] == match_ids[prev_idx[valid]]
    valid &= same_match

    for i in np.where(valid)[0]:
        p = i - lag
        prev_dist_goal = float(np.sqrt((105 - sx[p]) ** 2 + (34 - sy[p]) ** 2))
        out[i, 0] = sx[p]
        out[i, 1] = sy[p]
        out[i, 2] = ex[p]
        out[i, 3] = ey[p]
        out[i, 4] = prev_dist_goal
        out[i, 5] = float(etypes[p] == "pass")
        out[i, 6] = float(etypes[p] == "shot")
        out[i, 7] = float(etypes[p] == "cross")
        out[i, 8] = float(etypes[p] == "through_ball")
        out[i, 9] = float(etypes[p] == "dribble")
        out[i, 10] = float(etypes[p] == "carry")
        out[i, 11] = float(success[p])
        out[i, 12] = float(teams[p] == teams[i])
    return out


def build_state_features(df: pd.DataFrame) -> np.ndarray:
    """Return (n, 63) state-feature matrix per event."""
    return np.column_stack([
        build_features(df),
        previous_action_block(df, lag=1),
        previous_action_block(df, lag=2),
    ])


def train_one(label_name: str, X_train, X_test, y_train, y_test, scaler):
    print(f"\n  Training STATE_{label_name.upper()} ...")
    Xtr_s = scaler.fit_transform(X_train)
    Xte_s = scaler.transform(X_test)
    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, gamma=0.1,
        objective="binary:logistic", eval_metric="logloss",
        early_stopping_rounds=25, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(Xtr_s, y_train, eval_set=[(Xte_s, y_test)], verbose=200)
    p = model.predict_proba(Xte_s)[:, 1]
    base = np.full_like(p, y_train.mean())
    print(f"  ROC-AUC      {roc_auc_score(y_test, p):.4f}")
    print(f"  Brier        {brier_score_loss(y_test, p):.5f}  (baseline {brier_score_loss(y_test, base):.5f})")
    print(f"  log-loss     {log_loss(y_test, p):.4f}")
    print(f"  mean / max   {p.mean():.5f} / {p.max():.4f}")
    importances = sorted(zip(STATE_FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print(f"  Top 8 features by gain:")
    for fname, imp in importances[:8]:
        print(f"    {fname:<30s}  {imp:.4f}")
    return model


def main():
    print("Step 1: Loading events ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["period"].isin([1, 2])].copy()
    df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y", "match_id", "team", "event_type"]).reset_index(drop=True)
    print(f"  {len(df):,} regular-time events, {df['match_id'].nunique()} matches")

    print("\nStep 2: Sorting temporally within each match ...")
    df = df.sort_values(["match_id", "period", "minute", "second"], kind="stable").reset_index(drop=True)

    print(f"\nStep 3: Building 63-feature STATE vectors (current + 2 previous actions) ...")
    X = build_state_features(df)
    print(f"  X shape: {X.shape}")

    print(f"\nStep 4: Building 10-second labels ...")
    y_score = label_leads_to_goal(df, max_seconds=LOOKAHEAD_SECONDS)
    y_concede = label_leads_to_concede(df, max_seconds=LOOKAHEAD_SECONDS)
    print(f"  score   positive rate: {y_score.mean()*100:.3f}%  ({y_score.sum():,})")
    print(f"  concede positive rate: {y_concede.mean()*100:.3f}%  ({y_concede.sum():,})")

    print("\nStep 5: Match-based 80/20 split (same RANDOM_STATE as PV) ...")
    matches = df["match_id"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(matches)
    n_test = max(1, int(len(matches) * 0.20))
    test_set = set(matches[:n_test])
    test_mask = df["match_id"].isin(test_set).values
    X_train, X_test = X[~test_mask], X[test_mask]
    print(f"  train: {len(X_train):,}    test: {len(X_test):,}")

    score_scaler = StandardScaler()
    score_model = train_one("score", X_train, X_test, y_score[~test_mask], y_score[test_mask], score_scaler)
    concede_scaler = StandardScaler()
    concede_model = train_one("concede", X_train, X_test, y_concede[~test_mask], y_concede[test_mask], concede_scaler)

    print("\nStep 6: Saving artefacts ...")
    pickle.dump(score_model,    open("xt_state_score_model.pkl",    "wb"))
    pickle.dump(score_scaler,   open("xt_state_score_scaler.pkl",   "wb"))
    pickle.dump(concede_model,  open("xt_state_concede_model.pkl",  "wb"))
    pickle.dump(concede_scaler, open("xt_state_concede_scaler.pkl", "wb"))
    pickle.dump(STATE_FEATURE_NAMES, open("xt_state_features.pkl",  "wb"))
    print("  xt_state_score_model.pkl, xt_state_score_scaler.pkl")
    print("  xt_state_concede_model.pkl, xt_state_concede_scaler.pkl")
    print("  xt_state_features.pkl")
    print("\n[Done]")


if __name__ == "__main__":
    main()
