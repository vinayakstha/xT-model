"""
Train the VAEP concede model on StatsBomb event data.

Mirrors train_xt_model.py exactly except the label is flipped:
  PV (score):    same-team goal in next 10 SECONDS
  PV (concede):  opposing-team goal in next 10 SECONDS

Same 37-feature builder, same XGBoost hyperparameters. Subtracting the
two predictions yields VAEP(action) = PV_score - PV_concede.

Time-based window matches StatsPerform 2019's published PV convention
and is honoured at period boundaries.

Run:
  python3 train_concede_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Reuse the production feature builder verbatim.
from train_xt_model import build_features, FEATURE_NAMES, LOOKAHEAD_SECONDS  # noqa: E402

CSV_PATH = "statsbomb_xt_enhanced.csv.gz"
RANDOM_STATE = 42


def label_leads_to_concede(df, max_seconds=LOOKAHEAD_SECONDS):
    """
    Binary label = 1 if a goal-event by the OPPOSING team occurs within
    `max_seconds` of game-time after this event, in the same period of the
    same match. Mirror of label_leads_to_goal: same time window, flipped
    team-membership condition.
    Requires temporally sorted input within each match.
    """
    is_goal = ((df["event_type"] == "shot") & (df["success"] == True)).astype(int).values
    teams = df["team"].values
    periods = df["period"].values
    abs_sec = (df["minute"].values * 60 + df["second"].values).astype(np.int64)
    labels = np.zeros(len(df), dtype=int)

    for mid, idxs in df.groupby("match_id", sort=False).groups.items():
        idxs = np.asarray(idxs); idxs.sort()
        for i_pos, i in enumerate(idxs):
            for j_pos in range(i_pos + 1, len(idxs)):
                j = idxs[j_pos]
                if periods[j] != periods[i]:
                    break
                if abs_sec[j] - abs_sec[i] > max_seconds:
                    break
                if is_goal[j] and teams[j] != teams[i]:
                    labels[i] = 1
                    break
    return labels


def per_zone_mean_prediction(df, p, nx=12, ny=8, pitch_x=105.0, pitch_y=68.0):
    """Average the predicted probability per start_zone for diagnostics."""
    sx = df["start_x"].astype(float).values
    sy = df["start_y"].astype(float).values
    zx = np.clip((sx / pitch_x * nx).astype(int), 0, nx - 1)
    zy = np.clip((sy / pitch_y * ny).astype(int), 0, ny - 1)
    grid = np.zeros((nx, ny))
    counts = np.zeros((nx, ny))
    for i in range(len(df)):
        grid[zx[i], zy[i]] += p[i]
        counts[zx[i], zy[i]] += 1
    counts[counts == 0] = 1
    return grid / counts


def main():
    print("Step 1: Loading events ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["period"].isin([1, 2])].copy()
    df = df.dropna(subset=["start_x", "start_y", "end_x", "end_y", "match_id", "team", "event_type"]).reset_index(drop=True)
    print(f"  {len(df):,} regular-time events, {df['match_id'].nunique()} matches")

    print("\nStep 2: Sorting temporally within each match ...")
    df = df.sort_values(["match_id", "period", "minute", "second"], kind="stable").reset_index(drop=True)

    print(f"\nStep 3: Building leads_to_concede label (next {LOOKAHEAD_SECONDS} seconds, opposing team) ...")
    y = label_leads_to_concede(df, max_seconds=LOOKAHEAD_SECONDS)
    pos_rate = y.mean()
    print(f"  positive rate: {pos_rate*100:.3f}%  ({y.sum():,} / {len(y):,})")

    print("\nStep 4: Engineering 37 features (same builder as PV) ...")
    X = build_features(df)
    print(f"  X shape: {X.shape}")

    print("\nStep 5: Match-based 80/20 split (same RANDOM_STATE as PV) ...")
    matches = df["match_id"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(matches)
    n_test = max(1, int(len(matches) * 0.20))
    test_set = set(matches[:n_test])
    test_mask = df["match_id"].isin(test_set).values
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    print(f"  train: {len(X_train):,} from {len(matches)-n_test} matches  ({y_train.mean()*100:.3f}% positive)")
    print(f"  test:  {len(X_test):,} from {n_test} matches  ({y_test.mean()*100:.3f}% positive)")

    print("\nStep 6: Standardising features and training XGBoost ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=25,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=100)
    print(f"  best iteration: {model.best_iteration}")

    print("\nStep 7: Evaluating on holdout ...")
    p_test = model.predict_proba(X_test_s)[:, 1]
    auc = roc_auc_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)
    ll = log_loss(y_test, p_test)
    base = np.full_like(p_test, y_train.mean())
    print(f"  ROC-AUC:    {auc:.4f}")
    print(f"  Brier:      {brier:.5f}  (baseline: {brier_score_loss(y_test, base):.5f})")
    print(f"  log-loss:   {ll:.4f}")
    print(f"  test mean prediction: {p_test.mean():.5f}")
    print(f"  test max  prediction: {p_test.max():.4f}")

    importances = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\n  Top 8 features by importance:")
    for name, imp in importances[:8]:
        print(f"    {name:<30s}  {imp:.4f}")

    print("\nStep 8: Per-zone mean concede prediction (full dataset) ...")
    p_all = model.predict_proba(scaler.transform(X))[:, 1]
    print(f"  full-dataset mean: {p_all.mean():.5f}")
    print(f"  full-dataset max:  {p_all.max():.4f}")
    grid = per_zone_mean_prediction(df, p_all)
    own_box  = grid[0, 3:5].mean()       # own six-yard area (zx=0, zy=3-4)
    own_def  = grid[1:4, :].mean()       # defensive third overall
    midfield = grid[4:8, :].mean()
    att_3rd  = grid[8:11, :].mean()
    opp_box  = grid[11, 3:5].mean()      # opposition six-yard area (zx=11, zy=3-4)
    print(f"  own six-yard zone (zx=0, zy=3-4):     {own_box*100:.3f}%")
    print(f"  own defensive third (zx=1-3 mean):    {own_def*100:.3f}%")
    print(f"  midfield (zx=4-7 mean):               {midfield*100:.3f}%")
    print(f"  attacking third (zx=8-10 mean):       {att_3rd*100:.3f}%")
    print(f"  opposition six-yard zone (zx=11, zy=3-4): {opp_box*100:.3f}%")

    print("\nStep 9: Saving artefacts ...")
    with open("xt_concede_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("xt_concede_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("xt_concede_features.pkl", "wb") as f:
        pickle.dump(FEATURE_NAMES, f)
    print("  xt_concede_model.pkl, xt_concede_scaler.pkl, xt_concede_features.pkl")
    print("\n[Done]")


if __name__ == "__main__":
    main()
