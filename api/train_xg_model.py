"""
Train a calibrated xG model from our 5,987 StatsBomb shots.

Used as the shooting-payoff term in the Markov xT recurrence (Day 2):

    xT(z) = P(shoot|z) * xG(z)  +  P(move_success|z) * sum T(z,z') * xT(z')

where xG(z) = mean( shot_xg ) over shots originating in zone z. Replacing
the empirical goal rate (n_goals / n_shots per zone) with this calibrated
xG reduces noise in low-volume zones and exposes shot context (defender
geometry, angle, pressure) that raw conversion rate cannot.

Pipeline:
    1. Load shots from statsbomb_xt_enhanced.csv.gz (period 1 + 2 only)
    2. Build features: distance_to_goal, angle_to_goal, defender counts,
       goalkeeper distance, distance_to_nearest_defender, log_distance,
       sin/cos angle.
    3. Train a calibrated logistic regression (StandardScaler + LR + isotonic
       calibration via cross-validation).
    4. Hold out 20% of shots for evaluation: Brier, log-loss, ROC-AUC.
    5. Save xt_xg_model.pkl, xt_xg_scaler.pkl, xt_xg_features.pkl.

Run:
    python3 train_xg_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = "statsbomb_xt_enhanced.csv.gz"
RANDOM_STATE = 42

FEATURE_NAMES = [
    "distance_to_goal",
    "angle_to_goal",
    "log_distance_to_goal",
    "sin_angle",
    "cos_angle",
    "num_defenders_5m",
    "num_defenders_10m",
    "distance_to_nearest_defender",
    "goalkeeper_distance",
    "defenders_in_cone",
    "defensive_density",
    "under_pressure",
]


def build_xg_features(shots: pd.DataFrame) -> np.ndarray:
    d = shots["distance_to_goal"].astype(float).values
    a = shots["angle_to_goal"].astype(float).values
    rad = np.radians(a)
    return np.column_stack([
        d,
        a,
        np.log1p(d),
        np.sin(rad),
        np.cos(rad),
        shots["num_defenders_5m"].fillna(0).astype(float).values,
        shots["num_defenders_10m"].fillna(0).astype(float).values,
        shots["distance_to_nearest_defender"].fillna(30).astype(float).values,
        shots["goalkeeper_distance"].fillna(30).astype(float).values,
        shots["defenders_in_cone"].fillna(0).astype(float).values,
        shots["defensive_density"].fillna(0).astype(float).values,
        shots["under_pressure"].fillna(0).astype(int).values.astype(float),
    ])


def main():
    print("Step 1: Loading shots ...")
    df = pd.read_csv(CSV_PATH, low_memory=False)
    shots = df[(df["event_type"] == "shot") & (df["period"].isin([1, 2]))].copy()
    shots["success"] = shots["success"].fillna(False).astype(bool)
    print(f"  {len(shots):,} shots   {int(shots.success.sum()):,} goals   "
          f"conv = {shots.success.mean()*100:.2f}%")

    print("\nStep 2: Building features ...")
    X = build_xg_features(shots)
    y = shots["success"].astype(int).values
    print(f"  X shape: {X.shape}    feature count: {len(FEATURE_NAMES)}")

    print("\nStep 3: Train/test split (80/20 stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"  train: {len(X_train):,} (goals {y_train.sum()})    "
          f"test: {len(X_test):,} (goals {y_test.sum()})")

    print("\nStep 4: Standardising and training calibrated logistic regression ...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base = LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    # Isotonic calibration via 5-fold CV on the training set.
    model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=5)
    model.fit(X_train_s, y_train)

    print("\nStep 5: Evaluating on holdout ...")
    p_test = model.predict_proba(X_test_s)[:, 1]
    base_rate = y_train.mean()
    p_base = np.full_like(p_test, base_rate)

    auc = roc_auc_score(y_test, p_test)
    brier = brier_score_loss(y_test, p_test)
    brier_baseline = brier_score_loss(y_test, p_base)
    ll = log_loss(y_test, p_test)
    ll_base = log_loss(y_test, p_base, labels=[0, 1])

    print(f"  ROC-AUC:    {auc:.4f}")
    print(f"  Brier:      {brier:.5f}    (baseline {brier_baseline:.5f})")
    print(f"  log-loss:   {ll:.4f}     (baseline {ll_base:.4f})")
    print(f"  mean predicted xG on test: {p_test.mean()*100:.2f}%   "
          f"actual goal rate: {y_test.mean()*100:.2f}%")

    print("\nStep 6: Per-distance calibration (sanity check) ...")
    test_dist = X_test[:, 0]
    bins = [0, 6, 11, 16, 22, 100]
    labels = ["0-6m", "6-11m", "11-16m", "16-22m", "22m+"]
    print(f"  {'distance':10s}  {'shots':>6s}  {'pred xG':>10s}  {'actual':>10s}")
    for i in range(len(bins) - 1):
        m = (test_dist >= bins[i]) & (test_dist < bins[i + 1])
        if m.sum() == 0:
            continue
        print(f"  {labels[i]:10s}  {int(m.sum()):>6d}  "
              f"{p_test[m].mean()*100:>9.2f}%  "
              f"{y_test[m].mean()*100:>9.2f}%")

    print("\nStep 7: Saving artefacts ...")
    with open("xt_xg_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("xt_xg_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("xt_xg_features.pkl", "wb") as f:
        pickle.dump(FEATURE_NAMES, f)
    print("  xt_xg_model.pkl, xt_xg_scaler.pkl, xt_xg_features.pkl")
    print("\n[Done]")


if __name__ == "__main__":
    main()
