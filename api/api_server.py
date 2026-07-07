"""
Flow of Threat - standalone xT / PV / VAEP match analyzer + threat-grid explorer.

A FastAPI server + browser UI that runs all three threat metrics on a single
match, exactly as the production engine does:

  Pipeline A - canonical xT (Markov grid lookup, mirror-symmetrised)
  Pipeline B - PV (per-action XGBoost, P(same-team goal in next 10s))
  Pipeline C1 - VAEP per-action  = PV_score - PV_concede
  Pipeline C2 - VAEP state-delta = canonical Decroos (state-conditioned
                classifiers with possession-change perspective flips)

The feature builders and the state-delta routine are imported directly from the
same scripts the models were trained with, so the numbers match the course's
Day 6 output bit-for-bit. The interactive threat-grid explorer reads the exact
same per-zone pickles the production lab uses.

Run:
    pip install -r requirements.txt
    python api_server.py
    # open http://localhost:8000
"""

import io
import os
import pickle

# Serve model pickles and match data by relative path regardless of where the
# server is launched from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import warnings

# The model pickles were saved with scikit-learn 1.3.2. Newer scikit-learn
# raises a precautionary InconsistentVersionWarning when unpickling a scaler
# from an older version. It is harmless here: StandardScaler only stores the
# mean_/scale_ arrays and transform() is plain arithmetic on them, so
# predictions are identical. Silence the noise so the console stays clean.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import xgboost as _xgb

# Models were trained with an older XGBoost; loading them prints a harmless
# "export with Booster.save_model" notice. The booster still loads and predicts
# identically, so silence XGBoost's warning stream to keep the console clean.
_xgb.set_config(verbosity=0)

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Feature builders + state-delta routine, straight from the training scripts.
from train_xt_model import build_features            # 37-feature per-action vector
from train_state_models import build_state_features  # 63-feature state vector
from train_xg_model import build_xg_features          # 12-feature xG shot vector
from compare_three_metrics import (
    state_delta_vaep,
    BASELINE_STATE_SCORE,
    BASELINE_STATE_CONCEDE,
)

NX, NY = 12, 8
PITCH_X, PITCH_Y = 105.0, 68.0
MATCH_DIR = "match_datasets"
REQUIRED = ["start_x", "start_y", "end_x", "end_y", "event_type", "team",
            "period", "minute", "second"]

# ----------------------------------------------------------------------------
# Load every model + grid once at startup.
# ----------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PKL_DIR = "pkl"

def _load(name):
    return pickle.load(open(os.path.join(PKL_DIR, name), "rb"))

print("Loading models ...", flush=True)
GRID = np.array(_load("xt_zone_values.pkl"))            # combined xT grid


def _grid_or_combined(name):
    try:
        return np.array(_load(name))
    except FileNotFoundError:
        return GRID

# Period-specific xT grids for the explorer's Period control.
GRIDS = {
    "combined": GRID,
    "p1": _grid_or_combined("xt_zone_values_p1.pkl"),
    "p2": _grid_or_combined("xt_zone_values_p2.pkl"),
    "et": _grid_or_combined("xt_zone_values_et.pkl"),
}
XG_ZONES = np.array(_load("xt_xg_zone_means.pkl"))      # per-zone calibrated xG
PV_ZONES = np.array(_load("xt_pv_zone_means.pkl"))      # per-zone mean PV

PV_MODEL = _load("xt_model.pkl");         PV_SCALER = _load("xt_scaler.pkl")
CO_MODEL = _load("xt_concede_model.pkl"); CO_SCALER = _load("xt_concede_scaler.pkl")
SS_MODEL = _load("xt_state_score_model.pkl");   SS_SCALER = _load("xt_state_score_scaler.pkl")
SC_MODEL = _load("xt_state_concede_model.pkl"); SC_SCALER = _load("xt_state_concede_scaler.pkl")
XG_MODEL = _load("xt_xg_model.pkl");      XG_SCALER = _load("xt_xg_scaler.pkl")
print("Models loaded. Peak xT %.4f | peak xG-zone %.4f | peak PV-zone %.4f"
      % (GRID.max(), XG_ZONES.max(), PV_ZONES.max()), flush=True)

# xG-feature source columns and their fill defaults (mirrors build_xg_features).
XG_FILL = {
    "num_defenders_5m": 0.0, "num_defenders_10m": 0.0,
    "distance_to_nearest_defender": 30.0, "goalkeeper_distance": 30.0,
    "defenders_in_cone": 0.0, "defensive_density": 0.0, "under_pressure": 0.0,
}


def analyze(m: pd.DataFrame) -> dict:
    """Score one match dataframe with all three metrics; return aggregates + shots."""
    missing = [c for c in REQUIRED if c not in m.columns]
    if missing:
        raise HTTPException(400, f"Missing required columns: {', '.join(missing)}")

    m = m.dropna(subset=["start_x", "start_y", "end_x", "end_y", "team", "event_type"]).reset_index(drop=True)
    if len(m) == 0:
        raise HTTPException(400, "No usable rows after dropping missing coordinates.")
    m["success"] = m["success"].fillna(True).astype(bool) if "success" in m.columns else True
    if "under_pressure" not in m.columns:
        m["under_pressure"] = 0
    if "counter_attack" not in m.columns:
        m["counter_attack"] = 0
    if "player" not in m.columns:
        m["player"] = "(unknown)"
    if "match_id" not in m.columns:
        m["match_id"] = 0
    m = m.sort_values(["match_id", "period", "minute", "second"], kind="stable").reset_index(drop=True)

    # Pipeline A: canonical xT (Markov grid lookup)
    sx = np.clip((m["start_x"].values / PITCH_X * NX).astype(int), 0, NX - 1)
    sy = np.clip((m["start_y"].values / PITCH_Y * NY).astype(int), 0, NY - 1)
    ex = np.clip((m["end_x"].values / PITCH_X * NX).astype(int), 0, NX - 1)
    ey = np.clip((m["end_y"].values / PITCH_Y * NY).astype(int), 0, NY - 1)
    m["xt"] = np.where(m["success"].values, GRID[ex, ey] - GRID[sx, sy], 0.0)

    # Pipeline B + C1: PV, PV_concede, per-action VAEP
    X = build_features(m)
    m["pv"] = PV_MODEL.predict_proba(PV_SCALER.transform(X))[:, 1]
    m["pv_concede"] = CO_MODEL.predict_proba(CO_SCALER.transform(X))[:, 1]
    m["vaep_per_action"] = m["pv"] - m["pv_concede"]

    # Pipeline C2: canonical state-delta VAEP
    Xs = build_state_features(m)
    p_score = SS_MODEL.predict_proba(SS_SCALER.transform(Xs))[:, 1]
    p_concede = SC_MODEL.predict_proba(SC_SCALER.transform(Xs))[:, 1]
    m["vaep_state_delta"], _, _ = state_delta_vaep(m, p_score, p_concede)

    cols = ["xt", "pv", "vaep_per_action", "vaep_state_delta", "pv_concede"]

    team = m.groupby("team")[cols].sum().round(4)
    teams = list(team.index)
    team_rows = [dict(team=t, **{c: float(team.at[t, c]) for c in cols}) for t in teams]
    margin = None
    if len(teams) == 2:
        a, b = teams
        margin = {c: round(float(team.at[a, c] - team.at[b, c]), 4) for c in cols}
        margin["a"], margin["b"] = a, b

    by = m.groupby(["team", "player"])[cols].sum().round(4).reset_index()

    def top(sort_col, n=10, asc=False):
        d = by.sort_values(sort_col, ascending=asc).head(n)
        return [dict(team=r["team"], player=r["player"],
                     **{c: float(r[c]) for c in cols}) for _, r in d.iterrows()]

    # Per-shot calibrated xG (same model + features as the xG grid).
    m["xg"] = np.nan
    shot_mask = m["event_type"].values == "shot"
    if shot_mask.any():
        s = m[shot_mask].copy()
        if "distance_to_goal" not in s.columns:
            s["distance_to_goal"] = np.sqrt((105 - s["start_x"]) ** 2 + (34 - s["start_y"]) ** 2)
        if "angle_to_goal" not in s.columns:
            s["angle_to_goal"] = np.degrees(np.arctan2(7.32, 105 - s["start_x"] + 0.1))
        for c, d in XG_FILL.items():
            if c not in s.columns:
                s[c] = d
        try:
            m.loc[shot_mask, "xg"] = XG_MODEL.predict_proba(XG_SCALER.transform(build_xg_features(s)))[:, 1]
        except Exception:
            pass

    # Every pass / carry / dribble / shot as a plottable event (arrows + shot dots).
    ev_types = ("pass", "carry", "dribble", "shot")
    em = m[m["event_type"].isin(ev_types)].copy()
    for c, dp in [("start_x", 1), ("start_y", 1), ("end_x", 1), ("end_y", 1),
                  ("xt", 4), ("pv", 4), ("vaep_per_action", 4), ("xg", 4)]:
        em[c] = em[c].round(dp)
    events = []
    for r in em.to_dict("records"):
        events.append({
            "type": r["event_type"], "team": r["team"], "player": r["player"],
            "sx": r["start_x"], "sy": r["start_y"], "ex": r["end_x"], "ey": r["end_y"],
            "xt": r["xt"], "pv": r["pv"], "vaep": r["vaep_per_action"],
            "goal": bool(r["success"]) if r["event_type"] == "shot" else None,
            "xg": None if pd.isna(r["xg"]) else r["xg"],
        })

    # Top 20 events by |xT|.
    order = m["xt"].abs().sort_values(ascending=False).index[:20]
    top_events = [{
        "team": m.at[i, "team"], "player": m.at[i, "player"], "type": m.at[i, "event_type"],
        "minute": int(m.at[i, "minute"]) if pd.notna(m.at[i, "minute"]) else None,
        "sx": round(float(m.at[i, "start_x"]), 1), "sy": round(float(m.at[i, "start_y"]), 1),
        "ex": round(float(m.at[i, "end_x"]), 1), "ey": round(float(m.at[i, "end_y"]), 1),
        "xt": round(float(m.at[i, "xt"]), 4), "pv": round(float(m.at[i, "pv"]), 4),
        "vaep": round(float(m.at[i, "vaep_per_action"]), 4),
    } for i in order]

    # Per-team event-type counts + match totals.
    for tr in team_rows:
        tm = m[m["team"] == tr["team"]]
        tr["events"] = int(len(tm))
        tr["passes"] = int((tm["event_type"] == "pass").sum())
        tr["carries"] = int((tm["event_type"] == "carry").sum())
        tr["dribbles"] = int((tm["event_type"] == "dribble").sum())
    totals = {
        "events": int(len(m)),
        "passes": int((m["event_type"] == "pass").sum()),
        "carries": int((m["event_type"] == "carry").sum()),
        "dribbles": int((m["event_type"] == "dribble").sum()),
    }

    dist = {
        "xt": [round(float(m["xt"].mean()), 5), round(float(m["xt"].max()), 4), round(float(m["xt"].min()), 4)],
        "pv": [round(float(m["pv"].mean()), 5), round(float(m["pv"].max()), 4), round(float(m["pv"].min()), 6)],
        "vaep_per_action": [round(float(m["vaep_per_action"].mean()), 5), round(float(m["vaep_per_action"].max()), 4), round(float(m["vaep_per_action"].min()), 4)],
        "vaep_state_delta": [round(float(m["vaep_state_delta"].mean()), 5), round(float(m["vaep_state_delta"].max()), 4), round(float(m["vaep_state_delta"].min()), 4)],
    }

    return {
        "n_events": int(len(m)),
        "teams": team_rows,
        "team_names": teams,
        "totals": totals,
        "margin": margin,
        "by_xt": top("xt", n=15),
        "top_per_action": top("vaep_per_action"),
        "top_state": top("vaep_state_delta"),
        "bottom_per_action": top("vaep_per_action", asc=True),
        "bottom_state": top("vaep_state_delta", asc=True),
        "top_events": top_events,
        "distributions": dist,
        "events": events,
    }


URL = "http://localhost:8000"


@asynccontextmanager
async def lifespan(app):
    bar = "=" * 56
    print(f"\n  {bar}\n   Flow of Threat is ready.\n"
          f"   Open  {URL}  in your browser.\n  {bar}\n", flush=True)
    yield


app = FastAPI(title="Flow of Threat", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse("main.html")


@app.get("/style.css")
def css():
    return FileResponse("style.css", media_type="text/css")


@app.get("/script.js")
def js():
    return FileResponse("script.js", media_type="application/javascript")


@app.get("/api/matches")
def list_matches():
    out = []
    for name in sorted(os.listdir(MATCH_DIR)):
        if name.endswith(".csv"):
            label = name.replace("MATCH_", "").replace(".csv", "")
            parts = label.split("_", 1)
            label = parts[1].replace("_", " ") if len(parts) == 2 else label
            out.append({"file": name, "label": label})
    return out


@app.get("/api/analyze")
def analyze_bundled(match: str):
    safe = os.path.basename(match)
    path = os.path.join(MATCH_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Match not found: {safe}")
    return JSONResponse(analyze(pd.read_csv(path, low_memory=False)))


@app.post("/api/analyze")
async def analyze_upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        m = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    return JSONResponse(analyze(m))


@app.get("/api/grid")
def grid(mode: str = "xt", period: str = "combined"):
    """Return a 12x8 per-zone grid: mode = xt | xg | pv (xt honours period)."""
    if mode == "xg":
        g, label = XG_ZONES, "Per-zone calibrated xG"
    elif mode == "pv":
        g, label = PV_ZONES, "Per-zone mean PV"
    else:
        g = GRIDS.get(period, GRID)
        label = "Markov xT (%s)" % period
    return {"zones": g.tolist(), "max_value": float(g.max()),
            "mode": mode, "period": period, "label": label}


@app.post("/api/move")
def move(payload: dict = Body(...)):
    """xT of a move (Karun: end - start on success, 0 on fail) + the model PV
    of that exact action. Matches the production predict_server / xT lab."""
    try:
        sx = float(payload["start_x"]); sy = float(payload["start_y"])
        ex = float(payload["end_x"]); ey = float(payload["end_y"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "start_x, start_y, end_x, end_y are required numbers.")
    et = payload.get("event_type", "pass")
    success = bool(payload.get("success", True))
    pressure = 1 if payload.get("under_pressure") else 0
    counter = 1 if payload.get("counter_attack") else 0
    period = payload.get("period", "combined")
    g = GRIDS.get(period, GRID)

    szx = int(np.clip(sx / PITCH_X * NX, 0, NX - 1)); szy = int(np.clip(sy / PITCH_Y * NY, 0, NY - 1))
    ezx = int(np.clip(ex / PITCH_X * NX, 0, NX - 1)); ezy = int(np.clip(ey / PITCH_Y * NY, 0, NY - 1))
    start_v = float(g[szx, szy]); end_v = float(g[ezx, ezy])
    zone_xt = (end_v - start_v) if success else 0.0

    row = pd.DataFrame([{
        "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
        "event_type": et, "success": success,
        "under_pressure": pressure, "counter_attack": counter,
    }])
    Xa = build_features(row)
    model_pv = float(PV_MODEL.predict_proba(PV_SCALER.transform(Xa))[0, 1])
    pv_concede = float(CO_MODEL.predict_proba(CO_SCALER.transform(Xa))[0, 1])

    return {
        "zone_based_xt": zone_xt,
        "start_zone_xt": start_v, "end_zone_xt": end_v,
        "start_zone": [szx, szy], "end_zone": [ezx, ezy],
        "model_pv": model_pv, "pv_concede": pv_concede,
        "vaep_per_action": model_pv - pv_concede,
        "success": success,
    }


if __name__ == "__main__":
    # log_level="warning" hides uvicorn's INFO lines (incl. the confusing
    # "running on http://0.0.0.0:8000") so the ready banner above is what the
    # user sees. The server still binds 0.0.0.0 (reachable on your network);
    # you open it at http://localhost:8000.
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
