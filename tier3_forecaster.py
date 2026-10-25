# ============================================================
# TIER 3 — proactive decay forecaster
# ------------------------------------------------------------
# A: 2017 reference distributions          (checkpointed)
# B: per-window unsupervised drift vectors (checkpointed, resumable)
# C: forecaster w/ blocked time-series CV  (no leakage)
# D: LEAD-TIME analysis  <-- the paper's headline claim
# ============================================================

# %% ---- cell 0 ---------------------------------------------
# %run tier0_common.py
import glob, os
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.stats import wasserstein_distance
from pyspark.sql import functions as F

N_BINS = 10
EPS    = 1e-6
FORCE_REF = FORCE_DRIFT = False


# %% ---- cell 1: STAGE A — 2017 reference -------------------
# PSI/Wasserstein need a fixed reference: the training distribution the
# deployed model was fit on. Everything is measured against 2017.
if stage(P["ref"], FORCE_REF):
    ref_sdf = (load_year(PARQUET_2017, 2017)
               .select(*FEATURE_COLS)
               .sample(False, 1.0, seed=SEED)
               .limit(REF_SAMPLE_N))
    ref = ref_sdf.toPandas().astype("float64")

    reference = {}
    for c in FEATURE_COLS:
        v = ref[c].values
        edges = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:                        # near-constant feature
            reference[c] = None; continue
        edges[0], edges[-1] = -np.inf, np.inf
        cnt, _ = np.histogram(v, bins=edges)
        reference[c] = {
            "edges": edges,
            "p":     cnt / max(cnt.sum(), 1),
            "mu":    float(v.mean()),
            "sd":    float(v.std() or 1.0),
            "sample": v[np.random.RandomState(SEED).choice(
                          len(v), size=min(20_000, len(v)), replace=False)],
        }
    save_pickle(reference, P["ref"]); mark_done(P["ref"])

reference = load_pickle(P["ref"])
ACTIVE = [c for c in FEATURE_COLS if reference.get(c) is not None]
print(f"{len(ACTIVE)} active features for drift")


# %% ---- cell 2: STAGE B — per-window drift (resumable) -----
# Computed on the SAME capped sample Tier 2 streams, so drift vectors and
# ARF metrics describe identical windows. Sampling noise at PER_CLASS_CAP
# is a stated limitation; PSI/WD are stable at 10k+ per window.
def psi(expected_p, actual_v, edges):
    cnt, _ = np.histogram(actual_v, bins=edges)
    q = cnt / max(cnt.sum(), 1)
    p = np.clip(expected_p, EPS, None); q = np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))

win_dirs = sorted(glob.glob(os.path.join(P["samples"], "win_key=*")))
if FORCE_DRIFT: reset(P["drift"])

done = {}
if os.path.exists(P["drift"]):
    prev = pd.read_csv(P["drift"]); done = set(prev["win_key"]); rows = prev.to_dict("records")
    print(f"resumed drift: {len(done)} windows done")
else:
    rows, done = [], set()

for wd in win_dirs:
    key = wd.split("win_key=")[-1]
    if key in done: continue
    s = pq.read_table(wd).to_pandas()
    rec = {"win_key": key}
    psis, wds = [], []
    for c in ACTIVE:
        r = reference[c]
        v = s[c].values.astype("float64")
        v = v[np.isfinite(v)]
        if len(v) < 50: continue
        p_ = psi(r["p"], v, r["edges"])
        # standardize by 2017 stats so distances are comparable ACROSS features
        w_ = wasserstein_distance((v - r["mu"]) / r["sd"],
                                  (r["sample"] - r["mu"]) / r["sd"])
        rec[f"psi__{c}"] = p_; rec[f"wd__{c}"] = w_
        psis.append(p_); wds.append(w_)
    # aggregates (robust, low-dimensional) alongside the per-feature vector
    rec.update({
        "psi_mean": np.mean(psis), "psi_max": np.max(psis),
        "psi_p90":  np.percentile(psis, 90),
        "psi_n_gt_0p25": int(np.sum(np.array(psis) > 0.25)),  # classic PSI alarm
        "wd_mean":  np.mean(wds),  "wd_max": np.max(wds),
        "wd_p90":   np.percentile(wds, 90),
    })
    rows.append(rec); done.add(key)
    pd.DataFrame(rows).to_csv(P["drift"], index=False)      # save every window
    print(f"[{len(done)}/{len(win_dirs)}] {key} psi_mean={rec['psi_mean']:.3f}")

mark_done(P["drift"])


# %% ---- cell 3: assemble the modelling table ---------------
drift = pd.read_csv(P["drift"])
drift["TimeWindow"] = pd.to_datetime(drift["win_key"], format="%Y%m%d_%H%M")
tgt = pd.read_csv(P["targets"], parse_dates=["TimeWindow"])

data = (drift.merge(tgt[["TimeWindow", "Year", "Delta_F1", "Window_F1", "n_flows"]],
                    on="TimeWindow", how="inner")
             .sort_values("TimeWindow").reset_index(drop=True))
print("modelling table:", data.shape)

AGG = ["psi_mean","psi_max","psi_p90","psi_n_gt_0p25","wd_mean","wd_max","wd_p90"]
PER = [c for c in data.columns if c.startswith(("psi__","wd__"))]

train = data[data["Year"] == 2018].reset_index(drop=True)   # fit here
test  = data[data["Year"] == 2019].reset_index(drop=True)   # severe-drift holdout
print(f"train(2018)={len(train)}  test(2019)={len(test)}")


# %% ---- cell 4: STAGE C — forecaster, leakage-safe ---------
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
import json

# n is small (hundreds) and p is large (2 x n_features), so feature selection
# MUST live inside the CV pipeline or the R2 is leaked and inflated.
def build(k):
    return Pipeline([
        ("sel", SelectKBest(f_regression, k=k)),
        ("gbr", GradientBoostingRegressor(
            n_estimators=300, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=SEED)),
    ])

results = {}
for name, feats in [("aggregate_only", AGG), ("full_vector", AGG + PER)]:
    X, y = train[feats].fillna(0.0).values, train["Delta_F1"].values
    k = min(20, X.shape[1])
    cv = TimeSeriesSplit(n_splits=5)      # BLOCKED: adjacent windows are
                                          # autocorrelated; random KFold leaks.
    oof = cross_val_predict(build(k), X, y, cv=cv)
    m = build(k).fit(X, y)
    Xt, yt = test[feats].fillna(0.0).values, test["Delta_F1"].values
    pt = m.predict(Xt)

    results[name] = {
        "cv_mae_2018": float(mean_absolute_error(y, oof)),
        "cv_r2_2018":  float(r2_score(y, oof)),
        "mae_2019":    float(mean_absolute_error(yt, pt)),
        "r2_2019":     float(r2_score(yt, pt)),
        "baseline_mae_2018_predict_mean": float(
            mean_absolute_error(y, np.full_like(y, y.mean()))),
    }
    if name == "full_vector":
        test = test.assign(pred_Delta_F1=pt)
        train = train.assign(oof_Delta_F1=oof)
    print(name, json.dumps(results[name], indent=2))

# ALWAYS report against predict-the-mean. If the model cannot beat it, the
# honest finding is that the drift vector is not predictive at this resolution.


# %% ---- cell 5: STAGE D — LEAD TIME (the headline) ---------
# Claim: the unsupervised forecaster raises the alarm BEFORE measured decay
# and BEFORE ARF's reactive adaptation fires.
arf = pd.read_csv(P["arf"], parse_dates=["TimeWindow"])
lead = test.merge(arf[["TimeWindow","ARF_F1","is_decay_dip"]], on="TimeWindow", how="left")
lead = lead.sort_values("TimeWindow").reset_index(drop=True)

thr_pred = train["oof_Delta_F1"].quantile(0.90)    # alarm threshold from 2018 ONLY
thr_true = train["Delta_F1"].quantile(0.90)

lead["alarm_pred"]   = lead["pred_Delta_F1"] > thr_pred
lead["degraded_true"] = lead["Delta_F1"]      > thr_true

def first_true(s): 
    idx = np.where(s.values)[0]
    return int(idx[0]) if len(idx) else None

i_pred, i_true = first_true(lead["alarm_pred"]), first_true(lead["degraded_true"])
i_arf = first_true(lead["is_decay_dip"].fillna(False))

summary = {
    "window_hours": WINDOW_HOURS,
    "first_forecast_alarm_idx": i_pred,
    "first_measured_degradation_idx": i_true,
    "first_arf_dip_idx": i_arf,
    "lead_windows_vs_measured": (None if None in (i_pred, i_true) else i_true - i_pred),
    "lead_windows_vs_arf":      (None if None in (i_pred, i_arf)  else i_arf  - i_pred),
    "lead_hours_vs_arf": (None if None in (i_pred, i_arf)
                          else (i_arf - i_pred) * WINDOW_HOURS),
    "metrics": results,
}
with open(P["final"], "w") as f: json.dump(summary, f, indent=2, default=str)
mark_done(P["final"])
print(json.dumps(summary, indent=2, default=str))

lead.to_csv(f"{WORK}/08_leadtime_table.csv", index=False)
# Figure for the paper: plot pred_Delta_F1, Delta_F1 and ARF_F1 against
# TimeWindow with the two thresholds marked. Positive lead = the proactive
# forecaster fires first. That plot is the contribution; MAE/R2 support it.
