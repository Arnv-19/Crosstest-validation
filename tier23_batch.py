# ============================================================
# TIER 2 + TIER 3a/3b — single-cell safe, headless Kaggle run
# ------------------------------------------------------------
# PASTE THIS WHOLE FILE AS ONE CELL. Every stage is wrapped so a
# failure in one does not take down the others.
#
# v3 changes:
#   * `pip install river --no-deps`. A plain install pulls
#     numpy 2.5.3 over the build Kaggle's compiled packages
#     expect, which breaks numpy AND scipy. --no-deps installs
#     river's own code only; its deps (numpy/pandas/scipy) are
#     already present.
#   * scipy imported up front, so a broken environment fails in
#     the first seconds instead of after the ARF run.
#   * Tier 2 fully isolated in try/except — the drift vectors
#     are what this run most needs and do not require river.
#
# BEFORE RUNNING: Factory reset (pip damage survives a kernel
# restart), Internet ON.
# ============================================================

import os, glob, json, time, pickle, warnings
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from scipy.stats import wasserstein_distance      # fail fast if scipy is broken
warnings.filterwarnings("ignore")

T0 = time.time()
BUDGET_SECONDS = 8.5 * 3600
def budget_left(): return BUDGET_SECONDS - (time.time() - T0)
def elapsed():     return f"{(time.time()-T0)/60:.1f}min"

# ---------------- paths -------------------------------------
OUT = "/kaggle/working/decay"
os.makedirs(OUT, exist_ok=True)

def _find(rel):
    """Writable copy first, then any attached input at ANY nesting depth."""
    local = f"/kaggle/working/decay/{rel}"
    if os.path.exists(local):
        return local
    hits = sorted(glob.glob(f"/kaggle/input/**/decay/{rel}", recursive=True))
    if not hits:
        hits = sorted(glob.glob(f"/kaggle/input/**/{rel}", recursive=True))
    return hits[0] if hits else None

SAMPLES = _find("03_window_samples")
TARGETS = _find("02_decay_targets.csv")
print("samples:", SAMPLES)
print("targets:", TARGETS)
assert SAMPLES and TARGETS, "Tier-1 output not found — attach the committed notebook"

_p17 = (glob.glob("/kaggle/input/**/cicids2017_master_chronological.parquet",
                  recursive=True) or [None])[0]
print("2017 reference:", _p17)

P = {
    "arf":      f"{OUT}/05_arf_metrics.csv",
    "arf_ckpt": f"{OUT}/05_arf_state.pkl",
    "ref":      f"{OUT}/04_drift_reference.pkl",
    "drift":    f"{OUT}/06_drift_vectors.csv",
}

F1_BASE     = 0.9965683219319521
DEGRADED_F1 = 0.50
SEED        = 42
ARF_CAP     = 3_000      # river's ARF is pure-Python/single-threaded; the
                         # Stage-C 20k cap would not finish. Tier-1 metrics
                         # come from the FULL population, so this affects only
                         # the online baseline's throughput.
N_TREES     = 10
CKPT_EVERY  = 5
N_BINS, EPS = 10, 1e-6

WIN_DIRS = sorted(glob.glob(os.path.join(SAMPLES, "win_key=*")))
WIN_KEYS = [d.split("win_key=")[-1] for d in WIN_DIRS]
assert WIN_DIRS, f"no win_key=* partitions under {SAMPLES}"
print(f"{len(WIN_DIRS)} window partitions: {WIN_KEYS[0]} -> {WIN_KEYS[-1]}")

_schema = pq.read_schema(glob.glob(os.path.join(WIN_DIRS[0], "*.parquet"))[0])
FEATURE_COLS = [c for c in _schema.names if c not in ("win","Year","y","win_key")]
print(len(FEATURE_COLS), "features (from the sample schema)")

def read_window(d, cap=None, seed=SEED):
    df = pq.read_table(d).to_pandas()
    if cap:
        df = (df.groupby("y", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), cap), random_state=seed)))
    return df.reset_index(drop=True)


# ============================================================
# TARGETS
# ============================================================
tgt = pd.read_csv(TARGETS, parse_dates=["TimeWindow"])
tgt["win_key"]  = tgt["TimeWindow"].dt.strftime("%Y%m%d_%H%M")
tgt["degraded"] = (tgt["Window_F1"] < DEGRADED_F1).astype(int)

print("\n", tgt.groupby("Year")["degraded"].agg(["count","sum","mean"]), sep="")
print("\n2018 windows retaining any detection (Window_F1 > 0.1):")
print(tgt[(tgt.Year==2018) & (tgt.Window_F1 > 0.1)]
        [["TimeWindow","Window_F1","n_attack","n_benign"]].to_string(index=False))
# FINDING: detection is intact on 2018-02-15 11:00-11:30 (F1 ~0.97-0.999),
# partial on 02-15 09:15-09:45 (0.27-0.33) and on 02-22/02-23 (0.20-0.65),
# and collapsed everywhere after. Three regimes, not a smooth decay curve.


# ============================================================
# TIER 2 — ARF online baseline  (isolated; drift runs regardless)
# ============================================================
ARF_OK = False
try:
    try:
        import river
    except ImportError:
        # --no-deps is REQUIRED: a plain install upgrades numpy and breaks
        # scipy against Kaggle's precompiled wheels.
        os.system("pip install -q river --no-deps")
        import river
    import scipy.stats                      # confirm the env still works
    try:
        from river.forest import ARFClassifier          # river >= 0.21
    except ImportError:
        from river.ensemble import AdaptiveRandomForestClassifier as ARFClassifier
    from river import metrics as rmetrics, preprocessing as rprep, compose
    ARF_OK = True
    print("\nriver", river.__version__)
except Exception as e:
    print("\nTIER 2 SKIPPED — river unavailable:", repr(e))

if ARF_OK:
    try:
        prev = P["arf_ckpt"] if os.path.exists(P["arf_ckpt"]) else _find("05_arf_state.pkl")
        if prev:
            st = pickle.load(open(prev, "rb"))
            model, rows, done = st["model"], st["rows"], set(st["done"])
            print(f"resumed ARF: {len(done)}/{len(WIN_DIRS)} windows")
        else:
            model = compose.Pipeline(
                rprep.StandardScaler(),     # ARF standardizes ONLINE — it is
                ARFClassifier(n_models=N_TREES, seed=SEED),  # meant to adapt, so
            )                               # it must NOT get the 2017 scaler
            rows, done = [], set()

        def save_arf():
            pickle.dump({"model": model, "rows": rows, "done": list(done)},
                        open(P["arf_ckpt"], "wb"))
            pd.DataFrame(rows).to_csv(P["arf"], index=False)

        for i, d in enumerate(WIN_DIRS):
            key = WIN_KEYS[i]
            if key in done:
                continue
            if budget_left() < 1800:
                print(f"TIME BUDGET — stopping Tier 2 at {key}")
                save_arf(); break

            df = read_window(d, cap=ARF_CAP)
            tp = tn = fp = fn = 0
            wf1 = rmetrics.F1()
            for rec in df.to_dict("records"):
                y = int(rec["y"])
                x = {k: float(rec[k]) for k in FEATURE_COLS}
                yp = model.predict_one(x)              # TEST ...
                yp = 0 if yp is None else int(yp)      # cold start
                wf1.update(y, yp)
                tp += (y == 1 and yp == 1); tn += (y == 0 and yp == 0)
                fp += (y == 0 and yp == 1); fn += (y == 1 and yp == 0)
                model.learn_one(x, y)                  # ... THEN train

            sens = tp/(tp+fn) if (tp+fn) else 0.0
            spec = tn/(tn+fp) if (tn+fp) else 0.0
            rows.append({"win_key": key, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
                         "ARF_F1": wf1.get(), "ARF_TSS": sens + spec - 1.0})
            done.add(key)
            print(f"[{len(done)}/{len(WIN_DIRS)}] {key} ARF_F1={wf1.get():.4f} ({elapsed()})")
            if len(done) % CKPT_EVERY == 0:
                save_arf()

        save_arf()
        arf = pd.DataFrame(rows).sort_values("win_key").reset_index(drop=True)
        # river's ARF adapts internally via ADWIN and exposes no stable public
        # "tree reset" event, so the windows of interest are found post-hoc:
        # a local F1 minimum that then recovers == damage taken BEFORE adapting.
        arf["is_decay_dip"] = (arf["ARF_F1"] < arf["ARF_F1"].shift(1)) & \
                              (arf["ARF_F1"] < arf["ARF_F1"].shift(-1))
        arf.to_csv(P["arf"], index=False)
        print("Tier 2 done:", len(arf), "windows |",
              int(arf["is_decay_dip"].sum()), "decay dips")
    except Exception as e:
        print("TIER 2 FAILED mid-run, continuing to drift:", repr(e))


# ============================================================
# TIER 3a — 2017 reference distribution
# ============================================================
ref_src = P["ref"] if os.path.exists(P["ref"]) else _find("04_drift_reference.pkl")
if ref_src:
    reference = pickle.load(open(ref_src, "rb"))
    print("\nreference loaded from", ref_src)
else:
    assert _p17, "2017 parquet not found — attach the raw CIC dataset"
    ref = pq.read_table(_p17, columns=FEATURE_COLS).to_pandas()
    ref = ref.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")
    if len(ref) > 200_000:
        ref = ref.sample(200_000, random_state=SEED)
    print("\nreference rows:", len(ref))

    reference = {}
    rng = np.random.RandomState(SEED)
    for c in FEATURE_COLS:
        v = ref[c].values
        edges = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3 or v.std() == 0:
            reference[c] = None                 # constant / degenerate feature
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        cnt, _ = np.histogram(v, bins=edges)
        reference[c] = {
            "edges": edges,
            "p":     cnt / max(cnt.sum(), 1),
            "mu":    float(v.mean()),
            "sd":    float(v.std()),
            "sample": v[rng.choice(len(v), size=min(20_000, len(v)), replace=False)],
        }
    pickle.dump(reference, open(P["ref"], "wb"))
    del ref

ACTIVE = [c for c in FEATURE_COLS if reference.get(c) is not None]
print(len(ACTIVE), "active features for drift")


# ============================================================
# TIER 3b — per-window drift vectors
# ============================================================
def psi(expected_p, actual_v, edges):
    cnt, _ = np.histogram(actual_v, bins=edges)
    q = np.clip(cnt / max(cnt.sum(), 1), EPS, None)
    p = np.clip(expected_p, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))

src = P["drift"] if os.path.exists(P["drift"]) else _find("06_drift_vectors.csv")
if src:
    prevd = pd.read_csv(src)
    drows, ddone = prevd.to_dict("records"), set(prevd["win_key"].astype(str))
    print(f"resumed drift: {len(ddone)} windows")
else:
    drows, ddone = [], set()

for i, d in enumerate(WIN_DIRS):
    key = WIN_KEYS[i]
    if key in ddone:
        continue
    if budget_left() < 300:
        print("TIME BUDGET — stopping drift computation"); break

    s = read_window(d)
    rec, psis, wds = {"win_key": key}, [], []
    for c in ACTIVE:
        r = reference[c]
        v = s[c].values.astype("float64")
        v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        p_ = psi(r["p"], v, r["edges"])
        # standardize by 2017 stats so distances compare ACROSS features
        w_ = wasserstein_distance((v - r["mu"]) / r["sd"],
                                  (r["sample"] - r["mu"]) / r["sd"])
        rec[f"psi__{c}"] = p_; rec[f"wd__{c}"] = w_
        psis.append(p_); wds.append(w_)

    if psis:
        rec.update({
            "psi_mean": float(np.mean(psis)), "psi_max": float(np.max(psis)),
            "psi_p90": float(np.percentile(psis, 90)),
            "psi_n_gt_0p25": int(np.sum(np.array(psis) > 0.25)),   # classic alarm
            "wd_mean": float(np.mean(wds)), "wd_max": float(np.max(wds)),
            "wd_p90": float(np.percentile(wds, 90)),
        })
    drows.append(rec); ddone.add(key)
    pd.DataFrame(drows).to_csv(P["drift"], index=False)    # save every window
    if i % 10 == 0:
        print(f"[{len(ddone)}/{len(WIN_DIRS)}] drift {key} ({elapsed()})")

drift = pd.DataFrame(drows)
print("drift vectors:", drift.shape)


# ============================================================
# SUMMARY — no forecaster is fitted here
# ============================================================
# With 108/112 2018 windows already collapsed and 2019 single-class, any
# classifier scores ~96% by predicting one label. The framing has to change
# before a model is fitted; ARF and the drift vectors above are
# framing-independent and are what this run exists to produce.
try:
    d2 = drift.merge(tgt[["win_key","Year","Window_F1","degraded"]],
                     on="win_key", how="inner")
    if "psi_mean" in d2.columns:
        print("\nDrift magnitude by year:")
        print(d2.groupby("Year")[["psi_mean","psi_max","wd_mean","wd_max"]].median())
        print("\nDrift vs degradation (does drift separate the classes at all?):")
        print(d2.groupby("degraded")[["psi_mean","psi_max","wd_mean","wd_max"]].median())
        print("\nSpearman corr with Window_F1:")
        for c in ["psi_mean","psi_max","wd_mean","wd_max"]:
            print(f"  {c}: {d2[c].corr(d2['Window_F1'], method='spearman'):+.3f}")
except Exception as e:
    print("summary skipped:", repr(e))

print("\n=== FILES ===")
for f in sorted(glob.glob(f"{OUT}/*")):
    sz = os.path.getsize(f)/1024 if os.path.isfile(f) else 0
    print(" ", f, f"{sz:.0f} KB")
print("TOTAL", elapsed())
