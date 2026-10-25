# ============================================================
# TIER 6 — drift-signal battery + vectorised PSI benchmark
# ------------------------------------------------------------
# QUESTION: PSI saturated (alarm in 100% of windows, rho=0.028
# with time). Is that specific to PSI, or to every input-space
# drift signal? Five families are evaluated on identical
# windows against identical targets.
#
#   unbounded   PSI, Wasserstein          (baseline — saturates)
#   bounded     Jensen-Shannon, Hellinger (cannot run to infinity)
#   model-aware importance-weighted PSI   (drift in features the
#                                          detector ignores is irrelevant)
#   dynamics    EWMA velocity/acceleration(derivative, not level)
#   output      prediction entropy        (P(Y|X) proxy — the only
#                                          one that could catch Infiltration)
#
# Reruns NOTHING expensive: no RF training, no 35M-row pass.
# The one new scoring pass is over the per-window samples only.
#
# Run in a notebook with Spark + tier0 loaded.
# ============================================================

# %% ---- cell 0: locate artefacts ---------------------------
import glob, os, json, pickle, time
import numpy as np, pandas as pd
pd.set_option("display.width", 220)

def _find(name):
    h = (glob.glob(f"/kaggle/working/decay/{name}") or
         glob.glob(f"/kaggle/working/{name}") or
         glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not h:
        raise FileNotFoundError(name)
    return h[0]

DRIFT   = _find("06_drift_vectors.csv")
ARF     = _find("05_arf_metrics.csv")
TARGETS = _find("02_decay_targets.csv")
FAMREC  = _find("12_family_recall.csv")
SAMPLES = _find("03_window_samples")
FAMSAMP = _find("10_family_samples")
REF     = _find("04_drift_reference.pkl")
for p in (DRIFT, ARF, TARGETS, FAMREC, SAMPLES, FAMSAMP, REF):
    print(" ", p)

OUT = "/kaggle/working/decay"; os.makedirs(OUT, exist_ok=True)

dr  = pd.read_csv(DRIFT);  dr["win_key"]  = dr.win_key.astype(str)
arf = pd.read_csv(ARF);    arf["win_key"] = arf.win_key.astype(str)
tgt = pd.read_csv(TARGETS, parse_dates=["TimeWindow"])
tgt["win_key"] = tgt.TimeWindow.dt.strftime("%Y%m%d_%H%M")
famrec = pd.read_csv(FAMREC)
reference = pickle.load(open(REF, "rb"))
ACTIVE = [c for c in reference if reference.get(c) is not None]

base = (dr.merge(arf[["win_key","ARF_F1","is_decay_dip"]], on="win_key")
          .merge(tgt[["win_key","Year","Window_F1","n_flows","n_attack"]], on="win_key")
          .sort_values("win_key").reset_index(drop=True))
base["atk_frac"] = base.n_attack / base.n_flows
print(f"\n{len(base)} windows | {len(ACTIVE)} active features")


# %% ---- cell 1: DYNAMICS — EWMA velocity and acceleration --
# S_t = a*x_t + (1-a)*S_{t-1};  velocity = S_t - S_{t-1};
# acceleration = V_t - V_{t-1}. Computed WITHIN each year so the
# dataset boundary is not read as a derivative.
ALPHA = 0.3

def ewma_feats(g, cols=("psi_mean","psi_max","wd_mean")):
    g = g.sort_values("win_key").copy()
    for c in cols:
        s = g[c].ewm(alpha=ALPHA, adjust=False).mean()
        g[f"ewma_{c}"]  = s
        g[f"vel_{c}"]   = s.diff()
        g[f"acc_{c}"]   = s.diff().diff()
    return g

base = base.groupby("Year", group_keys=False).apply(ewma_feats)
DYN = [c for c in base.columns if c.startswith(("ewma_","vel_","acc_"))]
print("dynamics features:", DYN)
print(base.groupby("Year")[["vel_psi_mean","acc_psi_mean"]].describe().T.round(4))


# %% ---- cell 2: MODEL-AWARE — importance-weighted PSI ------
# Drift in a feature the detector barely uses cannot explain its
# failure. Weight each feature's PSI by the deployed model's reliance.
from pyspark.ml.classification import RandomForestClassificationModel
rf = RandomForestClassificationModel.load(MODEL_PATH)
imp = dict(zip(FEATURE_COLS, rf.featureImportances.toArray()))
pd.Series(imp).sort_values(ascending=False).head(10).to_csv(f"{OUT}/20_rf_importances.csv")
print("top features by importance:")
print(pd.Series(imp).sort_values(ascending=False).head(8).round(4).to_string())

psi_cols = [c for c in dr.columns if c.startswith("psi__")]
w = np.array([imp.get(c.replace("psi__",""), 0.0) for c in psi_cols])
w = w / w.sum() if w.sum() > 0 else w
base["psi_impw"] = (base[psi_cols].fillna(0).values * w).sum(axis=1)
# top-10-feature PSI: drift restricted to what the model leans on most
top10 = [f"psi__{k}" for k,_ in sorted(imp.items(), key=lambda t:-t[1])[:10]
         if f"psi__{k}" in dr.columns]
base["psi_top10"] = base[top10].fillna(0).mean(axis=1)
print(f"\npsi_impw  mean={base.psi_impw.mean():.3f} std={base.psi_impw.std():.3f}")
print(f"psi_top10 mean={base.psi_top10.mean():.3f} std={base.psi_top10.std():.3f}")
print("(compare unweighted psi_mean: "
      f"mean={base.psi_mean.mean():.3f} std={base.psi_mean.std():.3f})")


# %% ---- cell 3: BOUNDED divergences + VECTORISATION BENCHMARK
# Jensen-Shannon (base 2, bounded [0,1]) and Hellinger (bounded [0,1])
# from the SAME quantile bins as PSI, so the only thing that changes
# is the divergence, not the discretisation.
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

EPS = 1e-12
def _hist(v, edges):
    cnt, _ = np.histogram(v, bins=edges)
    return cnt / max(cnt.sum(), 1)

def js_div(p, q):
    p = np.clip(p, EPS, None); q = np.clip(q, EPS, None)
    m = 0.5*(p+q)
    kl = lambda a,b: np.sum(a*np.log2(a/b))
    return float(0.5*kl(p,m) + 0.5*kl(q,m))          # [0,1]

def hellinger(p, q):
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(p*q)))))   # [0,1]

def window_divergences(pdf):
    """One window -> bounded divergences, averaged over active features."""
    key = str(pdf["win_key"].iloc[0])
    js, he = [], []
    for c in ACTIVE:
        r = reference[c]
        v = pdf[c].values.astype("float64"); v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        q = _hist(v, r["edges"]); p = np.clip(r["p"], EPS, None)
        js.append(js_div(p, q)); he.append(hellinger(p, q))
    if not js:
        js, he = [np.nan], [np.nan]
    return pd.DataFrame([{"win_key": key,
                          "js_mean": float(np.nanmean(js)), "js_max": float(np.nanmax(js)),
                          "hell_mean": float(np.nanmean(he)), "hell_max": float(np.nanmax(he))}])

SCHEMA = StructType([StructField("win_key", StringType()),
                     StructField("js_mean", DoubleType()), StructField("js_max", DoubleType()),
                     StructField("hell_mean", DoubleType()), StructField("hell_max", DoubleType())])

sdf = spark.read.parquet(SAMPLES).select("win_key", *ACTIVE)

# --- (a) distributed: Arrow-backed grouped map across all cores
t0 = time.time()
div_spark = (sdf.groupBy("win_key").applyInPandas(window_divergences, schema=SCHEMA)
                .toPandas())
t_spark = time.time() - t0
print(f"applyInPandas (Arrow, distributed): {t_spark:6.1f}s for {len(div_spark)} windows")

# --- (b) single-threaded pandas loop over the same partitions, for the benchmark
import pyarrow.parquet as pq
dirs = sorted(glob.glob(os.path.join(SAMPLES, "win_key=*")))[:20]   # 20 windows is enough
t0 = time.time()
_ = [window_divergences(pq.read_table(d).to_pandas().assign(
         win_key=d.split("win_key=")[-1])) for d in dirs]
t_loop_20 = time.time() - t0
t_loop_est = t_loop_20 * len(div_spark) / len(dirs)
print(f"pandas loop: {t_loop_20:6.1f}s for {len(dirs)} windows "
      f"-> {t_loop_est:6.1f}s projected for {len(div_spark)}")
print(f"SPEEDUP: {t_loop_est/max(t_spark,1e-9):.1f}x")
pd.DataFrame([{"method":"applyInPandas_arrow","windows":len(div_spark),"seconds":t_spark},
              {"method":"pandas_loop_projected","windows":len(div_spark),"seconds":t_loop_est}]
            ).to_csv(f"{OUT}/21_vectorisation_benchmark.csv", index=False)

base = base.merge(div_spark, on="win_key", how="left")
print("\nbounded divergences (are they saturated too?):")
print(base.groupby("Year")[["js_mean","js_max","hell_mean","hell_max"]].describe().T.round(3))
# js_mean near its 1.0 ceiling in every window = bounded metrics saturate as well.


# %% ---- cell 4: OUTPUT-SPACE — prediction entropy ----------
# The only signal here that reflects P(Y|X). Scores the per-window
# SAMPLES (~millions of rows), not the full 35M population.
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.functions import vector_to_array

ENT = f"{OUT}/22_window_entropy.csv"
if not os.path.exists(ENT):
    asm = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                          handleInvalid="keep")
    sc_m = StandardScalerModel.load(SCALER_PATH)
    s = spark.read.parquet(SAMPLES).select("win_key","y",*FEATURE_COLS)
    p = rf.transform(sc_m.transform(asm.transform(s)))
    p = p.withColumn("p1", vector_to_array("probability")[1])
    p = p.withColumn("ent", -(F.col("p1")*F.log2(F.col("p1")+F.lit(EPS)) +
                              (1-F.col("p1"))*F.log2(1-F.col("p1")+F.lit(EPS))))
    (p.groupBy("win_key").agg(
        F.mean("ent").alias("ent_mean"), F.stddev("ent").alias("ent_std"),
        F.mean(F.abs(F.col("p1")-F.lit(0.5))).alias("conf_mean"),
        F.mean((F.col("ent") > 0.5).cast("double")).alias("frac_uncertain"))
      .toPandas().to_csv(ENT, index=False))
    print("entropy computed")
ent = pd.read_csv(ENT); ent["win_key"] = ent.win_key.astype(str)
base = base.merge(ent, on="win_key", how="left")
print(base.groupby("Year")[["ent_mean","conf_mean","frac_uncertain"]].median().round(4))

# per-FAMILY entropy — the Infiltration test
FAMENT = f"{OUT}/23_family_entropy.csv"
if not os.path.exists(FAMENT):
    asm = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                          handleInvalid="keep")
    sc_m = StandardScalerModel.load(SCALER_PATH)
    fs = spark.read.parquet(FAMSAMP).where("y = 1 AND Year = 2018")
    fp = rf.transform(sc_m.transform(asm.transform(fs)))
    fp = fp.withColumn("p1", vector_to_array("probability")[1])
    fp = fp.withColumn("ent", -(F.col("p1")*F.log2(F.col("p1")+F.lit(EPS)) +
                                (1-F.col("p1"))*F.log2(1-F.col("p1")+F.lit(EPS))))
    (fp.groupBy("family").agg(
        F.count("*").alias("n"), F.mean("ent").alias("ent_mean"),
        F.mean("p1").alias("p_attack_mean"),
        F.mean((F.col("ent") > 0.5).cast("double")).alias("frac_uncertain"))
       .toPandas().to_csv(FAMENT, index=False))
fament = pd.read_csv(FAMENT)
fam = (fament.merge(famrec[famrec.Year==2018][["family","recall","n_attack"]], on="family")
             .sort_values("recall", ascending=False))
print("\n=== PER-FAMILY OUTPUT-SPACE SIGNAL vs SURVIVAL ===")
print(fam.to_string(index=False))


# %% ---- cell 5: the battery table --------------------------
FAMILIES = {
 "unbounded":  ["psi_mean","psi_max","wd_mean","wd_max"],
 "bounded":    ["js_mean","js_max","hell_mean","hell_max"],
 "model_aware":["psi_impw","psi_top10"],
 "dynamics":   ["vel_psi_mean","acc_psi_mean","vel_wd_mean"],
 "output":     ["ent_mean","conf_mean","frac_uncertain"],
}
ALL = [c for v in FAMILIES.values() for c in v if c in base.columns]

b = base.copy()
b["f1_next"] = b.groupby("Year").ARF_F1.shift(-1)
rows = []
for famname, cols in FAMILIES.items():
    for c in cols:
        if c not in b.columns: continue
        rows.append({
            "family": famname, "metric": c,
            "std": b[c].std(),
            "rho_ARF_F1":   b[c].corr(b.ARF_F1, method="spearman"),
            "rho_next_F1":  b[c].corr(b.f1_next, method="spearman"),
            "rho_atk_frac": b[c].corr(b.atk_frac, method="spearman"),
        })
bat = pd.DataFrame(rows)
bat["confounded"] = bat.rho_atk_frac.abs() > bat.rho_ARF_F1.abs()
print("\n=== DRIFT-SIGNAL BATTERY (window level, n=%d) ===" % len(b))
print(bat.round(3).to_string(index=False))
print("""
rho_atk_frac larger than rho_ARF_F1 (confounded=True) means the metric tracks
workload composition more closely than performance — the failure mode already
shown for PSI, where attack fraction carried 0.957 of a fitted model.""")

bat.to_csv(f"{OUT}/24_drift_battery.csv", index=False)
base.to_csv(f"{OUT}/25_battery_window_features.csv", index=False)
print("\nsaved 24_drift_battery.csv, 25_battery_window_features.csv")
