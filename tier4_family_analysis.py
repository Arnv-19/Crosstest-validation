# ============================================================
# TIER 4 — family-resolved decay and drift
# ------------------------------------------------------------
# Tests the paper's central claim:
#   decay is attack-SUBTYPE structured, not temporal, and
#   subtype-resolved drift separates survivors from collapses
#   where window-aggregate drift cannot.
#
# Runs in the TIER-1 notebook (needs Spark + tier0 v5, which
# already attaches `family` in load_year).
#
# ONE pass over 2018+2019. Cheaper than Stage A: it aggregates
# instead of writing 35M scored rows, and samples per family
# for the drift half in the same pass.
# ============================================================

# %% ---- cell 1: score once, aggregate per (family, window) --
from pyspark.sql import functions as F, Window as W
import pandas as pd, numpy as np, json, os

FAM_COUNTS  = f"{WORK}/09_family_counts.csv"
FAM_SAMPLES = f"{WORK}/10_family_samples"
FAM_PER_CAP = 20_000        # rows per family per year, for drift

if stage(FAM_COUNTS):
    from pyspark.ml.feature import VectorAssembler, StandardScalerModel
    from pyspark.ml.classification import RandomForestClassificationModel

    assembler = VectorAssembler(inputCols=FEATURE_COLS,
                                outputCol="features_raw", handleInvalid="keep")
    scaler = StandardScalerModel.load(SCALER_PATH)
    rf     = RandomForestClassificationModel.load(MODEL_PATH)

    scored = None
    for path, year in [(PARQUET_2018, 2018), (PARQUET_2019, 2019)]:
        df  = load_year(path, year)                 # supplies y, family, win
        df  = scaler.transform(assembler.transform(df))
        out = rf.transform(df).select(
            "win", "Year", "family", "y",
            (F.col("prediction") == 1.0).cast("int").alias("y_pred"),
            *FEATURE_COLS)
        scored = out if scored is None else scored.unionByName(out)
    scored = scored.cache()

    # (a) per-family confusion, aggregated — this is the headline table
    (scored.groupBy("Year", "family", "win").agg(
        F.count("*").alias("n"),
        F.sum(((F.col("y")==1)&(F.col("y_pred")==1)).cast("int")).alias("TP"),
        F.sum(((F.col("y")==1)&(F.col("y_pred")==0)).cast("int")).alias("FN"),
        F.sum(((F.col("y")==0)&(F.col("y_pred")==1)).cast("int")).alias("FP"),
        F.sum(((F.col("y")==0)&(F.col("y_pred")==0)).cast("int")).alias("TN"),
     ).toPandas().to_csv(FAM_COUNTS, index=False))
    mark_done(FAM_COUNTS)
    print("family counts written")

    # (b) per-family feature sample, for subtype-resolved drift
    if stage(FAM_SAMPLES):
        w = W.partitionBy("Year", "family").orderBy(F.rand(SEED))
        (scored.withColumn("_rn", F.row_number().over(w))
               .where(F.col("_rn") <= FAM_PER_CAP).drop("_rn")
               .select("Year", "family", "y", *FEATURE_COLS)
               .write.mode("overwrite").partitionBy("family").parquet(FAM_SAMPLES))
        mark_done(FAM_SAMPLES)
        print("family samples written")
    scored.unpersist()


# %% ---- cell 2: per-family recall (the核 claim) -------------
fc = pd.read_csv(FAM_COUNTS)
atk = fc[fc.family != "Benign"].copy()

fam = (atk.groupby(["Year","family"])[["TP","FN"]].sum().reset_index())
fam["n_attack"] = fam.TP + fam.FN
fam["recall"]   = fam.TP / fam.n_attack.replace(0, np.nan)
fam = fam[fam.n_attack >= 20].sort_values(["Year","recall"], ascending=[True,False])

pd.set_option("display.width", 200)
print("=== PER-FAMILY RECALL OF THE STATIC 2017 DETECTOR ===")
print(fam[["Year","family","n_attack","TP","recall"]].to_string(index=False))

# Is decay bimodal across families rather than graded?
r = fam[fam.Year == 2018]["recall"].dropna()
print(f"\n2018 families: {len(r)} | recall >0.5: {(r>0.5).sum()} "
      f"| <0.05: {(r<0.05).sum()} | in between: {((r>=0.05)&(r<=0.5)).sum()}")
print("If the middle bucket is near-empty, decay is a step function per family,")
print("not a graded curve — the paper's first claim.")


# %% ---- cell 3: was the family in 2017 training? ------------
# The sharpest result: families the 2017 model WAS trained on that still
# collapse (e.g. Hulk/SlowHTTPTest) — label parity without behavioural parity.
trained_2017 = {"DoS","DDoS","BruteForce","Web","Bot","Infiltration","Recon","Other"}
fam["in_2017_training"] = fam.family.isin(trained_2017)
print("\n=== LABEL PARITY vs BEHAVIOURAL PARITY ===")
print(fam[fam.Year==2018][["family","in_2017_training","n_attack","recall"]]
        .to_string(index=False))

# finer cut: the raw sub-labels behind each family, by day
atk["day"] = pd.to_datetime(atk.win).dt.date
print("\nper-day recall (maps onto the published 2018 attack schedule):")
day = atk.groupby(["Year","day"])[["TP","FN"]].sum().reset_index()
day["recall"] = day.TP / (day.TP + day.FN).replace(0, np.nan)
print(day.to_string(index=False))


# %% ---- cell 4: subtype-resolved drift ----------------------
# THE FIX. Window-aggregate drift saturated (PSI alarm in 100% of windows,
# rho=0.028 with time). Does drift computed PER FAMILY separate survivors
# from collapses?
import glob, pickle
from scipy.stats import wasserstein_distance
import pyarrow.parquet as pq

reference = load_pickle(P["ref"]) if os.path.exists(P["ref"]) else None
assert reference, "run the Tier-3 reference step first (04_drift_reference.pkl)"
ACTIVE = [c for c in FEATURE_COLS if reference.get(c) is not None]
N_BINS, EPS = 10, 1e-6

def psi(expected_p, actual_v, edges):
    cnt, _ = np.histogram(actual_v, bins=edges)
    q = np.clip(cnt / max(cnt.sum(), 1), EPS, None)
    p = np.clip(expected_p, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))

rows = []
for d in sorted(glob.glob(f"{FAM_SAMPLES}/family=*")):
    f = d.split("family=")[-1]
    if f == "Benign":
        continue
    s = pq.read_table(d).to_pandas()
    s = s[s.y == 1]                       # attack flows of this family only
    if len(s) < 200:
        continue
    psis, wds = [], []
    for c in ACTIVE:
        r = reference[c]
        v = s[c].values.astype("float64"); v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        psis.append(psi(r["p"], v, r["edges"]))
        wds.append(wasserstein_distance((v - r["mu"]) / r["sd"],
                                        (r["sample"] - r["mu"]) / r["sd"]))
    rows.append({"family": f, "n": len(s),
                 "psi_mean": np.mean(psis), "psi_max": np.max(psis),
                 "psi_n_gt_0p25": int(np.sum(np.array(psis) > 0.25)),
                 "wd_mean": np.mean(wds), "wd_max": np.max(wds)})

fd = pd.DataFrame(rows)
merged = fd.merge(fam[fam.Year==2018][["family","recall","n_attack"]],
                  on="family", how="inner")
print("\n=== SUBTYPE-RESOLVED DRIFT vs SURVIVAL ===")
print(merged.sort_values("recall", ascending=False).to_string(index=False))

if len(merged) >= 4:
    print("\nSpearman(drift, recall) ACROSS FAMILIES:")
    for c in ["psi_mean","psi_max","psi_n_gt_0p25","wd_mean","wd_max"]:
        print(f"  {c}: {merged[c].corr(merged['recall'], method='spearman'):+.3f}")
    print("\nCompare against the window-aggregate result (psi_mean vs ARF_F1 = +0.670,")
    print("which the ablation showed was 96% attack-fraction). A strong NEGATIVE")
    print("correlation here — more drift, lower recall — is the demonstrated fix:")
    print("drift is informative once it is resolved by attack subtype.")

merged.to_csv(f"{WORK}/11_family_drift_vs_recall.csv", index=False)
fam.to_csv(f"{WORK}/12_family_recall.csv", index=False)
print("\nsaved 11_family_drift_vs_recall.csv, 12_family_recall.csv")
