# ============================================================
# TIER 1 — static RF evaluation + decay target generation  [v3]
# ------------------------------------------------------------
# A: score every flow ONCE          (expensive, checkpointed)
# B: window + filter + targets      (cheap, re-runnable)
# C: capped per-window sample       (shared by Tier 2 and Tier 3)
#
# Requires tier0_common.py v3 in the cell above (is_attack(),
# load_year() attaching y/win/family, F1_BASE set).
# Nothing is retrained here.
# ============================================================

# %% ---- cell 0: setup --------------------------------------
from pyspark.sql import functions as F, Window as W
import pandas as pd, json

FORCE_A = FORCE_B = FORCE_C = False      # flip to redo a stage

# Use the EXACT feature list the model was fit on. Column order matters to
# VectorAssembler — a silent reorder gives plausible but wrong predictions.
MODEL_DIR = "/kaggle/input/<train-notebook-slug>"     # <-- committed model dataset
try:
    FEATURE_COLS = json.load(open(f"{MODEL_DIR}/model_meta.json"))["feature_cols"]
    MODEL_PATH   = f"{MODEL_DIR}/rf_2017"
    SCALER_PATH  = f"{MODEL_DIR}/scaler_2017"
    print("loaded feature list from model_meta.json:", len(FEATURE_COLS))
except FileNotFoundError:
    print("WARN: model_meta.json not found — using tier0 FEATURE_COLS "
          f"({len(FEATURE_COLS)}) and tier0 MODEL_PATH. Verify they match "
          "what the RF was trained on.")


# %% ---- cell 1: diagnostics (run once) ---------------------
for yr, path in [("2017", PARQUET_2017), ("2018", PARQUET_2018), ("2019", PARQUET_2019)]:
    raw = parse_ts(spark.read.parquet(path))
    total  = raw.count()
    bad_ts = raw.where(F.col("_ts").isNull() | (F.year("_ts") < 2017)).count()
    atk    = raw.where(is_attack() == 1).count()
    print(f"[{yr}] rows={total:,} bad_ts={bad_ts:,} ({100*bad_ts/total:.2f}%) "
          f"attack={atk:,} ({100*atk/total:.2f}%)")
# Expected: 2017 ~19.7%, 2018 ~15.0%, 2019 ~99.7% attack (0.29% benign).
# 2019 at exactly 100.00% means the benign label is not matching — stop.

audit_labels()          # every year must report unmapped=0


# %% ---- cell 2: STAGE A — score every flow once ------------
if stage(P["preds"], FORCE_A):
    from pyspark.ml.feature import VectorAssembler, StandardScalerModel
    from pyspark.ml.classification import RandomForestClassificationModel

    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                                handleInvalid="keep")
    scaler    = StandardScalerModel.load(SCALER_PATH)
    rf        = RandomForestClassificationModel.load(MODEL_PATH)

    def score(path, year):
        df  = load_year(path, year)          # supplies _ts, win, Year, y, family
        df  = assembler.transform(df)
        df  = scaler.transform(df)           # outputCol "features"
        out = rf.transform(df)
        return out.select(
            F.col("_ts").alias("ts"), "win", "Year",
            F.col("y").alias("y_true"),                        # 0=benign 1=attack
            (F.col("prediction") == 1.0).cast("int").alias("y_pred"),
        )

    (score(PARQUET_2018, 2018).unionByName(score(PARQUET_2019, 2019))
       .write.mode("overwrite").parquet(P["preds"]))
    mark_done(P["preds"])
    print("stage A complete")

# Paper note: StandardScaler is a monotonic per-feature transform and does not
# change Random Forest split decisions — the decay measured below is genuine
# distribution shift, not a preprocessing artifact. Scaling is retained so
# Tier 3's Wasserstein distances are comparable across features.


# %% ---- cell 3: STAGE B — window, filter, targets ----------
# Cheap and re-runnable: change WINDOW_HOURS, reset(P["targets"]), rerun this
# cell alone. Scoring stays cached.
if stage(P["targets"], FORCE_B):
    preds = spark.read.parquet(P["preds"])
    preds = preds.withColumn("win", F.window(F.col("ts"), f"{WINDOW_HOURS} hours").start)

    agg = preds.groupBy("Year", "win").agg(
        F.sum(((F.col("y_true")==1)&(F.col("y_pred")==1)).cast("int")).alias("TP"),
        F.sum(((F.col("y_true")==0)&(F.col("y_pred")==0)).cast("int")).alias("TN"),
        F.sum(((F.col("y_true")==0)&(F.col("y_pred")==1)).cast("int")).alias("FP"),
        F.sum(((F.col("y_true")==1)&(F.col("y_pred")==0)).cast("int")).alias("FN"),
    )

    df = agg.toPandas().rename(columns={"win": "TimeWindow"})
    df = df.sort_values(["Year", "TimeWindow"]).reset_index(drop=True)
    df["n_flows"]  = df[["TP","TN","FP","FN"]].sum(axis=1)
    df["n_attack"] = df["TP"] + df["FN"]
    df["n_benign"] = df["TN"] + df["FP"]

    before = len(df)
    df = df[(df["n_attack"] > 0) &
            (df["n_benign"] >= MIN_BENIGN) &       # 2019 is only 0.29% benign
            (df["n_flows"]  >= MIN_FLOWS)].reset_index(drop=True)
    print(f"windows: {before} -> {len(df)} after filters")
    print(df.groupby("Year").size())
    # Zero-attack windows: F1 is trivially 0, not decay signal — they were the
    # source of the old target being pinned at the F1_BASE ceiling.
    # Thin-benign windows: a single FP swings specificity, so TSS is unusable.

    tp, tn, fp, fn = df["TP"], df["TN"], df["FP"], df["FN"]
    df["Window_F1"]  = (2*tp / (2*tp + fp + fn)).fillna(0.0)
    sens = (tp/(tp+fn)).fillna(0.0); spec = (tn/(tn+fp)).fillna(0.0)
    df["Window_TSS"] = sens + spec - 1.0
    df["Delta_F1"]   = F1_BASE - df["Window_F1"]

    df.to_csv(P["targets"], index=False)
    mark_done(P["targets"], {"window_hours": WINDOW_HOURS, "n": len(df)})

tgt = pd.read_csv(P["targets"], parse_dates=["TimeWindow"])


# %% ---- cell 3b: TARGET HEALTH CHECK (gates Tier 3) --------
d = tgt["Delta_F1"]
ceil_frac = (d > F1_BASE - 1e-3).mean()
print(f"Delta_F1  n={len(d)}  std={d.std():.4f}  min={d.min():.3f}  "
      f"max={d.max():.3f}  %at_ceiling={100*ceil_frac:.1f}%")
print(tgt.groupby("Year")["Delta_F1"].describe())
print(tgt.groupby("Year")[["n_flows","n_attack","n_benign"]].median())

# Decision rule:
#   std >> 0.05, %at_ceiling < ~50%, n >= ~100  -> proceed to stage C
#   still ceiling-heavy -> WINDOW_HOURS = 1; reset(P["targets"]); rerun cell 3
#   very few 2019 windows survive -> 2019 becomes a qualitative severe-drift
#     case study and 2018 carries the quantitative evaluation


# %% ---- cell 4: STAGE C — shared capped window sample ------
# One sample serves BOTH Tier 2 (ARF stream) and Tier 3 (drift vectors),
# guaranteeing they describe identical windows.
if stage(P["samples"], FORCE_C):
    valid = set(pd.read_csv(P["targets"])["TimeWindow"].astype(str))

    allw = None
    for path, year in [(PARQUET_2018, 2018), (PARQUET_2019, 2019)]:
        f = load_year(path, year).select("win", "Year", "y", *FEATURE_COLS)
        allw = f if allw is None else allw.unionByName(f)

    allw = allw.where(
        F.date_format("win", "yyyy-MM-dd HH:mm:ss").isin(list(valid)))

    w = W.partitionBy("win", "y").orderBy(F.rand(SEED))
    (allw.withColumn("_rn", F.row_number().over(w))
         .where(F.col("_rn") <= PER_CLASS_CAP).drop("_rn")
         .withColumn("win_key", F.date_format("win", "yyyyMMdd_HHmm"))
         .write.mode("overwrite").partitionBy("win_key").parquet(P["samples"]))
    mark_done(P["samples"], {"cap": PER_CLASS_CAP})
    print("stage C complete")

# Paper note: per-(window,class) capping at PER_CLASS_CAP is a stated
# computational necessity (the peak DDoS window holds ~19.5M attack flows).
# It preserves chronological order and within-window class presence. Tier 1
# metrics above are computed on the FULL population, not the sample.
