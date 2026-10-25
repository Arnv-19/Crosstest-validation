# ============================================================
# TIER 1 FIX — regenerate decay targets correctly
# ------------------------------------------------------------
# Reuses the already-trained 2017 RF + the 2017-fit scaler.
# Only the evaluation + target generation is redone.
#
# Design: SCORE ONCE (cell 2, expensive) -> write per-flow
# predictions. Then WINDOW + FILTER + TARGETS (cell 3, cheap)
# is re-runnable at any window size without re-scoring.
#
# Split into Kaggle cells at "# %% ----".
# ============================================================


# %% ---- 0. Config ------------------------------------------
PARQUET_2018 = "/kaggle/input/.../2018"      # year-wise parquet dirs
PARQUET_2019 = "/kaggle/input/.../2019"

MODEL_PATH   = "/kaggle/working/rf_2017"      # saved RandomForestClassificationModel
SCALER_PATH  = "/kaggle/working/scaler_2017"  # saved StandardScalerModel (fit on 2017)

TS_COL       = "Timestamp"                     # raw timestamp column
LABEL_COL    = "Label"                         # original label (string or int)
FEATURE_COLS = [ ]                             # <-- same features used in training

BENIGN_VALUE = "Benign"      # value of LABEL_COL meaning benign  (<-- verify)
BENIGN_IDX   = 0.0           # predicted class index that means benign (<-- verify
                             #     against your StringIndexer labels; benign is the
                             #     majority class so it's usually index 0.0)

WINDOW_HOURS = 2             # 4 was too coarse. Try 1 and 2, compare target spread.
MIN_FLOWS    = 500           # drop windows with fewer flows than this (noisy F1)

# Base F1 on the 2017 held-out test set. UNCHANGED because the model is unchanged,
# so reusing your existing value is correct. (It's the 0.9949589... baked into your
# old Delta_F1 where window F1 was 0.) Recompute only if you retrain the RF.
F1_BASE = 0.9949589274191449

PRED_PATH    = "/kaggle/working/tier1_flow_predictions"   # stage-A output
OUT_TARGETS  = "/kaggle/working/tier1_decay_targets.csv"  # stage-B output

from pyspark.sql import functions as F


# %% ---- 1. Diagnostic (run first, ~minutes) ----------------
# Quantifies the 3 issues before you trust anything downstream.
for yr, path in [("2018", PARQUET_2018), ("2019", PARQUET_2019)]:
    d = spark.read.parquet(path).withColumn(
        "_ts", F.to_timestamp(F.col(TS_COL))      # <-- add a format str if needed,
    )                                             #     e.g. "d/M/yyyy h:mm:ss a"
    total = d.count()
    bad_ts = d.where(F.col("_ts").isNull() | (F.year("_ts") < 2017)).count()
    atk = d.where(F.col(LABEL_COL) != BENIGN_VALUE).count()
    print(f"[{yr}] rows={total:,}  bad/epoch_ts={bad_ts:,}  "
          f"attack%={100*atk/total:.2f}")
# If bad_ts is ~millions for 2019, your timestamp FORMAT is wrong (the 1970 bug) —
# fix the to_timestamp format before scoring, or those flows never window correctly.

# Is the frozen scaler nuking 2019 into a dead range? Check a couple of features
# after scaling (do this inside cell 2 once `scaled` exists, or standalone).


# %% ---- 2. STAGE A: score every flow ONCE (expensive) ------
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScalerModel
from pyspark.ml.classification import RandomForestClassificationModel

scaler = StandardScalerModel.load(SCALER_PATH)
rf     = RandomForestClassificationModel.load(MODEL_PATH)
assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
# NOTE: if you saved ONE PipelineModel (assembler+scaler+rf together), delete the
# three objects above and replace the per-dataset block with:
#     scored = PipelineModel.load(MODEL_PATH).transform(df)

def score(path, year):
    df = (spark.read.parquet(path)
            .withColumn("_ts", F.to_timestamp(F.col(TS_COL)))
            .where(F.col("_ts").isNotNull() & (F.year("_ts") >= 2017)))  # drop 1970
    df = assembler.transform(df)
    df = scaler.transform(df).withColumnRenamed("features_raw_scaled", "features")
    #    ^ match scaler's inputCol/outputCol to how you fit it
    scored = rf.transform(df)
    return scored.select(
        F.col("_ts").alias("ts"),
        F.lit(year).alias("Year"),
        (F.col(LABEL_COL) != BENIGN_VALUE).cast("int").alias("y_true"),
        (F.col("prediction") != F.lit(BENIGN_IDX)).cast("int").alias("y_pred"),
    )

(score(PARQUET_2018, 2018).unionByName(score(PARQUET_2019, 2019))
   .write.mode("overwrite").parquet(PRED_PATH))
print("stage A done ->", PRED_PATH)


# %% ---- 3. STAGE B: window + filter + targets (cheap) ------
# Re-run this alone with a different WINDOW_HOURS to compare.
preds = spark.read.parquet(PRED_PATH)

win = F.window(F.col("ts"), f"{WINDOW_HOURS} hours")
agg = (preds.groupBy("Year", win.alias("w")).agg(
          F.sum(((F.col("y_true") == 1) & (F.col("y_pred") == 1)).cast("int")).alias("TP"),
          F.sum(((F.col("y_true") == 0) & (F.col("y_pred") == 0)).cast("int")).alias("TN"),
          F.sum(((F.col("y_true") == 0) & (F.col("y_pred") == 1)).cast("int")).alias("FP"),
          F.sum(((F.col("y_true") == 1) & (F.col("y_pred") == 0)).cast("int")).alias("FN"),
      )
      .withColumn("TimeWindow", F.col("w.start"))
      .drop("w"))

# small now (hundreds of rows) -> pandas for the metric math + filtering
import pandas as pd
df = agg.toPandas().sort_values(["Year", "TimeWindow"]).reset_index(drop=True)

df["n_flows"]  = df[["TP", "TN", "FP", "FN"]].sum(axis=1)
df["n_attack"] = df["TP"] + df["FN"]

# THE TWO FILTERS THAT FIX THE TARGET:
df = df[df["n_attack"] > 0]                 # no positives -> no measurable decay
df = df[df["n_flows"] >= MIN_FLOWS]         # too few flows -> noisy F1
df = df.reset_index(drop=True)

tp, tn, fp, fn = df["TP"], df["TN"], df["FP"], df["FN"]
df["Window_F1"]  = (2*tp / (2*tp + fp + fn)).fillna(0.0)
sens = (tp / (tp + fn)).fillna(0.0)
spec = (tn / (tn + fp)).fillna(0.0)
df["Window_TSS"] = sens + spec - 1.0
df["Delta_F1"]   = F1_BASE - df["Window_F1"]

df.to_csv(OUT_TARGETS, index=False)
print(f"saved {len(df)} windows ({WINDOW_HOURS}h) -> {OUT_TARGETS}")


# %% ---- 4. Did it actually fix the variance? ---------------
d = df["Delta_F1"]
ceiling = (d > F1_BASE - 1e-3).mean()       # fraction pinned at max decay
print(f"Delta_F1  n={len(d)}  std={d.std():.4f}  "
      f"min={d.min():.3f}  max={d.max():.3f}  %at_ceiling={100*ceiling:.1f}%")
print(d.describe())
# Goal: std well above ~0.05 and %at_ceiling well under ~50%. If still ceiling-heavy,
# drop to WINDOW_HOURS=1 and confirm the early-2018 attack days are well sampled —
# that's where the real decay gradient lives.
