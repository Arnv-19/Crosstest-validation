# ============================================================
# TIER 0b — train the 2017 baseline detector
# ------------------------------------------------------------
# Produces the three artifacts every later tier depends on:
#   scaler_2017  (StandardScalerModel, fit on TRAIN ONLY)
#   rf_2017      (RandomForestClassificationModel, binary)
#   meta.json    (BENIGN_IDX, F1_BASE, feature list, row counts)
#
# Binary by design: the decay metric is binary F1 and ARF is a
# binary learner. The `family` column is for descriptive analysis
# in the paper, not for the detector.
#
# RUN ORDER: Spark-init cell -> tier0_common.py -> this file.
# At the end: SAVE VERSION, then attach this notebook's output as
# a dataset input to the Tier 1 notebook.
# ============================================================

# %% ---- cell 0 ---------------------------------------------
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json, os

OUT_DIR    = "/kaggle/working"
META_PATH  = f"{OUT_DIR}/model_meta.json"
TRAIN_FRAC = 0.8
FORCE_RETRAIN = False


# %% ---- cell 1: load + audit -------------------------------
df17 = load_year(PARQUET_2017, 2017).select("y", "family", *FEATURE_COLS).cache()

print("rows:", df17.count())
df17.groupBy("y").count().show()          # expect ~2.27M benign / ~0.56M attack
df17.groupBy("family").count().orderBy(F.desc("count")).show(20, False)
assert df17.where(F.col("family") == "UNMAPPED").count() == 0, \
    "unmapped labels — fix FAMILY in tier0 before training"
assert len(FEATURE_COLS) > 0, "FEATURE_COLS is empty"

# No StringIndexer needed: `y` is already 0=benign, 1=attack, so the predicted
# class index IS the label. This removes the BENIGN_IDX ambiguity entirely.
BENIGN_IDX = 0.0


# %% ---- cell 2: stratified chronological-safe split --------
# 2017 is the TRAINING year, so a random stratified split is the right
# choice here (the chronological evaluation happens on 2018/2019).
# Stratify by class so the 19.7% attack rate is preserved in both halves.
train = df17.sampleBy("y", {0: TRAIN_FRAC, 1: TRAIN_FRAC}, seed=SEED).cache()
test  = df17.join(train, on=df17.columns, how="left_anti").cache()

print(f"train={train.count():,}  test={test.count():,}")
train.groupBy("y").count().show()
test.groupBy("y").count().show()


# %% ---- cell 3: fit scaler on TRAIN ONLY -------------------
# Fitting on train only (not all of 2017) keeps the held-out F1_BASE honest.
# NOTE for the paper: StandardScaler is a monotonic per-feature transform and
# does not change RF split decisions; it is retained so Tier 3's Wasserstein
# distances are comparable across features.
if stage(SCALER_PATH, FORCE_RETRAIN):
    asm = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                          handleInvalid="keep")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)
    scaler_model = scaler.fit(asm.transform(train))
    scaler_model.write().overwrite().save(SCALER_PATH)
    mark_done(SCALER_PATH)

from pyspark.ml.feature import StandardScalerModel
asm = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                      handleInvalid="keep")
scaler_model = StandardScalerModel.load(SCALER_PATH)

def prep(d):
    return scaler_model.transform(asm.transform(d))

train_s, test_s = prep(train).cache(), prep(test).cache()


# %% ---- cell 4: train the RF -------------------------------
if stage(MODEL_PATH, FORCE_RETRAIN):
    rf = RandomForestClassifier(
        featuresCol="features", labelCol="y",
        numTrees=200, maxDepth=20,
        maxBins=64, subsamplingRate=0.8,
        featureSubsetStrategy="sqrt", seed=SEED,
    )
    rf_model = rf.fit(train_s)
    rf_model.write().overwrite().save(MODEL_PATH)
    mark_done(MODEL_PATH)

from pyspark.ml.classification import RandomForestClassificationModel
rf_model = RandomForestClassificationModel.load(MODEL_PATH)


# %% ---- cell 5: F1_BASE on the held-out split --------------
# This single number is subtracted in EVERY Tier-3 target
# (Delta_F1 = F1_BASE - Window_F1), so it must come from data the
# model never saw.
pred = rf_model.transform(test_s).select("y", "prediction").cache()

tp = pred.where((F.col("y")==1) & (F.col("prediction")==1)).count()
tn = pred.where((F.col("y")==0) & (F.col("prediction")==0)).count()
fp = pred.where((F.col("y")==0) & (F.col("prediction")==1)).count()
fn = pred.where((F.col("y")==1) & (F.col("prediction")==0)).count()

f1_base  = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) else 0.0
tss_base = (tp/(tp+fn) if (tp+fn) else 0) + (tn/(tn+fp) if (tn+fp) else 0) - 1
prec = tp/(tp+fp) if (tp+fp) else 0.0
rec  = tp/(tp+fn) if (tp+fn) else 0.0

print(f"TP={tp:,} TN={tn:,} FP={fp:,} FN={fn:,}")
print(f"F1_BASE={f1_base:.10f}  TSS={tss_base:.6f}  P={prec:.4f}  R={rec:.4f}")

# Sanity: a 2017 binary RF should land ~0.99. If it is far off, stop and
# investigate BEFORE scoring 35M rows in Tier 1.
assert f1_base > 0.90, f"baseline F1 unexpectedly low ({f1_base:.4f})"


# %% ---- cell 6: persist metadata ---------------------------
meta = {
    "f1_base": f1_base, "tss_base": tss_base,
    "precision": prec, "recall": rec,
    "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "benign_idx": BENIGN_IDX,
    "n_train": train.count(), "n_test": test.count(),
    "n_trees": 200, "max_depth": 20, "seed": SEED,
    "feature_cols": FEATURE_COLS,
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps({k: v for k, v in meta.items() if k != "feature_cols"}, indent=2))

# Feature importances — a useful table for the paper, and a cross-check
# against which features Tier 3's drift vector finds most predictive.
imp = sorted(zip(FEATURE_COLS, rf_model.featureImportances.toArray()),
             key=lambda t: -t[1])[:20]
for name, score in imp:
    print(f"  {score:.4f}  {name}")

print("\nNow: paste F1_BASE into tier0_common.py, then SAVE VERSION.")
print(f"F1_BASE = {f1_base!r}")
