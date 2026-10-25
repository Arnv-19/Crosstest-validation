# ============================================================
# TIER 5 — (A) second model family   (B) label-budget detection
# ------------------------------------------------------------
# A: is the collapse a property of Random Forests, or of the
#    data? Train an MLP and a logistic regression on the SAME
#    2017 split and measure per-family 2018 recall. If all three
#    collapse alike, the finding is about cross-testbed shift,
#    not about one inductive bias.
#
# B: the positive contribution. Unsupervised drift cannot detect
#    the failure at any budget of UNLABELLED data. Quantify how
#    few LABELLED flows per family are needed instead. Uses the
#    rule of three: observing 0 detections in n labelled attack
#    flows bounds recall below 3/n at 95% confidence.
#
# Run in the TIER-1 notebook (Spark + tier0 + model paths set).
# Part B needs no Spark and no new scoring.
# ============================================================

# %% ---- cell 0: paths --------------------------------------
import glob, os, json
import numpy as np, pandas as pd
from pyspark.sql import functions as F

_IN = "/kaggle/input/notebooks/arnav1909/5oct-t1/decay"     # adjust if needed
FAM_COUNTS = f"{_IN}/09_family_counts.csv"
REC_CSV    = f"{_IN}/12_family_recall.csv"
MLP_PATH   = f"{WORK}/16_mlp_2017"
LR_PATH    = f"{WORK}/17_logreg_2017"
MODELCMP   = f"{WORK}/18_model_family_comparison.csv"
LABELBUD   = f"{WORK}/19_label_budget.csv"
pd.set_option("display.width", 220)


# ============================================================
# PART A — second and third model family
# ============================================================
# %% ---- cell A1: train on the SAME 2017 split --------------
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import (MultilayerPerceptronClassifier,
                                       LogisticRegression,
                                       RandomForestClassificationModel)

asm    = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw",
                         handleInvalid="keep")
scaler = StandardScalerModel.load(SCALER_PATH)

def prep(df):
    return scaler.transform(asm.transform(df))

# Same split recipe as tier0b so the baselines are comparable.
d17   = load_year(PARQUET_2017, 2017).select("y", "family", *FEATURE_COLS)
train = d17.sampleBy("y", {0: 0.8, 1: 0.8}, seed=SEED)
test  = d17.join(train, on=d17.columns, how="left_anti")
train_s, test_s = prep(train).cache(), prep(test).cache()
print("train:", train_s.count(), "test:", test_s.count())

# NOTE: unlike the RF, these two models genuinely depend on the frozen 2017
# scaler — StandardScaler is monotonic so it cannot change a tree's splits,
# but it directly changes an MLP's and a linear model's decision surface.
# That makes them a real test of whether scaling choice drove the collapse.

if stage(MLP_PATH):
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features", labelCol="y",
        layers=[len(FEATURE_COLS), 32, 16, 2],
        maxIter=60, blockSize=256, seed=SEED)
    mlp.fit(train_s).write().overwrite().save(MLP_PATH)
    mark_done(MLP_PATH)
    print("MLP trained")

if stage(LR_PATH):
    lr = LogisticRegression(featuresCol="features", labelCol="y",
                            maxIter=100, regParam=0.01, elasticNetParam=0.0)
    lr.fit(train_s).write().overwrite().save(LR_PATH)
    mark_done(LR_PATH)
    print("LogReg trained")

from pyspark.ml.classification import (MultilayerPerceptronClassificationModel,
                                       LogisticRegressionModel)
MODELS = {
    "RandomForest": RandomForestClassificationModel.load(MODEL_PATH),
    "MLP":          MultilayerPerceptronClassificationModel.load(MLP_PATH),
    "LogReg":       LogisticRegressionModel.load(LR_PATH),
}

# in-distribution baseline F1 for each, on the held-out 2017 split
base = {}
for name, m in MODELS.items():
    p = m.transform(test_s).select("y", "prediction")
    tp = p.where("y=1 and prediction=1").count(); fn = p.where("y=1 and prediction=0").count()
    fp = p.where("y=0 and prediction=1").count()
    base[name] = 2*tp / (2*tp + fp + fn) if (2*tp+fp+fn) else 0.0
    print(f"{name:13s} 2017 held-out F1 = {base[name]:.4f}")


# %% ---- cell A2: per-family 2018 recall for each model -----
if stage(MODELCMP):
    d18 = prep(load_year(PARQUET_2018, 2018)).cache()
    out = []
    for name, m in MODELS.items():
        agg = (m.transform(d18)
                 .groupBy("family")
                 .agg(F.sum(((F.col("y")==1)&(F.col("prediction")==1)).cast("int")).alias("TP"),
                      F.sum(((F.col("y")==1)&(F.col("prediction")==0)).cast("int")).alias("FN"),
                      F.sum(((F.col("y")==0)&(F.col("prediction")==1)).cast("int")).alias("FP"))
                 .toPandas())
        agg["model"] = name
        out.append(agg)
        print(name, "scored")
    cmp = pd.concat(out, ignore_index=True)
    cmp = cmp[cmp.family != "Benign"]
    cmp["n_attack"] = cmp.TP + cmp.FN
    cmp["recall"]   = cmp.TP / cmp.n_attack.replace(0, np.nan)
    cmp.to_csv(MODELCMP, index=False)
    mark_done(MODELCMP)
    d18.unpersist()

cmp = pd.read_csv(MODELCMP)
piv = (cmp[cmp.n_attack >= 20]
         .pivot(index="family", columns="model", values="recall")
         .join(cmp.groupby("family").n_attack.first()))
print("\n=== PER-FAMILY 2018 RECALL, THREE MODEL FAMILIES ===")
print(piv.sort_values("n_attack", ascending=False).to_string())
print("\n2017 held-out F1:", {k: round(v,4) for k,v in base.items()})
print("""
If all three collapse on the same families, the failure is a property of the
cross-testbed shift, not of the Random Forest's inductive bias — which is the
objection a reviewer raises first.""")


# ============================================================
# PART B — label budget (the positive contribution)
# ============================================================
# %% ---- cell B1: how few labels reveal the collapse? -------
# No Spark, no new scoring: resampled from the known per-family confusion
# counts already computed in Tier 4.
fc  = pd.read_csv(FAM_COUNTS)
atk = fc[fc.family != "Benign"].groupby("family")[["TP","FN"]].sum().reset_index()
atk["n_attack"] = atk.TP + atk.FN
atk["true_recall"] = atk.TP / atk.n_attack
atk = atk[atk.n_attack >= 20]

rng = np.random.RandomState(SEED)
BUDGETS = [10, 25, 50, 100, 250, 500]
ALARM   = 0.50          # "detector has failed on this family" if recall < 50%
TRIALS  = 2000

rows = []
for _, r in atk.iterrows():
    p = r.true_recall
    for n in BUDGETS:
        # draw n labelled attack flows of this family, count detections
        hits = rng.binomial(n, p, size=TRIALS)
        obs  = hits / n
        # Wilson 95% upper bound on recall from the sample
        z = 1.96
        den = 1 + z**2/n
        centre = (obs + z**2/(2*n)) / den
        half   = z*np.sqrt(obs*(1-obs)/n + z**2/(4*n**2)) / den
        upper  = centre + half
        rows.append({"family": r.family, "true_recall": p, "budget": n,
                     "mean_obs_recall": obs.mean(),
                     "P_correctly_flags_failure": float((upper < ALARM).mean())
                                                   if p < ALARM else np.nan})
lb = pd.DataFrame(rows)
print("\n=== LABEL BUDGET: P(correctly concluding 'failed') ===")
print(lb.pivot(index="family", columns="budget",
               values="P_correctly_flags_failure").round(3).to_string())

# The clean closed-form result worth quoting in the abstract.
print("""
Rule of three: observing ZERO detections in n labelled attack flows bounds the
true recall below 3/n at 95% confidence. So:""")
for n in [10, 30, 60, 100, 300]:
    print(f"  n = {n:>3} labelled flows  ->  recall < {3/n:.3f} at 95% confidence")
print("""
Contrast with the unsupervised side: PSI and Wasserstein, at window resolution
AND at family resolution, against a global AND a family-matched reference, give
no usable ranking of which families failed — at ANY volume of unlabelled data.
A few dozen labels per family do what unlimited unlabelled traffic cannot.""")

lb.to_csv(LABELBUD, index=False)
print("saved", MODELCMP, "and", LABELBUD)
