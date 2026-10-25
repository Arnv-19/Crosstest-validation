# ============================================================
# TIER 2 — Adaptive Random Forest (ARF) online baseline
# ------------------------------------------------------------
# Online continual-learning baseline for the drift study.
# ARF re-evaluates itself prequentially (test-then-train) on the
# same chronological windows used in Tier 1, so its per-window
# F1 / TSS is directly comparable to the static RF.
#
# Split into Kaggle notebook cells at the "# %% ----" markers.
# river >= 0.21 assumed. Runs CPU-only.
# ============================================================


# %% ---- 0. Config ------------------------------------------
# RAW numeric feature columns (NOT the frozen-2017-scaled ones).
# ARF standardizes online itself, so give it the unscaled values.
FEATURE_COLS = [
    # "Flow Duration", "Tot Fwd Pkts", ...  <-- fill in your columns
]
LABEL_COL   = "Label"          # binary int: 0 = Benign, 1 = any attack
WIN_COL     = "win_id"         # chronological window id (see cell 1)

PER_CLASS_CAP = 25_000         # max rows per (window, class). Keeps the
                               # 19.5M-attack DDoS window from choking river.
N_TREES = 10                   # river ARF is online+pure-Python. 10-25,
                               # NOT 200. 200 will not finish on Kaggle.
SEED = 42

SAMPLE_DIR = "/kaggle/working/arf_stream"      # Spark writes here
OUT_METRICS = "/kaggle/working/tier2_arf_window_metrics.csv"


# %% ---- 1. Spark: build the chronological, capped stream ----
# Run this against the SAME harmonized dataframe you windowed in
# Tier 1. Expectation: `df` has the raw FEATURE_COLS, a binary
# LABEL_COL, and a 4h (or finer) window start column.
#
# If you move to finer windows to fix the Tier-3 target variance,
# change the window derivation here and in Tier 1 together.

from pyspark.sql import functions as F, Window as W

# derive a lexically-sortable window id from the window-start timestamp
# (ISO string sorts == chronological order). Adjust `ts_col` to yours.
ts_col = "TimeWindow"
df = df.withColumn(
    WIN_COL,
    F.date_format(F.col(ts_col).cast("timestamp"), "yyyy-MM-dd_HH")
).where(F.col(WIN_COL).isNotNull())          # drops the 1970/null-ts rows

# per-(window, class) reservoir cap via random row_number
w = W.partitionBy(WIN_COL, LABEL_COL).orderBy(F.rand(SEED))
stream = (
    df.withColumn("_rn", F.row_number().over(w))
      .where(F.col("_rn") <= PER_CLASS_CAP)
      .drop("_rn")
      .select(WIN_COL, LABEL_COL, *FEATURE_COLS)
)

# one folder per window -> river reads them in sorted order
(stream.write.mode("overwrite")
       .partitionBy(WIN_COL)
       .parquet(SAMPLE_DIR))

# quick sanity: how many windows / rows survived the cap
stream.groupBy(WIN_COL, LABEL_COL).count().orderBy(WIN_COL).show(200, False)


# %% ---- 2. river: prequential ARF over the window stream -----
import glob, os
import pyarrow.parquet as pq
from river import metrics, preprocessing, compose

# ARF import name moved across river versions — handle both.
try:
    from river.forest import ARFClassifier            # river >= 0.21
except ImportError:
    from river.ensemble import AdaptiveRandomForestClassifier as ARFClassifier

model = compose.Pipeline(
    preprocessing.StandardScaler(),                   # online standardization
    ARFClassifier(n_models=N_TREES, seed=SEED),
)

running_f1 = metrics.F1()                             # cumulative, positive class
rows = []

win_dirs = sorted(glob.glob(os.path.join(SAMPLE_DIR, f"{WIN_COL}=*")))
assert win_dirs, f"no window partitions found under {SAMPLE_DIR}"

for wd in win_dirs:
    win_label = wd.split(f"{WIN_COL}=")[-1]
    tp = tn = fp = fn = 0
    win_f1 = metrics.F1()

    tbl = pq.read_table(wd).to_pandas()
    for rec in tbl.to_dict("records"):
        y = int(rec[LABEL_COL])
        x = {k: rec[k] for k in FEATURE_COLS}

        y_pred = model.predict_one(x)                 # test ...
        if y_pred is None:                            # cold start
            y_pred = 0
        y_pred = int(y_pred)

        running_f1.update(y, y_pred)
        win_f1.update(y, y_pred)
        tp += (y == 1 and y_pred == 1)
        tn += (y == 0 and y_pred == 0)
        fp += (y == 0 and y_pred == 1)
        fn += (y == 1 and y_pred == 0)

        model.learn_one(x, y)                         # ... then train

    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    rows.append({
        "TimeWindow": win_label,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "ARF_F1": win_f1.get(),
        "ARF_TSS": sens + spec - 1.0,
        "ARF_running_F1": running_f1.get(),
    })
    print(f"{win_label}  F1={win_f1.get():.4f}  runF1={running_f1.get():.4f}")


# %% ---- 3. Save + flag the "dangerous decay" windows ---------
import pandas as pd

arf = pd.DataFrame(rows)

# river's ARF adapts internally via ADWIN; it does not expose a clean
# public "a tree just reset" timestamp across versions. So we locate the
# decay dips the research cares about post-hoc: a local F1 minimum that
# then recovers == the model adapted *after* taking damage.
arf["prev"] = arf["ARF_F1"].shift(1)
arf["next"] = arf["ARF_F1"].shift(-1)
arf["is_decay_dip"] = (arf["ARF_F1"] < arf["prev"]) & (arf["next"] > arf["ARF_F1"])

arf.drop(columns=["prev", "next"]).to_csv(OUT_METRICS, index=False)
print("saved", OUT_METRICS)
print("recovery dips (ARF took damage then adapted):",
      int(arf["is_decay_dip"].sum()))
