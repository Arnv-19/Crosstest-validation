# ============================================================
# TIER 1 — STAGE B (replacement for cell 3)
# ------------------------------------------------------------
# The old cell re-scanned 35M prediction rows on every run, which
# is why it took ~1h. This version scans ONCE at 15-minute
# granularity, caches the counts, and then derives any window
# size from that cache in pandas — instant, no Spark.
#
# 15 min is the atomic unit: every window size you care about
# (30m, 1h, 2h, 4h) is a whole multiple of it, so re-windowing is
# exact, not approximate.
# ============================================================

# %% ---- cell 3a: ONE scan -> 15-minute counts --------------
import pandas as pd, numpy as np
from pyspark.sql import functions as F

COUNTS_15 = f"{WORK}/02a_counts_15min.csv"

if stage(COUNTS_15):
    preds = spark.read.parquet(P["preds"])
    base = (preds
        .withColumn("w15", F.window(F.col("ts"), "15 minutes").start)
        .groupBy("Year", "w15")
        .agg(
            F.sum(((F.col("y_true")==1)&(F.col("y_pred")==1)).cast("int")).alias("TP"),
            F.sum(((F.col("y_true")==0)&(F.col("y_pred")==0)).cast("int")).alias("TN"),
            F.sum(((F.col("y_true")==0)&(F.col("y_pred")==1)).cast("int")).alias("FP"),
            F.sum(((F.col("y_true")==1)&(F.col("y_pred")==0)).cast("int")).alias("FN"),
        ))
    base.toPandas().to_csv(COUNTS_15, index=False)
    mark_done(COUNTS_15)
    print("15-min counts cached")

B15 = pd.read_csv(COUNTS_15, parse_dates=["w15"])
print(f"{len(B15)} 15-min buckets | "
      f"2018={int((B15.Year==2018).sum())} 2019={int((B15.Year==2019).sum())}")


# %% ---- cell 3b: look at 2019 before filtering anything ----
# CIC-DDoS2019 separates benign and attack traffic in TIME, so a
# benign-count filter deletes exactly the flood windows the study
# is about. Inspect before deciding.
n19 = B15[B15.Year == 2019].copy()
n19["atk"] = n19.TP + n19.FN
n19["ben"] = n19.TN + n19.FP
print(n19[["w15","atk","ben"]].to_string(index=False))
print("\n2019 totals: attack=%d benign=%d" % (n19.atk.sum(), n19.ben.sum()))


# %% ---- cell 3c: re-windowing (pure pandas, instant) -------
def targets(hours, min_benign=0, min_flows=500, df=B15):
    """
    Build the decay-target table at an arbitrary window size.
    min_benign defaults to 0: in 2019 the attack windows contain
    almost no benign traffic, and excluding them would remove the
    severe-drift period entirely. TSS is only meaningful where
    n_benign is non-trivial — filter at REPORTING time, not here.
    """
    g = df.copy()
    g["TimeWindow"] = g["w15"].dt.floor(f"{int(hours*60)}min")
    d = (g.groupby(["Year", "TimeWindow"])[["TP","TN","FP","FN"]]
           .sum().reset_index())

    d["n_flows"]  = d[["TP","TN","FP","FN"]].sum(axis=1)
    d["n_attack"] = d.TP + d.FN
    d["n_benign"] = d.TN + d.FP
    d = d[(d.n_attack > 0) &
          (d.n_benign >= min_benign) &
          (d.n_flows  >= min_flows)].reset_index(drop=True)

    tp, tn, fp, fn = d.TP, d.TN, d.FP, d.FN
    d["Window_F1"]  = (2*tp / (2*tp + fp + fn)).fillna(0.0)
    sens = (tp/(tp+fn)).fillna(0.0)
    spec = (tn/(tn+fp)).fillna(0.0)
    d["Window_TSS"] = sens + spec - 1.0
    # TSS needs a real benign population to mean anything
    d.loc[d.n_benign < 50, "Window_TSS"] = np.nan
    d["Delta_F1"]   = F1_BASE - d["Window_F1"]
    return d.sort_values(["Year","TimeWindow"]).reset_index(drop=True)


# %% ---- cell 3d: compare window sizes ----------------------
for h in [0.25, 0.5, 1, 2, 4]:
    t = targets(h)
    ceil = (t.Delta_F1 > F1_BASE - 1e-3).mean()
    by_yr = dict(t.Year.value_counts().sort_index())
    print(f"{h:>4}h  n={len(t):>4}  {by_yr}  "
          f"std={t.Delta_F1.std():.3f}  ceiling={100*ceil:>4.0f}%  "
          f"mid={(t.Delta_F1.between(0.05, 0.9)).sum()}")
# "mid" = windows with a genuine partial-decay target (not 0, not collapsed).
# Those are the rows that actually carry regression signal — pick the window
# size that maximises them while keeping enough total windows.


# %% ---- cell 3e: commit the chosen size --------------------
WINDOW_HOURS = 1           # <-- set from the comparison above
tgt = targets(WINDOW_HOURS)
tgt.to_csv(P["targets"], index=False)
mark_done(P["targets"], {"window_hours": WINDOW_HOURS, "n": len(tgt)})

d = tgt["Delta_F1"]
print(f"Delta_F1  n={len(d)}  std={d.std():.4f}  min={d.min():.3f}  "
      f"max={d.max():.3f}  %at_ceiling={100*(d > F1_BASE-1e-3).mean():.1f}%")
print(tgt.groupby("Year")["Delta_F1"].describe())
print(tgt.groupby("Year")[["n_flows","n_attack","n_benign"]].median())

# Censoring note for the paper: the static detector does not decay smoothly —
# it collapses to F1 ~ 0 and stays there. Delta_F1 is therefore heavily
# censored at the F1_BASE ceiling, which is a finding about static detectors,
# not a defect in the target. If "mid" stays small at every window size,
# reframe Tier 3 as (a) time-to-collapse or (b) binary next-window degradation
# rather than forcing a continuous regression onto a censored target.
