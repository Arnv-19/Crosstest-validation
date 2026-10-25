# ============================================================
# TIER 4b — FAMILY-MATCHED drift reference
# ------------------------------------------------------------
# Fixes a confound in Tier 4: that version compared attack-only
# 2018 flows against a 200k random 2017 reference that is ~80%
# BENIGN. So part of the measured PSI was just "attack traffic
# differs from benign traffic", not "2018 DoS differs from
# 2017 DoS".
#
# Here each 2018/2019 family is compared against the SAME family
# in 2017. That is the comparison the claim actually needs.
#
# Run in the TIER-1 notebook, same session as Tier 4
# (Spark up, tier0 loaded, FAM_SAMPLES already written).
# ============================================================

# %% ---- cell 1: per-family 2017 reference ------------------
from pyspark.sql import functions as F, Window as W
import pandas as pd, numpy as np, glob, os, pickle
import pyarrow.parquet as pq
from scipy.stats import wasserstein_distance

FAM_REF_DIR = f"{WORK}/13_family_reference"
FAM_REF_CAP = 20_000
MIN_REF_N   = 500        # below this a reference distribution is not trustworthy
N_BINS, EPS = 10, 1e-6

if stage(FAM_REF_DIR):
    r17 = load_year(PARQUET_2017, 2017).select("family", "y", *FEATURE_COLS)
    w = W.partitionBy("family").orderBy(F.rand(SEED))
    (r17.withColumn("_rn", F.row_number().over(w))
        .where(F.col("_rn") <= FAM_REF_CAP).drop("_rn")
        .write.mode("overwrite").partitionBy("family").parquet(FAM_REF_DIR))
    mark_done(FAM_REF_DIR)
    print("2017 per-family reference written")

# how much 2017 data backs each family? (several are tiny)
print("\n2017 reference sizes per family:")
ref_sizes = {}
for d in sorted(glob.glob(f"{FAM_REF_DIR}/family=*")):
    fam = d.split("family=")[-1]
    n = sum(pq.read_metadata(f).num_rows for f in glob.glob(f"{d}/*.parquet"))
    ref_sizes[fam] = n
    print(f"  {fam:14s} {n:>8,}")
# NOTE: 2017 Infiltration is only ~36 flows and Web ~2,180. Any family-matched
# drift estimate for those is unreliable and must be flagged, not quietly used.


# %% ---- cell 2: build per-family reference distributions ---
def build_reference(df, cols):
    """Quantile bin edges + standardisation stats for one family."""
    rng = np.random.RandomState(SEED)
    ref = {}
    for c in cols:
        v = df[c].values.astype("float64")
        v = v[np.isfinite(v)]
        if len(v) < 50 or v.std() == 0:
            ref[c] = None; continue
        edges = np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)))
        if len(edges) < 3:
            ref[c] = None; continue
        edges[0], edges[-1] = -np.inf, np.inf
        cnt, _ = np.histogram(v, bins=edges)
        ref[c] = {"edges": edges, "p": cnt / max(cnt.sum(), 1),
                  "mu": float(v.mean()), "sd": float(v.std()),
                  "sample": v[rng.choice(len(v), size=min(20_000, len(v)),
                                         replace=False)]}
    return ref

def psi(expected_p, actual_v, edges):
    cnt, _ = np.histogram(actual_v, bins=edges)
    q = np.clip(cnt / max(cnt.sum(), 1), EPS, None)
    p = np.clip(expected_p, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))

FAM_REFS = {}
for d in sorted(glob.glob(f"{FAM_REF_DIR}/family=*")):
    fam = d.split("family=")[-1]
    if fam == "Benign":
        continue
    s = pq.read_table(d).to_pandas()
    s = s[s.y == 1] if "y" in s.columns else s
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(s) < 50:
        print(f"  skip {fam}: only {len(s)} 2017 flows"); continue
    FAM_REFS[fam] = {"ref": build_reference(s, FEATURE_COLS), "n": len(s)}
print("\nbuilt references for:", sorted(FAM_REFS))


# %% ---- cell 3: family-matched drift -----------------------
rows = []
for d in sorted(glob.glob(f"{FAM_SAMPLES}/family=*")):
    fam = d.split("family=")[-1]
    if fam == "Benign" or fam not in FAM_REFS:
        continue
    s = pq.read_table(d).to_pandas()
    s = s[(s.y == 1) & (s.Year == 2018)]
    if len(s) < 200:
        continue
    R = FAM_REFS[fam]["ref"]
    active = [c for c in FEATURE_COLS if R.get(c) is not None]
    psis, wds = [], []
    for c in active:
        r = R[c]
        v = s[c].values.astype("float64"); v = v[np.isfinite(v)]
        if len(v) < 50:
            continue
        psis.append(psi(r["p"], v, r["edges"]))
        wds.append(wasserstein_distance((v - r["mu"]) / r["sd"],
                                        (r["sample"] - r["mu"]) / r["sd"]))
    if not psis:
        continue
    rows.append({"family": fam, "n_2018": len(s),
                 "n_2017_ref": FAM_REFS[fam]["n"],
                 "ref_reliable": FAM_REFS[fam]["n"] >= MIN_REF_N,
                 "n_features": len(psis),
                 "psi_mean": float(np.mean(psis)),
                 "psi_max": float(np.max(psis)),
                 "psi_n_gt_0p25": int(np.sum(np.array(psis) > 0.25)),
                 "wd_mean": float(np.mean(wds)),
                 "wd_max": float(np.max(wds))})

fm = pd.DataFrame(rows)

# join the per-family recall from Tier 4
rec = pd.read_csv(f"{WORK}/12_family_recall.csv")
rec = rec[rec.Year == 2018][["family", "recall", "n_attack"]]
fm = fm.merge(rec, on="family", how="inner").sort_values("recall", ascending=False)

pd.set_option("display.width", 220)
print("\n=== FAMILY-MATCHED DRIFT (2018 family vs SAME 2017 family) ===")
print(fm.to_string(index=False))

rel = fm[fm.ref_reliable]
print(f"\nreliable rows (2017 reference >= {MIN_REF_N} flows): {len(rel)} of {len(fm)}")
if len(rel) >= 4:
    print("Spearman(drift, recall), reliable families only:")
    for c in ["psi_mean","psi_max","psi_n_gt_0p25","wd_mean","wd_max"]:
        print(f"  {c}: {rel[c].corr(rel['recall'], method='spearman'):+.3f}")
else:
    print("too few reliable families for a correlation — report the table itself,")
    print("not a coefficient. With n<4 a Spearman value is noise.")

fm.to_csv(f"{WORK}/14_family_matched_drift.csv", index=False)
print("\nsaved 14_family_matched_drift.csv")


# %% ---- cell 4: the comparison that goes in the paper ------
# Side-by-side: does matching the reference by family change the picture
# versus the global (mostly-benign) reference used in Tier 4?
try:
    old = pd.read_csv(f"{WORK}/11_family_drift_vs_recall.csv")[
            ["family","psi_mean","wd_mean"]].rename(
            columns={"psi_mean":"psi_global_ref","wd_mean":"wd_global_ref"})
    cmp = fm[["family","recall","psi_mean","wd_mean","ref_reliable"]].merge(old, on="family")
    cmp = cmp.rename(columns={"psi_mean":"psi_family_ref","wd_mean":"wd_family_ref"})
    print("\n=== GLOBAL vs FAMILY-MATCHED REFERENCE ===")
    print(cmp.sort_values("recall", ascending=False).to_string(index=False))
    print("\nIf PSI stays high for families whose recall is ~0 even when compared")
    print("against their OWN 2017 counterpart, the failure is behavioural: the")
    print("same named attack is implemented differently across testbeds.")
    print("If PSI drops to ~0 for a collapsed family, that family is a PURE")
    print("concept-drift case — P(X) preserved, P(Y|X) changed — which no")
    print("P(X)-based drift metric can detect by construction.")
except FileNotFoundError:
    print("11_family_drift_vs_recall.csv not found — skipping comparison")
