# ============================================================
# TIER 0 — shared config, checkpointing, cleaning   [v5]
# ------------------------------------------------------------
# COMPLETE FILE. Replace your whole tier0 cell with this.
# Paste as the cell directly BELOW the Spark-init cell.
#
# v5 fixes: ONE add_window (there were two; the stale one built
# an invalid "0.25 hours" interval and won because it came last).
# Windows are now expressed in MINUTES throughout.
# ============================================================

import os, json, shutil, pickle
from pyspark.sql import functions as F

# ---------------- paths -------------------------------------
BASE = "/kaggle/input/datasets/arnav1909/cic-ids-dataset/PROCESSED"
PARQUET_2017 = f"{BASE}/cicids2017_master_chronological.parquet"
PARQUET_2018 = f"{BASE}/cicids2018_master_chronological.parquet"
PARQUET_2019 = f"{BASE}/cicids2019_master_chronological.parquet"

# Tier 0b (training notebook): leave as /kaggle/working.
# Later notebooks: repoint at the committed model dataset, e.g.
#   MODEL_DIR = "/kaggle/input/<train-notebook-slug>"
MODEL_DIR    = "/kaggle/working"
MODEL_PATH   = f"{MODEL_DIR}/rf_2017"
SCALER_PATH  = f"{MODEL_DIR}/scaler_2017"
META_PATH    = f"{MODEL_DIR}/model_meta.json"

WORK = "/kaggle/working/decay"
os.makedirs(WORK, exist_ok=True)

P = {
    "preds":    f"{WORK}/01_flow_predictions",
    "targets":  f"{WORK}/02_decay_targets.csv",
    "samples":  f"{WORK}/03_window_samples",
    "ref":      f"{WORK}/04_drift_reference.pkl",
    "arf":      f"{WORK}/05_arf_metrics.csv",
    "arf_ckpt": f"{WORK}/05_arf_state.pkl",
    "drift":    f"{WORK}/06_drift_vectors.csv",
    "final":    f"{WORK}/07_forecast_results.json",
}

# ---------------- schema (must precede FEATURE_COLS) --------
TS_COL    = "Timestamp"
LABEL_COL = "Label"

BENIGN_IDX = 0.0          # `y` is 0=benign / 1=attack, so the predicted class
                          # index IS the label. No StringIndexer in the pipeline.

# Timestamps are INT64 nanoseconds-since-epoch.
# Verified: 2017 bad_ts=0, 2018 bad_ts=14, 2019 bad_ts=0.
TS_DIVISOR = 1_000_000_000
TS_FORMAT  = None

# ---------------- experiment knobs --------------------------
WINDOW_MINUTES = 15       # 15-min windows: n=145 windows, the most partial-decay
                          # targets available (mid=11). Derived WINDOW_HOURS is
                          # kept only for display / legacy references.
WINDOW_HOURS   = WINDOW_MINUTES / 60

MIN_FLOWS      = 500      # min total flows per window
MIN_BENIGN     = 0        # benign/attack are temporally separated in CIC-DDoS2019,
                          # so a benign filter would delete the flood windows.
                          # TSS is masked at report time where n_benign < 50.
PER_CLASS_CAP  = 20_000
REF_SAMPLE_N   = 200_000

SEED = 42

# Held-out 2017 test split, 200 trees @ depth 20.
# TP=66,357 TN=357,747 FP=185 FN=272 | TSS=0.995401
F1_BASE = 0.9965683219319521
# ^ single source of truth; no second F1_BASE assignment in this file.


# ---------------- column naming -----------------------------
_DROP = {
    TS_COL, LABEL_COL,
    "Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
    "Src Port", "Source Port", "Dst Port", "Destination Port",
    "Protocol", "SimillarHTTP", "Unnamed: 0", "Inbound",
    "Fwd Header Length.1",          # duplicate column in the raw CIC files
}

# CIC-DDoS2019 keeps the original CICFlowMeter column names; 2017/2018 use the
# cleaned variants. Renamed on load so all three years share one schema.
# Four of these are top-10 features by importance, so leaving them unmatched
# would have looked like 2019 "drift" rather than a naming artifact.
ALIASES = {
    "Total Length of Fwd Packets": "Fwd Packets Length Total",
    "Total Length of Bwd Packets": "Bwd Packets Length Total",
    "Min Packet Length":           "Packet Length Min",
    "Max Packet Length":           "Packet Length Max",
    "Average Packet Size":         "Avg Packet Size",
    "Init_Win_bytes_forward":      "Init Fwd Win Bytes",
    "Init_Win_bytes_backward":     "Init Bwd Win Bytes",
    "act_data_pkt_fwd":            "Fwd Act Data Packets",
    "min_seg_size_forward":        "Fwd Seg Size Min",
}

def apply_aliases(df):
    """Normalise 2019's legacy column names to the 2017/2018 schema."""
    for old, new in ALIASES.items():
        if old in df.columns and new not in df.columns:
            df = df.withColumnRenamed(old, new)
    return df

def read_year(path):
    """Always read parquet through this so aliases apply everywhere."""
    return apply_aliases(spark.read.parquet(path))        # noqa: F821


# ---------------- feature list ------------------------------
def derive_features(intersect_years=True):
    """Columns present in ALL THREE years (post-aliasing), in 2017 order."""
    cols17 = read_year(PARQUET_2017).columns
    c17 = set(cols17)
    if not intersect_years:
        keep = c17
    else:
        c18 = set(read_year(PARQUET_2018).columns)
        c19 = set(read_year(PARQUET_2019).columns)
        only17 = sorted((c17 - c19) - _DROP)
        only19 = sorted((c19 - c17) - _DROP)
        if only17: print("WARN in 2017 not 2019:", only17)
        if only19: print("WARN in 2019 not 2017:", only19)
        keep = c17 & c18 & c19
    # preserve 2017 column order: VectorAssembler is order-sensitive
    return [c for c in cols17 if c in keep and c not in _DROP]

FEATURE_COLS = derive_features()
print(len(FEATURE_COLS), "features from schema intersection")

# Prefer the EXACT list the model was fit on, when it exists.
if os.path.exists(META_PATH):
    _meta = json.load(open(META_PATH))
    _trained = _meta["feature_cols"]
    _absent = [c for c in _trained if c not in FEATURE_COLS]
    if _absent:
        print("ERROR: model needs columns absent from the intersection:", _absent)
    else:
        FEATURE_COLS = _trained
        print(f"using trained feature list from model_meta.json: {len(FEATURE_COLS)}")


# ---------------- label harmonisation -----------------------
FAMILY = {
    "DoS Hulk": "DoS",                "DoS attacks-Hulk": "DoS",
    "DoS GoldenEye": "DoS",           "DoS attacks-GoldenEye": "DoS",
    "DoS slowloris": "DoS",           "DoS attacks-Slowloris": "DoS",
    "DoS Slowhttptest": "DoS",        "DoS attacks-SlowHTTPTest": "DoS",
    "DDoS": "DDoS",                   "DDoS attacks-LOIC-HTTP": "DDoS",
    "DDOS attack-HOIC": "DDoS",       "DDOS attack-LOIC-UDP": "DDoS",
    "MSSQL": "DDoS", "Syn": "DDoS", "UDP": "DDoS", "UDPLag": "DDoS",
    "NetBIOS": "DDoS", "LDAP": "DDoS", "Portmap": "DDoS",
    "FTP-Patator": "BruteForce",      "FTP-BruteForce": "BruteForce",
    "SSH-Patator": "BruteForce",      "SSH-Bruteforce": "BruteForce",
    # 2017 web-attack labels use an EN-DASH (\u2013), not a hyphen
    "Web Attack \u2013 Brute Force": "Web",   "Brute Force -Web": "Web",
    "Web Attack \u2013 XSS": "Web",           "Brute Force -XSS": "Web",
    "Web Attack \u2013 Sql Injection": "Web", "SQL Injection": "Web",
    "Bot": "Bot",
    "Infiltration": "Infiltration",   "Infilteration": "Infiltration",  # CIC typo
    "PortScan": "Recon",
    "Heartbleed": "Other",
}

def is_attack():
    """Binary attack flag. Case-insensitive: 2019 uses BENIGN, others Benign."""
    return (F.upper(F.trim(F.col(LABEL_COL))) != F.lit("BENIGN")).cast("int")

def family_col():
    expr = F.when(F.upper(F.trim(F.col(LABEL_COL))) == F.lit("BENIGN"), F.lit("Benign"))
    for raw, fam in FAMILY.items():
        expr = expr.when(F.trim(F.col(LABEL_COL)) == F.lit(raw), F.lit(fam))
    return expr.otherwise(F.lit("UNMAPPED"))

def audit_labels(paths=None):
    """Run once. Any UNMAPPED row means a label is missing from FAMILY."""
    paths = paths or [("2017", PARQUET_2017), ("2018", PARQUET_2018),
                      ("2019", PARQUET_2019)]
    for yr, p in paths:
        d = read_year(p).withColumn("fam", family_col())
        bad = d.where(F.col("fam") == "UNMAPPED")
        n = bad.count()
        print(f"[{yr}] unmapped={n:,}")
        if n:
            bad.groupBy(LABEL_COL).count().show(20, False)


# ---------------- checkpoint helpers ------------------------
def _marker(path): return path.rstrip("/") + ".done"

def is_done(path): return os.path.exists(_marker(path))

def mark_done(path, meta=None):
    with open(_marker(path), "w") as f:
        json.dump(meta or {"ok": True}, f)

def reset(path):
    for p in (path, _marker(path)):
        if os.path.isdir(p):    shutil.rmtree(p)
        elif os.path.exists(p): os.remove(p)
    print("reset:", path)

def stage(path, force=False):
    """`if stage(P['preds']):` -> True means DO the work."""
    if force:
        reset(path); return True
    if is_done(path):
        print(f"SKIP (checkpoint exists): {path}"); return False
    return True


# ---------------- data hygiene ------------------------------
# CIC-DDoS2019 evaluation day is 2019-03-11 (CSV-03-11), but the source parquet
# carries 2018-11-03 (day/month transposed, year off). Intra-day structure is
# intact, so the date is shifted and the clock kept. Applied in load_year so
# every consumer — targets, samples, drift — sees the corrected timeline.
SHIFT_2019_DAYS = 128        # 2018-11-03 -> 2019-03-11

def parse_ts(df):
    """Handles nanos-as-bigint, native timestamp, and string timestamps."""
    dt = dict(df.dtypes)[TS_COL]
    if dt in ("bigint", "long", "int"):
        return df.withColumn("_ts",
            (F.col(TS_COL) / F.lit(TS_DIVISOR)).cast("timestamp"))
    if dt.startswith("timestamp"):
        return df.withColumn("_ts", F.col(TS_COL).cast("timestamp"))
    ts = F.to_timestamp(F.col(TS_COL), TS_FORMAT) if TS_FORMAT \
         else F.to_timestamp(F.col(TS_COL))
    return df.withColumn("_ts", ts)

def drop_bad_ts(df):
    return df.where(F.col("_ts").isNotNull() & (F.year("_ts") >= 2017))

def clean_features(df, cols=None):
    """
    CIC flow-rate columns carry Inf/NaN from divide-by-zero on near-zero-duration
    flows (rampant in CIC-DDoS2019). Replaced with 0.0; document as preprocessing.
    """
    for c in (cols or FEATURE_COLS):
        df = df.withColumn(c,
            F.when(F.col(c).isNull() | F.isnan(F.col(c)) |
                   (F.abs(F.col(c)) == float("inf")), F.lit(0.0))
             .otherwise(F.col(c).cast("double")))
    return df

def add_window(df, minutes=None):
    """THE ONLY add_window. Minutes, so sub-hour windows are expressible."""
    m = int(minutes or WINDOW_MINUTES)
    return df.withColumn("win", F.window(F.col("_ts"), f"{m} minutes").start)

def load_year(path, year):
    """aliases -> timestamps -> re-date 2019 -> clean -> window + labels."""
    df = parse_ts(read_year(path))
    if year == 2019:
        df = df.withColumn("_ts",
             F.col("_ts") + F.expr(f"INTERVAL {SHIFT_2019_DAYS} DAYS"))
    df = drop_bad_ts(df)
    df = clean_features(df)
    return (add_window(df)
            .withColumn("Year", F.lit(year))
            .withColumn("y", is_attack())
            .withColumn("family", family_col()))


def save_pickle(obj, path):
    with open(path, "wb") as f: pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f: return pickle.load(f)


print(f"tier0 v5 | WORK={WORK} | window={WINDOW_MINUTES}min | "
      f"F1_BASE={F1_BASE:.6f} | features={len(FEATURE_COLS)}")
