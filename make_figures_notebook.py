# ============================================================
# make_figures_notebook.py — Figs. 2-9 for the IEEE paper
# ------------------------------------------------------------
# NOTEBOOK VERSION. Paste the whole file into ONE Kaggle cell
# and run it. No argparse, so Jupyter's own -f flag cannot
# break it, and the CSV paths are discovered automatically
# across every attached input.
#
# Every number below is transcribed from the measured results
# in Section IV. Where a figure needs a per-window series and
# the CSV is not found, that figure is SKIPPED rather than
# drawn from invented data.
# ============================================================

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/kaggle/working/figures"
os.makedirs(OUT, exist_ok=True)

def _find(pattern):
    """Search working dir first, then every attached input, at any depth."""
    for root in ("/kaggle/working", "/kaggle/input"):
        hits = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
        if hits:
            return hits[0]
    return None

TARGETS = _find("02_decay_targets.csv")
DRIFT   = _find("06_drift_vectors.csv")
print("targets:", TARGETS or "NOT FOUND — fig2 will be skipped")
print("drift  :", DRIFT   or "NOT FOUND — fig7 will be skipped")
print("output :", OUT)

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
COL1, COL2 = 3.5, 7.16                       # IEEE column widths, inches
GREY, ACC, WARN = "#4D4D4D", "#1F4E79", "#A63603"

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")

# ---------------------------------------------------------------- measured constants
F1_BASE = 0.9966

FAM_2018 = [                                  # Table 4
    ("Web",           928,    75, 0.080819),
    ("DoS",        506075, 20501, 0.040510),
    ("BruteForce", 156668,     7, 0.000045),
    ("Infiltration",160604,    4, 0.000025),
    ("Bot",        282310,     0, 0.0),
    ("DDoS",      1246382,     0, 0.0),
]
FAM_2019 = ("DDoS", 19500705, 70296, 0.003605)

# Second field is deliberately short: fig5 prints it on a rotated,
# single-line tick label and the full tool names collide at COL2 width.
DAYS = [                                      # Table 5
    ("02-14", "BruteForce",     156668,     7),
    ("02-15", "DoS GE/Slow",     51740, 17672),
    ("02-16", "DoS SHT/Hulk",   454335,  2829),
    ("02-20", "DDoS LOIC-HTTP", 576191,     0),
    ("02-21", "DDoS LOIC/HOIC", 670191,     0),
    ("02-22", "Web BF/XSS",        362,    29),
    ("02-23", "Web/SQLi",          566,    46),
    ("02-28", "Infiltration",    68224,     2),
    ("03-01", "Infiltration",    92380,     2),
    ("03-02", "Bot",            282310,     0),
]

PSI = {"2018": dict(mean=1.256, lo=0.519, hi=6.671),
       "2019": dict(mean=0.912, lo=0.428, hi=2.316)}
RHO_TIME = 0.028

ENTROPY = [                                   # Table 6
    ("Web",          0.4049, 0.1425, 0.2963),
    ("DoS",          0.3049, 0.0958, 0.1485),
    ("BruteForce",   0.4919, 0.1089, 0.3837),
    ("DDoS",         0.3549, 0.0851, 0.4332),
    ("Bot",          0.1184, 0.0168, 0.0026),
    ("Infiltration", 0.1059, 0.0197, 0.0251),
]

BUDGETS = [10, 25, 50, 100, 250, 500]         # Table 7
LABEL_BUDGET = {
    "Bot":          [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "DDoS":         [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Infiltration": [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "BruteForce":   [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "DoS":          [0.940, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Web":          [0.814, 1.0, 1.0, 1.0, 1.0, 1.0],
}

BATTERY = [                                   # Table 10: name, rho_F1, rho_next, rho_atk, confounded
    ("PSI mean",            0.670,  0.486,  0.712, True),
    ("PSI max",            -0.110, -0.138, -0.049, False),
    ("Wasserstein mean",    0.227,  0.138,  0.213, False),
    ("JS mean",             0.687,  0.495,  0.743, True),
    ("Hellinger mean",      0.694,  0.504,  0.723, True),
    ("Imp.-weighted PSI",   0.691,  0.515,  0.659, False),
    ("EWMA velocity",       0.349,  0.212,  0.530, True),
    ("EWMA acceleration",  -0.042, -0.081, -0.087, True),
    ("Mean entropy",        0.278,  0.125,  0.330, True),
    ("Uncertain fraction", -0.425, -0.465, -0.326, False),
]

PARTIAL = [                                   # Table 12
    ("none",                      -0.465),
    ("attack fraction",           -0.373),
    ("current F1",                -0.290),
    ("attack frac. + current F1", -0.296),
]
BOOT_CI = (-0.445, -0.118)

# ---------------------------------------------------------------- figures
def fig2_window_f1():
    if not TARGETS:
        print("  SKIP fig2 — 02_decay_targets.csv not found"); return
    d = pd.read_csv(TARGETS).sort_values(["Year", "TimeWindow"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL2, 2.2))
    x = np.arange(len(d))
    ax.plot(x, d.Window_F1, lw=0.9, color=ACC, marker="o", ms=2.2)
    ax.axhline(F1_BASE, ls="--", lw=0.9, color=WARN)
    ax.text(len(d) * 0.985, F1_BASE - 0.055, "2017 held-out baseline (0.9966)",
            ha="right", fontsize=6.5, color=WARN)
    split = int((d.Year == 2018).sum())
    ax.axvline(split - 0.5, ls=":", lw=0.9, color=GREY)
    ax.text(2, 0.86, "CSE-CIC-IDS2018", fontsize=6.5, color=GREY)
    ax.text(split + 1, 0.86, "CIC-DDoS2019", fontsize=6.5, color=GREY)
    ax.set_xlabel("Window index (15-minute tumbling, chronological)")
    ax.set_ylabel("Window $F_1$")
    ax.set_ylim(-0.03, 1.05); ax.set_xlim(-1, len(d))
    save(fig, "fig2_window_f1")

def fig3_survival():
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    k = np.arange(0, 11)
    s = np.where(k < 1, 1.0, 0.0)
    ax.step(k, s, where="post", lw=1.4, color=ACC)
    ax.fill_between(k, 0, s, step="post", alpha=0.12, color=ACC)
    ax.annotate("median survival = 1 window (15 min)\nall 3 architectures",
                xy=(1, 0.0), xytext=(2.4, 0.46), fontsize=6.5,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))
    ax.set_xlabel("Windows since deployment")
    ax.set_ylabel(r"$\hat{S}(k)$,  $F_1 \geq 0.5$")
    ax.set_ylim(-0.04, 1.08); ax.set_xlim(0, 10)
    save(fig, "fig3_survival")

def fig4_family_recall():
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    names = [f[0] for f in FAM_2018]
    vals  = [max(f[3], 1e-6) for f in FAM_2018]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=ACC, alpha=0.85, height=0.62)
    bars[0].set_color(WARN)
    ax.barh([len(names)], [max(FAM_2019[3], 1e-6)], color=GREY, alpha=0.85, height=0.62)
    ax.set_yticks(list(y) + [len(names)])
    ax.set_yticklabels(names + ["DDoS (2019)"])
    ax.set_xscale("log"); ax.set_xlim(1e-6, 1)
    ax.axvline(0.5, ls="--", lw=0.9, color=WARN)
    ax.text(0.52, 0.2, "operational\nfloor", fontsize=6, color=WARN)
    for i, (n, tot, det, r) in enumerate(FAM_2018):
        ax.text(max(r, 1e-6) * 1.4, i, f"{det:,}/{tot:,}", va="center", fontsize=5.6)
    ax.text(FAM_2019[3] * 1.4, len(names), f"{FAM_2019[2]:,}/{FAM_2019[1]:,}",
            va="center", fontsize=5.6)
    ax.set_xlabel("Recall (log scale)"); ax.invert_yaxis()
    save(fig, "fig4_family_recall")

def fig5_day_recall():
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    # Single-line rotated labels: a two-line label at this width overlapped
    # its neighbours, so the date and the short tool name share one line.
    labels = [f"{d}  {a}" for d, a, _, _ in DAYS]
    vals   = [max(det / tot, 1e-6) for _, _, tot, det in DAYS]
    x = np.arange(len(DAYS))
    ax.bar(x, vals, color=[WARN if v > 0.05 else ACC for v in vals],
           alpha=0.85, width=0.62)
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=6.2)
    ax.set_ylabel("Recall (log scale)")
    # Two arrows rather than one: the 55x gap is between these two bars,
    # and a single annotation left it ambiguous which pair was meant.
    ax.annotate("0.3416", xy=(1, vals[1]), xytext=(1.45, 0.62),
                fontsize=6, color=GREY,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))
    ax.annotate("0.0062 — both DoS days,\nboth families in training",
                xy=(2, vals[2]), xytext=(2.75, 0.22),
                fontsize=6, color=GREY,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))
    save(fig, "fig5_day_recall")

def fig6_psi_saturation():
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    if DRIFT:
        d = pd.read_csv(DRIFT)
        ax.hist(d.psi_mean.dropna(), bins=28, color=ACC, alpha=0.8,
                edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Windows")
    else:
        for i, (yr, s) in enumerate(PSI.items(), start=1):
            ax.errorbar([s["mean"]], [i],
                        xerr=[[s["mean"] - s["lo"]], [s["hi"] - s["mean"]]],
                        fmt="o", color=ACC, capsize=3, lw=1.2, ms=4)
        ax.set_yticks([1, 2]); ax.set_yticklabels(["2018", "2019"])
        ax.set_ylim(0.4, 2.6)
        print("  fig6 using summary range (per-window CSV not found)")
    ax.axvline(0.25, ls="--", lw=1.1, color=WARN)
    ax.text(0.30, ax.get_ylim()[1] * 0.86,
            "conventional alarm\nthreshold (0.25)", fontsize=6, color=WARN)
    ax.set_xlabel("Mean PSI per window")
    ax.set_title("100% of 145 windows exceed the threshold", fontsize=7.5)
    save(fig, "fig6_psi_saturation")

def fig7_psi_vs_time():
    if not DRIFT:
        print("  SKIP fig7 — 06_drift_vectors.csv not found"); return
    d = pd.read_csv(DRIFT)
    d["win_key"] = d.win_key.astype(str)
    d = d[~d.win_key.str.startswith("2019")].sort_values("win_key").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    ax.scatter(np.arange(len(d)), d.psi_mean, s=7, color=ACC,
               alpha=0.75, edgecolors="none")
    ax.axhline(0.25, ls="--", lw=1.0, color=WARN)
    ax.text(len(d) * 0.02, 0.33, "alarm threshold", fontsize=6, color=WARN)
    ax.set_xlabel("Window index, CSE-CIC-IDS2018")
    ax.set_ylabel("Mean PSI")
    ax.set_title(fr"Spearman $\rho$ = {RHO_TIME:.3f} with window order", fontsize=7.5)
    save(fig, "fig7_psi_vs_time")

def fig8_output_space():
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    names = [e[0] for e in ENTROPY]
    ent, patk, unc = ([e[i] for e in ENTROPY] for i in (1, 2, 3))
    x = np.arange(len(names)); w = 0.26
    ax.bar(x - w, ent,  w, label="Mean predictive entropy", color=ACC, alpha=0.85)
    ax.bar(x,     unc,  w, label="Uncertain-flow fraction", color=GREY, alpha=0.85)
    ax.bar(x + w, patk, w, label="Mean P(attack) on true attacks", color=WARN, alpha=0.85)
    ax.axhline(0.5, ls="--", lw=1.0, color="k")
    ax.text(len(names) - 0.55, 0.53, "decision threshold", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("Statistic value"); ax.set_ylim(0, 0.62)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("The detector is confident and wrong: P(attack) never approaches 0.5",
                 fontsize=7.5)
    save(fig, "fig8_output_space")

def fig9_label_budget():
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    for name, series in LABEL_BUDGET.items():
        ax.plot(BUDGETS, series, "-o" if name in ("DoS", "Web") else "-",
                lw=1.1, ms=3, label=name, alpha=0.9)
    ax.axvline(25, ls="--", lw=1.0, color=WARN)
    ax.text(27, 0.845, "n = 25:\nall families certain", fontsize=6, color=WARN)
    ax.set_xscale("log"); ax.set_xticks(BUDGETS)
    ax.set_xticklabels(BUDGETS); ax.set_ylim(0.78, 1.02)
    ax.set_xlabel("Labelled attack flows per family")
    ax.set_ylabel("P(correctly establishing failure)")
    ax.legend(fontsize=5.8, ncol=2, loc="lower right", framealpha=0.9)
    save(fig, "fig9_label_budget")

def fig10_battery():
    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    names = [b[0] for b in BATTERY]
    y = np.arange(len(names)); w = 0.38
    ax.barh(y + w/2, [abs(b[1]) for b in BATTERY], w,
            label=r"|$\rho$| with current $F_1$", color=ACC, alpha=0.85)
    ax.barh(y - w/2, [abs(b[3]) for b in BATTERY], w,
            label=r"|$\rho$| with attack fraction", color=WARN, alpha=0.7)
    # Marker goes to the RIGHT of the longer bar in each row. Placing it at
    # x < 0 put it on top of the y-tick label and both became unreadable.
    for i, b in enumerate(BATTERY):
        if not b[4]:
            ax.text(max(abs(b[1]), abs(b[3])) + 0.015, i, "admitted",
                    ha="left", va="center", fontsize=5.8,
                    color=ACC, style="italic")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlabel(r"|Spearman $\rho$|"); ax.set_xlim(0, 0.88)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9); ax.invert_yaxis()
    ax.set_title("A statistic is admitted only when it tracks performance more than workload",
                 fontsize=7.5)
    save(fig, "fig10_drift_battery")

def fig11_partial():
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    labels = [p[0] for p in PARTIAL]; vals = [p[1] for p in PARTIAL]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=ACC, alpha=0.85, height=0.55)
    ax.axvspan(BOOT_CI[0], BOOT_CI[1], color=WARN, alpha=0.15)
    ax.axvline(0, color="k", lw=0.8)
    ax.text(BOOT_CI[1] + 0.01, len(labels) - 0.4,
            "bootstrap 95% CI\n[-0.445, -0.118]", fontsize=5.8, color=WARN)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_xlabel(r"Partial Spearman $\rho$ with next-window $F_1$")
    ax.set_xlim(-0.55, 0.12); ax.invert_yaxis()
    save(fig, "fig11_partial_correlation")

# ---------------------------------------------------------------- run
for f in (fig2_window_f1, fig3_survival, fig4_family_recall, fig5_day_recall,
          fig6_psi_saturation, fig7_psi_vs_time, fig8_output_space,
          fig9_label_budget, fig10_battery, fig11_partial):
    try:
        f()
    except Exception as e:
        print(f"  FAILED {f.__name__}: {type(e).__name__}: {e}")

print("\nfiles in", OUT)
for p in sorted(glob.glob(f"{OUT}/*.png")):
    print("  ", os.path.basename(p))
print("\nFigs 2-9 are the numbered set for Section 4.3; 10-11 are spares for 4.5.")
