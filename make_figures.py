#!/usr/bin/env python3
"""
make_figures.py — generates Figs. 2-9 for the IEEE paper.

Every value below is transcribed from the measured results tabulated in
Section IV. Nothing here is synthetic: where a per-window series is needed
and only summary statistics were recorded, the script loads the real CSV
if present and otherwise SKIPS that figure rather than inventing a curve.

Usage
    python make_figures.py --data /path/to/decay --out ./figures

    --data   directory holding the pipeline CSVs (optional; figures that
             need per-window series are skipped if absent)
    --out    output directory (default ./figures)

Outputs 300 dpi PDF + PNG, sized for an IEEE double-column page
(3.5 in single column, 7.16 in double column).
"""

import argparse, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})
COL1, COL2 = 3.5, 7.16          # IEEE column widths, inches
GREY, ACC, WARN = "#4D4D4D", "#1F4E79", "#A63603"


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


def find(data_dir, pattern):
    if not data_dir:
        return None
    hits = glob.glob(os.path.join(data_dir, "**", pattern), recursive=True)
    return hits[0] if hits else None


# ---------------------------------------------------------------- measured constants
F1_BASE = 0.9966

# Table 4 — per-family recall, static detector
FAM_2018 = [
    ("Web",          928,     75, 0.080819),
    ("DoS",       506075,  20501, 0.040510),
    ("BruteForce",156668,      7, 0.000045),
    ("Infiltration",160604,    4, 0.000025),
    ("Bot",       282310,      0, 0.0),
    ("DDoS",     1246382,      0, 0.0),
]
FAM_2019 = ("DDoS", 19500705, 70296, 0.003605)

# Table 5 — per-day recall mapped to the published 2018 schedule
DAYS = [
    ("02-14", "Brute force",      156668,     7),
    ("02-15", "DoS GoldenEye/Slowloris", 51740, 17672),
    ("02-16", "DoS SlowHTTP/Hulk", 454335,  2829),
    ("02-20", "DDoS LOIC-HTTP",    576191,     0),
    ("02-21", "DDoS LOIC-UDP/HOIC",670191,     0),
    ("02-22", "Web BF/XSS",           362,    29),
    ("02-23", "Web/SQLi",             566,    46),
    ("02-28", "Infiltration",       68224,     2),
    ("03-01", "Infiltration",       92380,     2),
    ("03-02", "Bot",               282310,     0),
]

# Section 4.3 — PSI summary statistics
PSI = {"2018": dict(mean=1.256, sd=1.455, lo=0.519, hi=6.671, n=112),
       "2019": dict(mean=0.912, sd=0.639, lo=0.428, hi=2.316, n=33)}
PSI_FEATS = dict(minimum=35, q25=42, median=44, q75=46, maximum=56, total=56)
RHO_TIME = 0.028

# Table 6 — output-space behaviour on 2018 attack flows
ENTROPY = [
    ("Web",          0.4049, 0.1425, 0.2963, 0.080819),
    ("DoS",          0.3049, 0.0958, 0.1485, 0.040510),
    ("BruteForce",   0.4919, 0.1089, 0.3837, 0.000045),
    ("DDoS",         0.3549, 0.0851, 0.4332, 0.0),
    ("Bot",          0.1184, 0.0168, 0.0026, 0.0),
    ("Infiltration", 0.1059, 0.0197, 0.0251, 0.000025),
]

# Table 7 — label-budget triage
BUDGETS = [10, 25, 50, 100, 250, 500]
LABEL_BUDGET = {
    "Bot":          [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "DDoS":         [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Infiltration": [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "BruteForce":   [1.000, 1.0, 1.0, 1.0, 1.0, 1.0],
    "DoS":          [0.940, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Web":          [0.814, 1.0, 1.0, 1.0, 1.0, 1.0],
}

# Table 8 — cross-architecture comparison
ARCH = {
    "DDoS":         (0.0000, 0.1306, 0.1086),
    "DoS":          (0.0405, 0.0633, 0.0295),
    "Bot":          (0.0000, 0.0000, 0.0006),
    "Infiltration": (0.000025, 0.1844, 0.1944),
    "BruteForce":   (0.000045, 0.2520, 0.1375),
    "Web":          (0.0808, 0.0011, 0.0000),
}
ARCH_F1  = {"RF": 0.9966, "MLP": 0.9341, "LogReg": 0.8226}
ARCH_FPR = {"RF": 0.00042, "MLP": 0.01757, "LogReg": 0.03915}

# Table 10 — drift battery (metric, rho vs F1, rho vs next F1, rho vs attack frac)
BATTERY = [
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

# Table 12 — partial correlation of the admitted monitor
PARTIAL = [
    ("none",                     -0.465),
    ("attack fraction",          -0.373),
    ("current F1",               -0.290),
    ("attack frac. + current F1",-0.296),
]
BOOT_CI = (-0.445, -0.118)

# Survival (Section 4.3): all three architectures fail at window 1 under F1 < 0.5
SURVIVAL_FAIL_WINDOW = {"RF": 1, "MLP": 1, "LogReg": 1}


# ---------------------------------------------------------------- figures
def fig2_window_f1(out, data_dir):
    """Per-window F1 in chronological order. Needs the real per-window series."""
    path = find(data_dir, "02_decay_targets.csv")
    if not path:
        print("  SKIP fig2 — 02_decay_targets.csv not found (needs per-window series)")
        return
    import pandas as pd
    d = pd.read_csv(path).sort_values(["Year", "TimeWindow"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL2, 2.2))
    x = np.arange(len(d))
    ax.plot(x, d.Window_F1, lw=0.9, color=ACC, marker="o", ms=2.2)
    ax.axhline(F1_BASE, ls="--", lw=0.9, color=WARN)
    ax.text(len(d) * 0.985, F1_BASE - 0.055, "2017 held-out baseline (0.9966)",
            ha="right", fontsize=6.5, color=WARN)
    split = int((d.Year == 2018).sum())
    ax.axvline(split - 0.5, ls=":", lw=0.9, color=GREY)
    ax.text(split + 1, 0.86, "CIC-DDoS2019", fontsize=6.5, color=GREY)
    ax.text(2, 0.86, "CSE-CIC-IDS2018", fontsize=6.5, color=GREY)
    ax.set_xlabel("Window index (15-minute tumbling, chronological)")
    ax.set_ylabel("Window $F_1$")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-1, len(d))
    save(fig, out, "fig2_window_f1")


def fig3_survival(out):
    """Kaplan-Meier survival. All three architectures fail at the first window."""
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
    ax.set_ylim(-0.04, 1.08)
    ax.set_xlim(0, 10)
    save(fig, out, "fig3_survival")


def fig4_family_recall(out):
    fig, ax = plt.subplots(figsize=(COL1, 2.3))
    names = [f[0] for f in FAM_2018]
    vals = [max(f[3], 1e-6) for f in FAM_2018]
    y = np.arange(len(names))
    bars = ax.barh(y, vals, color=ACC, alpha=0.85, height=0.62)
    bars[0].set_color(WARN)
    ax.barh([len(names)], [max(FAM_2019[3], 1e-6)], color=GREY, alpha=0.85, height=0.62)
    ax.set_yticks(list(y) + [len(names)])
    ax.set_yticklabels(names + ["DDoS (2019)"])
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1)
    ax.axvline(0.5, ls="--", lw=0.9, color=WARN)
    ax.text(0.52, 0.2, "operational\nfloor", fontsize=6, color=WARN)
    for i, (n, tot, det, r) in enumerate(FAM_2018):
        ax.text(max(r, 1e-6) * 1.4, i, f"{det:,}/{tot:,}", va="center", fontsize=5.6)
    ax.text(FAM_2019[3] * 1.4, len(names), f"{FAM_2019[2]:,}/{FAM_2019[1]:,}",
            va="center", fontsize=5.6)
    ax.set_xlabel("Recall (log scale)")
    ax.invert_yaxis()
    save(fig, out, "fig4_family_recall")


def fig5_day_recall(out):
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    labels = [f"{d}\n{a}" for d, a, _, _ in DAYS]
    vals = [max(det / tot, 1e-6) for _, _, tot, det in DAYS]
    x = np.arange(len(DAYS))
    cols = [WARN if v > 0.05 else ACC for v in vals]
    ax.bar(x, vals, color=cols, alpha=0.85, width=0.62)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5.4)
    ax.set_ylabel("Recall (log scale)")
    ax.annotate("both DoS days;\nboth families in training",
                xy=(1.5, 0.34), xytext=(2.8, 0.52), fontsize=6,
                arrowprops=dict(arrowstyle="->", lw=0.7, color=GREY))
    save(fig, out, "fig5_day_recall")


def fig6_psi_saturation(out, data_dir):
    path = find(data_dir, "06_drift_vectors.csv")
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    if path:
        import pandas as pd
        d = pd.read_csv(path)
        ax.hist(d.psi_mean.dropna(), bins=28, color=ACC, alpha=0.8,
                edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Windows")
    else:
        # summary-only fallback: show the measured range, not an invented shape
        for yr, s in PSI.items():
            ax.errorbar([s["mean"]], [1 if yr == "2018" else 2],
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
    save(fig, out, "fig6_psi_saturation")


def fig7_psi_vs_time(out, data_dir):
    path = find(data_dir, "06_drift_vectors.csv")
    if not path:
        print("  SKIP fig7 — 06_drift_vectors.csv not found (needs per-window series)")
        return
    import pandas as pd
    d = pd.read_csv(path)
    d["win_key"] = d.win_key.astype(str)
    d = d[~d.win_key.str.startswith("2019")].sort_values("win_key").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    x = np.arange(len(d))
    ax.scatter(x, d.psi_mean, s=7, color=ACC, alpha=0.75, edgecolors="none")
    ax.axhline(0.25, ls="--", lw=1.0, color=WARN)
    ax.text(len(d) * 0.02, 0.33, "alarm threshold", fontsize=6, color=WARN)
    ax.set_xlabel("Window index, CSE-CIC-IDS2018")
    ax.set_ylabel("Mean PSI")
    ax.set_title(fr"Spearman $\rho$ = {RHO_TIME:.3f} with window order", fontsize=7.5)
    save(fig, out, "fig7_psi_vs_time")


def fig8_output_space(out):
    fig, ax = plt.subplots(figsize=(COL2, 2.3))
    names = [e[0] for e in ENTROPY]
    ent   = [e[1] for e in ENTROPY]
    patk  = [e[2] for e in ENTROPY]
    unc   = [e[3] for e in ENTROPY]
    x = np.arange(len(names)); w = 0.26
    ax.bar(x - w, ent,  w, label="Mean predictive entropy", color=ACC, alpha=0.85)
    ax.bar(x,     unc,  w, label="Uncertain-flow fraction", color=GREY, alpha=0.85)
    ax.bar(x + w, patk, w, label="Mean P(attack) on true attacks", color=WARN, alpha=0.85)
    ax.axhline(0.5, ls="--", lw=1.0, color="k")
    ax.text(len(names) - 0.55, 0.53, "decision threshold", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6.6)
    ax.set_ylabel("Statistic value")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="upper left", framealpha=0.9, ncol=1)
    ax.set_title("The detector is confident and wrong: P(attack) never approaches 0.5",
                 fontsize=7.5)
    save(fig, out, "fig8_output_space")


def fig9_label_budget(out):
    fig, ax = plt.subplots(figsize=(COL1, 2.1))
    for name, series in LABEL_BUDGET.items():
        style = "-o" if name in ("DoS", "Web") else "-"
        ax.plot(BUDGETS, series, style, lw=1.1, ms=3, label=name, alpha=0.9)
    ax.axvline(25, ls="--", lw=1.0, color=WARN)
    ax.text(27, 0.845, "n = 25:\nall families certain", fontsize=6, color=WARN)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
    ax.set_ylim(0.78, 1.02)
    ax.set_xlabel("Labelled attack flows per family")
    ax.set_ylabel("P(correctly establishing failure)")
    ax.legend(fontsize=5.8, ncol=2, loc="lower right", framealpha=0.9)
    save(fig, out, "fig9_label_budget")


def fig10_battery(out):
    """Supplementary: the drift battery. Use in 4.5.1 if a tenth figure is wanted."""
    fig, ax = plt.subplots(figsize=(COL2, 2.8))
    names = [b[0] for b in BATTERY]
    y = np.arange(len(names)); w = 0.38
    ax.barh(y + w/2, [abs(b[1]) for b in BATTERY], w,
            label=r"|$\rho$| with current $F_1$", color=ACC, alpha=0.85)
    ax.barh(y - w/2, [abs(b[3]) for b in BATTERY], w,
            label=r"|$\rho$| with attack fraction", color=WARN, alpha=0.7)
    for i, b in enumerate(BATTERY):
        if not b[4]:
            ax.text(-0.02, i, "admitted", ha="right", va="center",
                    fontsize=5.6, color=ACC)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.4)
    ax.set_xlabel(r"|Spearman $\rho$|")
    ax.set_xlim(0, 0.82)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.invert_yaxis()
    ax.set_title("A statistic is admitted only when it tracks performance "
                 "more than workload", fontsize=7.5)
    save(fig, out, "fig10_drift_battery")


def fig11_partial(out):
    """Supplementary: partial correlation ladder with the bootstrap interval."""
    fig, ax = plt.subplots(figsize=(COL1, 1.9))
    labels = [p[0] for p in PARTIAL]
    vals = [p[1] for p in PARTIAL]
    y = np.arange(len(labels))
    ax.barh(y, vals, color=ACC, alpha=0.85, height=0.55)
    ax.axvspan(BOOT_CI[0], BOOT_CI[1], color=WARN, alpha=0.15)
    ax.axvline(0, color="k", lw=0.8)
    ax.text(BOOT_CI[1] + 0.01, len(labels) - 0.4,
            "bootstrap 95% CI\n[-0.445, -0.118]", fontsize=5.8, color=WARN)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.4)
    ax.set_xlabel(r"Partial Spearman $\rho$ with next-window $F_1$")
    ax.set_xlim(-0.55, 0.12)
    ax.invert_yaxis()
    save(fig, out, "fig11_partial_correlation")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="directory holding pipeline CSVs (for per-window figures)")
    ap.add_argument("--out", default="./figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"writing to {a.out}")
    fig2_window_f1(a.out, a.data)
    fig3_survival(a.out)
    fig4_family_recall(a.out)
    fig5_day_recall(a.out)
    fig6_psi_saturation(a.out, a.data)
    fig7_psi_vs_time(a.out, a.data)
    fig8_output_space(a.out)
    fig9_label_budget(a.out)
    fig10_battery(a.out)
    fig11_partial(a.out)
    print("done. Figs 2-9 are the numbered set; 10-11 are spares for 4.5.")


if __name__ == "__main__":
    main()
