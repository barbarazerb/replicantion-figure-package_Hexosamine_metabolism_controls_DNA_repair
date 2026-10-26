import logging
import os
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_common import contrast_pvalues, fit_mixed_model, holm, make_split_violin_plot, paired_ttest_p

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# =====================================================================
# FIGURES
# =====================================================================
GENO_COLORS = {"RUVBL2 WT": (255, 212, 0), "RUVBL2 MUT": (175, 95, 235)}     # yellow = WT, purple = MUT
GENO_DOTS   = {"RUVBL2 WT": (255, 246, 200), "RUVBL2 MUT": (70, 0, 115)}     # dots: light / dark
GENO_BARS   = {"RUVBL2 WT": (184, 135, 12), "RUVBL2 MUT": (106, 14, 173), "time": (0, 0, 0)}
# statistics lines: gold = CTRL vs GEM in WT, purple = CTRL vs GEM in MUT, black = WT vs MUT

FIGURE = {
    "HEK293T_RUVBL2_WT_MUT": {
        "file":        "Figure_7B_RUVB2_WT_MUT.xlsx",
        "ylabel":      "HA-RUVBL2 Nucleus/Cytoplasm ratio",
        "time_colors": GENO_COLORS,
        "dot_colors":  GENO_DOTS,
        "bar_colors":  GENO_BARS,
        "axes_cm":     (4.0, 4.0),          # X axis, Y axis (cm)
        "y_step":      1,                   # Y-axis tick step
    },
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasheets")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FORMATS = ["tif", "pdf", "svg"]

# =====================================================================
# COMMON SETTINGS
# =====================================================================
CONDITIONS  = ["CTRL", "GEM"]                         # treatments on the X axis
TIMES       = ["RUVBL2 WT", "RUVBL2 MUT"]             # left half, right half
SHOW_TIME_LEGEND = False
STATS_MODE  = "mixed"                   # P shown in the plot: "mixed" (mixed model) or "robust" (paired
                                        # comparisons on experiment means, each with its own variability)
REMOVE_OUTLIERS = True                  # True = excludes extreme outliers (applies to plot and statistics)
IQR_FACTOR  = 3.0                       # Tukey on log2, per sample: outside Q1 - 3*IQR / Q3 + 3*IQR

SHOW_TIME_BARS = True                   # black lines: WT vs MUT within each treatment
SHOW_COND_BARS = True                   # gold/purple lines: CTRL vs GEM within each genotype

PAIRS = [("CTRL", "GEM")]
HOLM_FAMILIES = "separate"                  # "all" = Holm across the 4 comparisons together;
                                        # "separate" = separate Holm: CTRL vs GEM (2) and WT vs MUT (2)

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "axes.labelweight": "bold",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def comparison_key(row, time):
    return time


# =====================================================================
# DATA
# =====================================================================
def load_data(path):
    """Rows: 'Rep' / time / condition, then one cell per row. Empty columns = missing samples."""
    raw = pd.read_excel(path, header=None)
    rep_row = raw.index[raw.apply(lambda r: r.astype(str).str.startswith("Rep").any(), axis=1)][0]
    reps, times, conds = raw.iloc[rep_row].ffill(), raw.iloc[rep_row + 1].ffill(), raw.iloc[rep_row + 2]
    rows = raw.iloc[rep_row + 3:]
    rows = rows[rows.iloc[:, 0].isna()]           # excludes any "Mean" rows
    parts = []
    for c in raw.columns:
        if pd.isna(conds[c]):
            continue
        v = pd.to_numeric(rows[c], errors="coerce").dropna().values
        if len(v):
            parts.append(pd.DataFrame({"Exp": str(reps[c]).replace("Rep", "Exp"), "Time": str(conds[c]).strip(),
                                       "Cond": str(times[c]).strip(), "V": v}))   # Time = genotype, Cond = treatment
    d = pd.concat(parts, ignore_index=True)
    if (d.V <= 0).any():
        raise ValueError("values <= 0: the log2 transformation is not possible")
    d = d.assign(L=np.log2(d.V))
    if REMOVE_OUTLIERS:
        def keep(l):
            q1, q3 = np.percentile(l, [25, 75]); iqr = q3 - q1
            return (l >= q1 - IQR_FACTOR * iqr) & (l <= q3 + IQR_FACTOR * iqr)
        k = d.groupby(["Exp", "Time", "Cond"]).L.transform(lambda l: keep(l.values)).astype(bool)
        d = d[k].reset_index(drop=True)
    return d


# =====================================================================
# STATISTICS
# =====================================================================
def compute_stats(d):
    """Single mixed model with the 4 genotype x treatment combinations (as for gammaH2AX):
    log2 ~ Group + (1 | Experiment) + (1 | Experiment:Group). The 4 comparisons shown in the plot
    are Holm-corrected within their family (CTRL vs GEM; WT vs MUT), or all together."""
    groups = [f"{c}|{t}" for t in TIMES for c in CONDITIONS]          # CTRL|WT, GEM|WT, CTRL|MUT, GEM|MUT
    x = d.assign(G=d.Cond + "|" + d.Time)
    fit, fe, V, df_err = fit_mixed_model(x, "G", groups, "Exp")
    means = x.groupby(["Exp", "G"]).L.mean().unstack()

    rows = []                                    # the 4 comparisons shown in the plot
    for t in TIMES:                              # CTRL vs GEM within each genotype
        for a, b in PAIRS:
            rows.append(dict(kind="cond", T=t, A=a, B=b, pa=f"{a}|{t}", pb=f"{b}|{t}"))
    for cnd in CONDITIONS:                       # WT vs MUT within each treatment
        rows.append(dict(kind="time", Cond=cnd, pa=f"{cnd}|{TIMES[0]}", pb=f"{cnd}|{TIMES[1]}"))
    for r in rows:
        r["p_raw"] = contrast_pvalues(fe, V, df_err, groups, [(r["pa"], r["pb"])])[2][0]
        r["s_p"] = paired_ttest_p(means, r["pa"], r["pb"])
    tab = pd.DataFrame(rows)

    if HOLM_FAMILIES == "separate":              # two families: CTRL vs GEM / WT vs MUT
        for k in ("cond", "time"):
            m = tab.kind == k
            tab.loc[m, "p_holm"] = holm(tab.loc[m, "p_raw"].values)
            tab.loc[m, "s_p_holm"] = holm(tab.loc[m, "s_p"].fillna(1).values)
    else:                                        # Holm across the 4 comparisons
        tab["p_holm"] = holm(tab.p_raw); tab["s_p_holm"] = holm(tab.s_p.fillna(1))

    between_cond = {t: tab[(tab.kind == "cond") & (tab["T"] == t)].reset_index(drop=True) for t in TIMES}
    between_time = tab[tab.kind == "time"].reset_index(drop=True)
    return {"between_cond": between_cond, "between_time": between_time}


# =====================================================================
# MAIN
# =====================================================================
def main():
    names = sys.argv[1:] or list(FIGURE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for name in names:
        cfg = FIGURE[name]
        print(f"=== {name} ===")
        d = load_data(os.path.join(DATA_DIR, cfg["file"]))
        st = compute_stats(d)
        out = os.path.join(OUTPUT_DIR, f"split_violin_{name}")
        saved = make_split_violin_plot(cfg, d, st, out, conditions=CONDITIONS, times=TIMES,
                                       comparison_key=comparison_key, formats=FORMATS,
                                       stats_mode=STATS_MODE, show_time_bars=SHOW_TIME_BARS,
                                       show_cond_bars=SHOW_COND_BARS, show_time_legend=SHOW_TIME_LEGEND,
                                       default_dot_colors=GENO_DOTS, default_bar_colors=GENO_BARS)
        print("  saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
