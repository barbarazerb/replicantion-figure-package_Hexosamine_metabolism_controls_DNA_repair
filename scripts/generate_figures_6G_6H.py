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
TIME_COLORS = {"24h": (150, 205, 235), "48h": (24, 131, 192)}     # light blue = 24h, blue = 48h
TIME_COLORS_GREEN = {"24h": (150, 240, 150), "48h": (0, 175, 40)}  # bright light green = 24h, bright green = 48h
DOTS_GREEN  = {"24h": (225, 252, 225), "48h": (0, 95, 20)}
BAR_COLORS_PANC = {"vs CTRL": (0, 160, 40), "vs COMBO": (120, 40, 160), "time": (0, 0, 0)}  # green, purple
BAR_COLORS  = {"vs CTRL": (21, 101, 192), "vs COMBO": (183, 28, 28), "time": (0, 0, 0)}

FIGURE = {
    "MIAPaCa-2_RUVBL2": {
        "file":       "Figure_6G_MIA.xlsx",
        "ylabel":     "RUVBL2 Nucleus/Cytoplasm ratio",
        "time_colors": TIME_COLORS,
        "axes_cm":    (5.0, 4.0),          # X axis, Y axis (cm)
        "y_step":     1,                   # Y-axis tick step
        # "y_max":    4,                   # Y-axis maximum (if absent: computed from the data)
    },
    "PANC-1_RUVBL2": {
        "file":       "Figure_6H_PANC.xlsx",
        "ylabel":     "RUVBL2 Nucleus/Cytoplasm ratio",
        "time_colors": TIME_COLORS_GREEN,
        "dot_colors":  DOTS_GREEN,
        "bar_colors":  BAR_COLORS_PANC,
        "axes_cm":    (5.0, 4.0),          # X axis, Y axis (cm)
        "y_step":     1,                   # Y-axis tick step
        # "y_max":    4,                   # Y-axis maximum (if absent: computed from the data)
    },
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasheets")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FORMATS = ["tif", "pdf", "svg"]

# =====================================================================
# COMMON SETTINGS
# =====================================================================
CONDITIONS  = ["CTRL", "FR054", "GEM", "COMBO"]
TIMES       = ["24h", "48h"]            # left half, right half
DOT_COLORS  = {"24h": (205, 232, 248), "48h": (10, 55, 95)}      # default experiment dots (RGB)
SHOW_TIME_LEGEND = False                # small 24h / 48h legend
STATS_MODE  = "mixed"                   # P shown in the plot: "mixed" (mixed model) or "robust" (paired
                                        # comparisons on experiment means, each with its own variability)
REMOVE_OUTLIERS = True                  # True = excludes extreme outliers (applies to plot and statistics)
IQR_FACTOR  = 3.0                       # Tukey on log2, per sample: outside Q1 - 3*IQR / Q3 + 3*IQR

# comparisons shown in the plot: all (5 per time point + 24h vs 48h for each condition)
SHOW_TIME_BARS = True                   # black lines: 48h vs 24h within each condition
SHOW_COND_BARS = True                   # blue (vs CTRL) and red (vs COMBO) lines, for each time point

PAIRS = [("CTRL", "FR054"), ("CTRL", "GEM"), ("CTRL", "COMBO"), ("GEM", "COMBO"), ("FR054", "COMBO")]

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "axes.labelweight": "bold",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def comparison_key(row, time):
    return "vs CTRL" if row.A == "CTRL" else "vs COMBO"


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
            parts.append(pd.DataFrame({"Exp": str(reps[c]).replace("Rep", "Exp"), "Time": str(times[c]).strip(),
                                       "Cond": str(conds[c]).strip(), "V": v}))
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
    """Holm-corrected p-values (main model + sensitivity) used to label the figure."""
    means = d.groupby(["Exp", "Time", "Cond"]).L.mean()

    between_cond = {}
    for t in TIMES:                                         # conditions, within each time point
        x = d[d.Time == t]
        fit, fe, V, df_err = fit_mixed_model(x, "Cond", CONDITIONS, "Exp")
        _, _, _, p_holm = contrast_pvalues(fe, V, df_err, CONDITIONS, PAIRS)
        t_means = means.xs(t, level="Time").unstack()
        s_p = pd.Series([paired_ttest_p(t_means, a, b) for a, b in PAIRS])
        between_cond[t] = pd.DataFrame({"A": [a for a, _ in PAIRS], "B": [b for _, b in PAIRS],
                                        "p_holm": p_holm, "s_p_holm": holm(s_p.fillna(1))})

    rows = []
    for cond in CONDITIONS:                                 # 48h vs 24h, within each condition
        x = d[d.Cond == cond]
        fit, fe, V, df_err = fit_mixed_model(x, "Time", TIMES, "Exp")
        _, _, p_raw, _ = contrast_pvalues(fe, V, df_err, TIMES, [(TIMES[0], TIMES[1])])
        s_p = paired_ttest_p(means.xs(cond, level="Cond").unstack(), TIMES[0], TIMES[1])
        rows.append((cond, p_raw[0], s_p))
    between_time = pd.DataFrame(rows, columns=["Cond", "p_raw", "s_p"])
    between_time["p_holm"] = holm(between_time.p_raw)
    between_time["s_p_holm"] = holm(between_time.s_p.fillna(1))

    return {"between_cond": between_cond, "between_time": between_time[["Cond", "p_holm", "s_p_holm"]]}


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
                                       default_dot_colors=DOT_COLORS, default_bar_colors=BAR_COLORS)
        print("  saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
