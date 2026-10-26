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
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from scipy.stats import gaussian_kde

from plot_common import contrast_pvalues, fit_mixed_model, p_text, save_plot

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# =====================================================================
# PLOTS TO GENERATE
# =====================================================================
COL_MIA = {"CTRL": (0, 0, 192), "FR054": (98, 255, 255), "GEM": (85, 160, 251), "COMBO": (200, 222, 249)}
COL_PANC = {"CTRL": (0, 64, 0), "FR054": (0, 128, 0), "GEM": (192, 192, 0), "COMBO": (160, 255, 160)}
DOTS_PANC = {"CTRL": (140, 175, 140), "FR054": (110, 210, 180)}   # dot colors chosen by hand
FORMATO_24H = {"axes_cm": (2.0, 1.5), "y_step": 2000, "y_max": 8000,
               "rep_offset": 0.28, "font_pt": 5, "ylabel_pad": 1.2}

FIGURE = {
    "MIAPaCa-2_48h": {"file": "Figure_2F_MIA.xlsx", "colors_rgb": COL_MIA},
    "PANC-1_48h":    {"file": "Figure_2H_PANC.xlsx", "colors_rgb": COL_PANC, "dot_colors_rgb": DOTS_PANC},
    "MIAPaCa-2_24h": {"file": "Supl__Figure_2F.xlsx", "colors_rgb": COL_MIA, **FORMATO_24H},
    "PANC-1_24h":    {"file": "Suppl__Figure_2H.xlsx", "colors_rgb": COL_PANC,
                      "dot_colors_rgb": DOTS_PANC, **FORMATO_24H},
}
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "datasheets")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
FORMATS = ["tif", "pdf", "svg"]          # remove the ones you don't need

# =====================================================================
# COMMON SETTINGS (apply to all plots, unless overridden in FIGURE)
# =====================================================================
CONDITIONS  = ["CTRL", "FR054", "GEM", "COMBO"]
LINE_COLORS = {"CTRL": "white", "FR054": "black", "GEM": "black", "COMBO": "black"}  # median/quartiles
AXES_CM     = (2.5, 3.0)                # X axis, Y axis length (cm)
Y_STEP      = 2000                      # Y-axis tick step
Y_EXP       = 3                         # Y-axis labels as multiples of 10^Y_EXP
FS_AXES, FS_PVAL = 7, 5                 # font: axes, p-value (pt); "font_pt" overrides both
LW          = 1.0                       # axes, ticks, statistics lines (pt)
VIOLIN_LW   = 0.75                      # violin outline, median, quartiles (pt)
TICK_LEN    = 3                         # tick length (pt)
HALF_W      = 0.42                      # maximum half-width of the violin
DOT_SIZE    = 9                         # dot area (pt^2)
DOT_EDGE_LW = 0.5                       # dot edge width (pt)
DOT_DARKEN  = 0.55                      # dot color = darkened violin color
REP_OFFSET  = {"Rep 1": -1, "Rep 2": 0, "Rep 3": 1}   # relative position of the dots
REP_STEP    = 0.2                       # distance between dots ("rep_offset" overrides it)
YLABEL_PARTS = ("MFI ", "γ", "H2AX")    # the gamma is in Symbol font, the rest in Arial
SYMBOL_FONTS = ["Symbol", "Standard Symbols PS"]
GAMMA_GAP   = 1.5                       # space after the gamma (pt)
GAMMA_BOLD  = 0.35                      # simulated bold for the gamma (pt)
YLABEL_PAD  = -2                        # distance between Y-axis title and numbers (pt)
SUP_SCALE, SUP_RISE = 0.7, 0.45         # p-value exponent: size, raise
ALPHA       = 0.05                      # above this threshold "ns" is shown
P_ABOVE, LABEL_GAP = 2.55, 1.70         # p above the violin; spacing between nearby labels (pt)
BAR_GAP, BAR_STEP, BAR_TEXT = 3.40, 11.055, 0.85   # comparison lines (pt)
SCALE       = 1000                      # data in the files are x10^3 -> absolute values
DPI         = 300

# planned comparisons: vs CTRL (written above the violins) and lines between conditions
STATS_VS_CTRL = ["FR054", "GEM", "COMBO"]
STATS_BARS    = [("GEM", "COMBO"), ("FR054", "COMBO")]
PAIRS = [("CTRL", c) for c in STATS_VS_CTRL] + STATS_BARS      # (A, B): B vs A

plt.rcParams.update({
    "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.weight": "bold", "axes.labelweight": "bold",
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


# =====================================================================
# DATA
# =====================================================================
def load_data(path):
    """Reads the Excel file ('Rep' row, condition row, then one cell per row)."""
    raw = pd.read_excel(path, header=None)
    rep_row = raw.index[raw.apply(lambda r: r.astype(str).str.startswith("Rep").any(), axis=1)][0]
    cond_row = rep_row + 1
    reps, conds = raw.iloc[rep_row].ffill(), raw.iloc[cond_row]
    rows = raw.iloc[cond_row + 1:]
    rows = rows[rows.iloc[:, 0].isna()]          # excludes the "Mean" row at the bottom of the sheet
    return pd.concat([
        pd.DataFrame({"Replicate": reps[c], "Condition": str(conds[c]).strip(),
                      "MFI": pd.to_numeric(rows[c], errors="coerce").dropna().values * SCALE})
        for c in raw.columns if pd.notna(conds[c])
    ], ignore_index=True)


# =====================================================================
# STATISTICS
# =====================================================================
def compute_stats(data):
    """Holm-corrected p-values for the planned comparisons, used to label the figure."""
    d = data.assign(L=np.log2(data.MFI))
    fit, fe, V, df_err = fit_mixed_model(d, "Condition", CONDITIONS, "Replicate")
    _, _, _, p_holm = contrast_pvalues(fe, V, df_err, CONDITIONS, PAIRS)
    return p_holm


# =====================================================================
# PLOT
# =====================================================================
def sci_fmt(v, _):
    if v == 0:
        return "0"
    mant = f"{v / 10**Y_EXP:.1f}".rstrip("0").rstrip(".")
    return f"{mant}×10" + str(Y_EXP).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))


def find_gamma():
    """Returns (gamma character, family, uses_Symbol).
    Tries the Symbol font; if the gamma cannot be found there (e.g. Windows'
    symbol.ttf with non-Unicode encoding), falls back to the Unicode gamma of the text font."""
    try:
        path = fm.findfont(fm.FontProperties(family=SYMBOL_FONTS, weight="normal"),
                           fallback_to_default=False)
        ft = fm.get_font(path)            # same object matplotlib uses for drawing
        for ch in ("γ", ""):
            if ft.get_char_index(ord(ch)):
                return ch, ft.family_name, True
    except Exception:
        pass
    path = fm.findfont(fm.FontProperties(family=plt.rcParams["font.family"], weight="bold"))
    fam = fm.get_font(path).family_name
    print(f"  NOTE: gamma not available in the Symbol font, using the gamma from {fam}")
    return "γ", fam, False


def make_plot(cfg, data, p_holm, out_base):
    fs_axes = cfg.get("font_pt", FS_AXES)
    fs_p = cfg.get("font_pt", FS_PVAL)
    x_cm, y_cm = cfg.get("axes_cm", AXES_CM)
    k = len(CONDITIONS)

    violin_col = [tuple(v / 255 for v in cfg["colors_rgb"][c]) for c in CONDITIONS]
    manual = cfg.get("dot_colors_rgb", {})
    dot_col = [tuple(v / 255 for v in manual[c]) if c in manual else tuple(v * DOT_DARKEN for v in col)
               for c, col in zip(CONDITIONS, violin_col)]
    rep_means = data.groupby(["Condition", "Replicate"])["MFI"].mean().reset_index()

    # figure with fixed-size axes area
    CM = 1 / 2.54
    ML, MB, MT, MR = 1.6, 1.2, 1.4, 2.4
    W, H = ML + x_cm + MR, MB + y_cm + MT
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([ML / W, MB / H, x_cm / W, y_cm / H])
    pos = np.arange(1, k + 1)

    # violins, median, quartiles, dots
    for i, cond in enumerate(CONDITIONS):
        y = data.loc[data.Condition == cond, "MFI"].values
        grid = np.linspace(y.min(), y.max(), 400)
        kde = gaussian_kde(y); dmax = kde(grid).max()
        w = kde(grid) / dmax * HALF_W
        ax.fill_betweenx(grid, pos[i] - w, pos[i] + w, facecolor=violin_col[i],
                         edgecolor="black", lw=VIOLIN_LW, joinstyle="round", zorder=2)
        for q, ls in zip(np.percentile(y, [50, 25, 75]), ["-", (0, (3, 2.5)), (0, (3, 2.5))]):
            hw = kde(q)[0] / dmax * HALF_W
            ax.plot([pos[i] - hw + 0.03, pos[i] + hw - 0.03], [q, q], color=LINE_COLORS[cond],
                    lw=VIOLIN_LW, ls=ls, solid_capstyle="butt", zorder=3)
        for _, r in rep_means[rep_means.Condition == cond].iterrows():
            ax.scatter(pos[i] + REP_OFFSET.get(r.Replicate, 0) * cfg.get("rep_offset", REP_STEP), r.MFI,
                       s=DOT_SIZE, facecolor=dot_col[i], edgecolor="black", lw=DOT_EDGE_LW, zorder=5)

    # axes
    ymax = 10000 if data.MFI.max() <= 10000 else int(np.ceil(data.MFI.max() / 2000) * 2000)
    ymax = cfg.get("y_max", ymax)
    if data.MFI.max() > ymax:
        print(f"  WARNING: values up to {data.MFI.max():.0f} exceed y_max = {ymax}")
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 1, cfg.get("y_step", Y_STEP)))
    ax.yaxis.set_major_formatter(FuncFormatter(sci_fmt))
    ax.set_xlim(0.4, k + 0.6)
    ax.set_xticks(pos, CONDITIONS, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", labelsize=fs_axes, width=LW, length=TICK_LEN, pad=2)
    ax.tick_params(axis="x", length=0, pad=3)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(LW)
    ax.spines[["top", "right"]].set_visible(False)

    # Y-axis title: "MFI " + gamma (Symbol, simulated bold) + "H2AX"
    gamma, gamma_fam, gamma_sym = find_gamma()
    ax.set_ylabel("".join(YLABEL_PARTS), fontsize=fs_axes, labelpad=cfg.get("ylabel_pad", YLABEL_PAD), alpha=0)
    fig.canvas.draw()
    ref = ax.yaxis.label.get_window_extent()
    to_fig = fig.transFigure.inverted()
    pieces, y_run = [], 0.0
    for txt, fam in zip((YLABEL_PARTS[0], gamma, YLABEL_PARTS[2]), (None, gamma_fam, None)):
        kw = dict(rotation=90, rotation_mode="anchor", ha="left", va="baseline",
                  fontsize=fs_axes, fontweight="bold")
        if fam and gamma_sym:
            kw.update(family=fam, fontweight="normal",
                      path_effects=[pe.withStroke(linewidth=GAMMA_BOLD, foreground="black")])
        t = fig.text(0, y_run, txt, **kw)
        fig.canvas.draw()
        y_run = to_fig.transform(t.get_window_extent().p1)[1]
        if fam:
            y_run += GAMMA_GAP / 72 / fig.get_figheight()
        pieces.append(t)
    fig.canvas.draw()
    union = Bbox.union([p.get_window_extent() for p in pieces])
    (rx1, ryc), (ux1, uyc) = to_fig.transform([(ref.x1, (ref.y0 + ref.y1) / 2),
                                              (union.x1, (union.y0 + union.y1) / 2)])
    for p in pieces:
        x, y = p.get_position()
        p.set_position((x + rx1 - ux1, y + ryc - uyc))

    # p-value
    to_data = ax.transData.inverted()
    pt = lambda v: v / (y_cm / 2.54 * 72) * ymax          # typographic points -> data units

    def draw_p(x, y, value):
        text, exp = value
        if text == "ns":
            t = ax.text(x, y, "ns", ha="center", va="bottom", fontsize=fs_p,
                        fontweight="normal", clip_on=False)
            fig.canvas.draw()
            return t.get_window_extent(), [t]
        kw = dict(fontsize=fs_p, fontweight="normal", annotation_clip=False)
        p_lbl = ax.text(x, y, "P", ha="left", va="bottom", fontsize=fs_p,
                        fontweight="normal", fontstyle="italic", clip_on=False)
        parts = [p_lbl, ax.annotate(text, xy=(1, 0), xycoords=p_lbl, ha="left", va="bottom", **kw)]
        if exp:
            parts.append(ax.annotate(exp, xy=(1, SUP_RISE), xycoords=parts[1], ha="left", va="bottom",
                                     **{**kw, "fontsize": fs_p * SUP_SCALE}))
        fig.canvas.draw()
        x_left = to_data.transform(parts[0].get_window_extent().p0)[0]
        x_right = to_data.transform(parts[-1].get_window_extent().p1)[0]
        p_lbl.set_x(x - (x_right - x_left) / 2)               # centers everything on x
        fig.canvas.draw()
        return Bbox.union([p.get_window_extent() for p in parts]), parts

    idx = dict(zip(CONDITIONS, pos))
    n1 = len(STATS_VS_CTRL)
    placed = []
    for (a, b), p in zip(PAIRS[:n1], p_holm[:n1]):
        y = data.loc[data.Condition == b, "MFI"].max() + pt(P_ABOVE)
        while True:
            bb, parts = draw_p(idx[b], y, p_text(p, ALPHA))
            clash = [c for c in placed if c.x1 > bb.x0 and c.x0 < bb.x1 and c.y1 > bb.y0 and c.y0 < bb.y1]
            if not clash:
                placed.append(bb); break
            for q in parts:                                    # overlapping: move it up
                q.remove()
            y = to_data.transform((0, max(c.y1 for c in clash)))[1] + pt(LABEL_GAP)
    text_top = max([to_data.transform((0, b.y1))[1] for b in placed], default=0)
    for j, ((a, b), p) in enumerate(zip(PAIRS[n1:], p_holm[n1:])):
        yb = text_top + pt(BAR_GAP + BAR_STEP * j)
        ax.plot([idx[a], idx[b]], [yb, yb], color="black", lw=LW, clip_on=False, solid_capstyle="butt")
        draw_p((idx[a] + idx[b]) / 2, yb + pt(BAR_TEXT), p_text(p, ALPHA))

    def fix_gamma_svg(svg):
        if gamma_sym:
            svg = svg.replace(f"font-family: '{gamma_fam}'", "font-family: 'Symbol', 'Standard Symbols PS'")
        return svg

    saved = save_plot(fig, out_base, FORMATS, DPI, svg_postprocess=fix_gamma_svg)
    plt.close(fig)
    return saved


# =====================================================================
# MAIN
# =====================================================================
def main():
    names = sys.argv[1:] or list(FIGURE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    font = fm.findfont(fm.FontProperties(family=plt.rcParams["font.family"], weight="bold"))
    fname = fm.get_font(font).family_name
    print(f"Text font: {fname}" + ("" if fname == "Arial" else "  (WARNING: Arial not found, using a substitute)"))
    for name in names:
        if name not in FIGURE:
            print(f"\n{name}: unknown figure. Available: {', '.join(FIGURE)}"); continue
        cfg = FIGURE[name]
        print(f"\n=== {name} ===")
        file_path = os.path.join(DATA_DIR, cfg["file"])
        if not os.path.exists(file_path):
            print(f"  file {file_path} not found: skipping"); continue
        data = load_data(file_path)
        p_holm = compute_stats(data)
        out_base = os.path.join(OUTPUT_DIR, f"violin_plot_{name}")
        saved = make_plot(cfg, data, p_holm, out_base)
        print("  saved: " + ", ".join(saved))


if __name__ == "__main__":
    main()
