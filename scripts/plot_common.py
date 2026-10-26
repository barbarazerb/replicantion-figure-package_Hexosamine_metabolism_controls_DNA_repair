"""Shared statistics and I/O helpers used by the figure-generation scripts.

All three scripts (generate_figure_gH2AX.py, generate_split_violin.py,
generate_split_violin_WT_MUT.py) only need the mixed-model p-values to label
the plots; none of this module computes or stores anything for a separate
statistics report.
"""
import io
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from PIL import Image
from scipy.stats import gaussian_kde, t as tdist


def holm(p):
    """Holm-Bonferroni step-down correction."""
    p = np.asarray(p, float); out = np.empty(len(p)); run = 0
    for k, j in enumerate(np.argsort(p)):
        run = max(run, min(1, (len(p) - k) * p[j])); out[j] = run
    return out


def contrast(levels, a, b):
    """Contrast vector for b - a (first level of `levels` is the reference)."""
    c = np.zeros(len(levels))
    if levels.index(a) > 0: c[levels.index(a)] -= 1
    if levels.index(b) > 0: c[levels.index(b)] += 1
    return c


def fit_mixed_model(df, factor, levels, group_col):
    """Mixed model log2(L) ~ factor + (1|group) + (1|group:factor).
    Returns (fit, fixed effects, covariance of fixed effects, residual df)."""
    x = df.assign(F=pd.Categorical(df[factor], levels))
    fit = smf.mixedlm("L ~ F", x, groups=group_col, re_formula="1",
                      vc_formula={"camp": "0 + C(F)"}).fit(reml=True, method=["nm", "powell"], maxiter=5000)
    if not fit.converged:
        raise RuntimeError("the mixed model did not converge")
    k = len(levels)
    n_samp = x.groupby([group_col, "F"], observed=True).ngroups
    df_err = n_samp - x[group_col].nunique() - (k - 1)
    return fit, fit.fe_params.iloc[:k].values, fit.cov_params().iloc[:k, :k].values, df_err


def contrast_pvalues(fe, V, df_err, levels, pairs):
    """Raw and Holm-corrected two-sided p-values for each (a, b) pair in `pairs` (b vs a)."""
    est = np.array([contrast(levels, a, b) @ fe for a, b in pairs])
    se = np.array([np.sqrt(contrast(levels, a, b) @ V @ contrast(levels, a, b)) for a, b in pairs])
    p_raw = 2 * tdist.sf(np.abs(est / se), df_err)
    return est, se, p_raw, holm(p_raw)


def paired_ttest_p(means, a, b):
    """Two-sided paired t-test p-value on experiment-level log2 means (only experiments with
    both samples). NaN if fewer than 2 such experiments."""
    dd = (means[b] - means[a]).dropna().values
    n = len(dd)
    if n < 2:
        return np.nan
    se = dd.std(ddof=1) / np.sqrt(n)
    return 2 * tdist.sf(abs(dd.mean() / se), n - 1)


def p_text(p, alpha=0.05):
    """("ns", None) | ("=0.014", None) | ("=3.3E", "-6")"""
    if p >= alpha:
        return "ns", None
    if p >= 0.001:
        return f"={p:.3f}", None
    mant, exp = f"{p:.1E}".split("E")
    return f"={mant}E", str(int(exp)).replace("-", "−")


def save_plot(fig, out_base, formats, dpi=300, svg_postprocess=None):
    """Saves fig as .tif (RGB, LZW), .pdf and/or .svg depending on `formats`.
    svg_postprocess, if given, is a function(svg_text) -> svg_text applied before writing the .svg."""
    save_kw = dict(bbox_inches="tight", pad_inches=0.05, facecolor="white")
    saved = []
    if "tif" in formats:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, **save_kw); buf.seek(0)
        Image.open(buf).convert("RGB").save(out_base + ".tif", dpi=(dpi, dpi), compression="tiff_lzw")
        saved.append(out_base + ".tif")
    if "pdf" in formats:
        fig.savefig(out_base + ".pdf", **save_kw); saved.append(out_base + ".pdf")
    if "svg" in formats:
        fig.savefig(out_base + ".svg", **save_kw)
        if svg_postprocess:
            svg = open(out_base + ".svg", encoding="utf-8").read()
            open(out_base + ".svg", "w", encoding="utf-8").write(svg_postprocess(svg))
        saved.append(out_base + ".svg")
    return saved


def svg_mm_postprocess(svg):
    """Rewrites the SVG width/height from pt to mm (explicit physical measurements)."""
    return re.sub(r'(width|height)="([\d.]+)pt"',
                  lambda m: f'{m.group(1)}="{float(m.group(2)) * 25.4 / 72:.3f}mm"', svg, count=2)


# =====================================================================
# Split violin plot (shared by generate_split_violin.py and generate_split_violin_WT_MUT.py)
# =====================================================================
_SV_FS_AXES, _SV_FS_PVAL = 7, 5
_SV_LW, _SV_VIOLIN_LW, _SV_TICK_LEN = 1.0, 0.75, 3
_SV_HALF_W = 0.42
_SV_DOT_SIZE, _SV_DOT_EDGE_LW, _SV_DOT_DARKEN = 9, 0.5, 0.6
_SV_DOT_X = (0.155, 0.28)
_SV_BOX_DX, _SV_BOX_W = 0.10, 0.07
_SV_CAP_W = 0.05
_SV_YLABEL_PAD = 2
_SV_SUP_SCALE, _SV_SUP_RISE = 0.7, 0.45
_SV_ALPHA = 0.05
_SV_BAR_GAP, _SV_BAR_STEP, _SV_BAR_TEXT = 2.5, 8.0, 0.85


def make_split_violin_plot(cfg, d, st, out_base, *, conditions, times, comparison_key,
                            formats, dpi=300, stats_mode="mixed",
                            show_time_bars=True, show_cond_bars=True, show_time_legend=False,
                            default_dot_colors=None, default_bar_colors=None):
    """Split violin (left/right halves) with box plot, experiment dots and planned-comparison
    p-values. `comparison_key(row, time)` maps a row of the between-conditions table to a key in
    `bar_colors` (or `default_bar_colors`), used to color that comparison's line."""
    pcol = "s_p_holm" if stats_mode == "robust" else "p_holm"
    x_cm, y_cm = cfg.get("axes_cm", (4.5, 5.0))
    fs_axes, fs_p = cfg.get("font_pt", _SV_FS_AXES), cfg.get("font_pt", _SV_FS_PVAL)
    k = len(conditions)
    tcol = {t: np.array(cfg["time_colors"][t]) / 255 for t in times}
    bcol = {key: tuple(np.array(v) / 255) for key, v in cfg.get("bar_colors", default_bar_colors).items()}
    exp_means = d.groupby(["Cond", "Time", "Exp"]).V.mean().reset_index()
    exps = sorted(d.Exp.unique())
    dot_dx = dict(zip(exps, np.linspace(_SV_DOT_X[0], _SV_DOT_X[1], len(exps))))
    side = {times[0]: -1, times[1]: 1}

    CM = 1 / 2.54
    ML, MB, MT, MR = 1.6, 1.2, 1.6, 0.6
    W, H = ML + x_cm + MR, MB + y_cm + MT
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([ML / W, MB / H, x_cm / W, y_cm / H])
    pos = np.arange(1, k + 1)
    idx = dict(zip(conditions, pos))
    top_of = {}                                             # highest point of each violin

    for cond in conditions:
        x0 = idx[cond]
        top_of[cond] = d.loc[d.Cond == cond, "V"].max()
        for t in times:
            y = d.loc[(d.Cond == cond) & (d.Time == t), "V"].values
            if len(y) < 2:
                continue
            sd = side[t]
            grid = np.linspace(y.min(), y.max(), 400)
            kde = gaussian_kde(y)
            w = kde(grid) / kde(grid).max() * _SV_HALF_W
            ax.fill_betweenx(grid, x0, x0 + sd * w, facecolor=tuple(tcol[t]), edgecolor=tuple(tcol[t] * 0.8),
                             lw=_SV_VIOLIN_LW, joinstyle="round", zorder=2)
            # box plot (Tukey): Q1-Q3 box, median, whiskers up to 1.5*IQR
            q1, med, q3 = np.percentile(y, [25, 50, 75]); iqr = q3 - q1
            lo, hi = y[y >= q1 - 1.5 * iqr].min(), y[y <= q3 + 1.5 * iqr].max()
            bx = x0 + sd * _SV_BOX_DX
            ax.plot([bx, bx], [lo, hi], color="black", lw=_SV_VIOLIN_LW, zorder=3, solid_capstyle="butt")
            for yc in (lo, hi):                          # whisker caps
                ax.plot([bx - _SV_CAP_W / 2, bx + _SV_CAP_W / 2], [yc, yc], color="black", lw=_SV_VIOLIN_LW,
                        zorder=3, solid_capstyle="butt")
            ax.add_patch(plt.Rectangle((bx - _SV_BOX_W / 2, q1), _SV_BOX_W, q3 - q1, facecolor="white",
                                       edgecolor="black", lw=_SV_VIOLIN_LW, zorder=4))
            ax.plot([bx - _SV_BOX_W / 2, bx + _SV_BOX_W / 2], [med, med], color="black", lw=_SV_LW, zorder=4,
                    solid_capstyle="butt")
            # dots: mean of each experiment
            dots = cfg.get("dot_colors", default_dot_colors)
            dc = tuple(np.array(dots[t]) / 255) if t in dots else tuple(tcol[t] * _SV_DOT_DARKEN)
            for _, r in exp_means[(exp_means.Cond == cond) & (exp_means.Time == t)].iterrows():
                ax.scatter(x0 + sd * dot_dx[r.Exp], r.V, s=_SV_DOT_SIZE, facecolor=dc,
                           edgecolor="black", lw=_SV_DOT_EDGE_LW, zorder=5)

    # axes
    step = cfg.get("y_step", 1)
    ymax = cfg.get("y_max", float(np.ceil(d.V.max() / step) * step))
    if d.V.max() > ymax:
        print(f"  WARNING: values up to {d.V.max():.2f} exceed y_max = {ymax}")
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + step / 2, step))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.4, k + 0.6)
    ax.set_xticks(pos, conditions, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="both", labelsize=fs_axes, width=_SV_LW, length=_SV_TICK_LEN, pad=2)
    ax.tick_params(axis="x", length=0, pad=3)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(_SV_LW)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(cfg["ylabel"], fontsize=fs_axes, labelpad=cfg.get("ylabel_pad", _SV_YLABEL_PAD))

    # p-value
    to_data = ax.transData.inverted()
    pt = lambda v: v / (y_cm / 2.54 * 72) * ymax

    def draw_p(x, y, value, color):
        text, exp = value
        kw = dict(fontsize=fs_p, fontweight="normal", color=color)
        if text == "ns":
            t = ax.text(x, y, "ns", ha="center", va="bottom", clip_on=False, **kw)
            fig.canvas.draw()
            return Bbox(to_data.transform(t.get_window_extent())), [t]
        p_lbl = ax.text(x, y, "P", ha="left", va="bottom", fontstyle="italic", clip_on=False, **kw)
        parts = [p_lbl, ax.annotate(text, xy=(1, 0), xycoords=p_lbl, ha="left", va="bottom",
                                    annotation_clip=False, **kw)]
        if exp:
            parts.append(ax.annotate(exp, xy=(1, _SV_SUP_RISE), xycoords=parts[1], ha="left", va="bottom",
                                     annotation_clip=False, **{**kw, "fontsize": fs_p * _SV_SUP_SCALE}))
        fig.canvas.draw()
        x_left = to_data.transform(parts[0].get_window_extent().p0)[0]
        x_right = to_data.transform(parts[-1].get_window_extent().p1)[0]
        p_lbl.set_x(x - (x_right - x_left) / 2)
        fig.canvas.draw()
        return Bbox(to_data.transform(Bbox.union([p.get_window_extent() for p in parts]))), parts

    def bar(x1, x2, y, value, color):
        ax.plot([x1, x2], [y, y], color=color, lw=_SV_LW, clip_on=False, solid_capstyle="butt", zorder=6)
        return draw_p((x1 + x2) / 2, y + pt(_SV_BAR_TEXT), value, color)[0]

    # 1) second time point vs first within each condition: short black line above the violin
    occupied = []                                           # boxes (in data units) already occupied
    if show_time_bars:
        bt = st["between_time"].set_index("Cond")
        for cond in conditions:
            yb = top_of[cond] + pt(_SV_BAR_GAP)
            x1, x2 = idx[cond] - _SV_BOX_DX, idx[cond] + _SV_BOX_DX
            lbl = bar(x1, x2, yb, p_text(bt.loc[cond, pcol], _SV_ALPHA), bcol["time"])
            occupied.append(Bbox([[min(x1, lbl.x0), yb], [max(x2, lbl.x1), lbl.y1]]))

    # 2) comparisons between conditions, for each time point.
    #    Shorter lines sit lower; each line goes to the lowest free level.
    if show_cond_bars:
        base = max([b.y1 for b in occupied] + [max(top_of.values())]) + pt(_SV_BAR_GAP)
        bars = []
        for t in times:
            for _, r in st["between_cond"][t].iterrows():
                kind = comparison_key(r, t)
                x1, x2 = sorted((idx[r.A] + side[t] * _SV_BOX_DX, idx[r.B] + side[t] * _SV_BOX_DX))
                bars.append((x2 - x1, times.index(t), x1, x2, r[pcol], kind))
        bars.sort(key=lambda b: (round(b[0], 3), b[1]))
        levels = []                                         # per level: occupied x intervals
        for _, _, x1, x2, p, kind in bars:
            # label width: estimated by drawing it and then removing it
            probe, parts = draw_p((x1 + x2) / 2, 0, p_text(p, _SV_ALPHA), "none")
            for q in parts:
                q.remove()
            span = (min(x1, probe.x0), max(x2, probe.x1))
            lvl = 0
            while lvl < len(levels) and any(span[0] < b and span[1] > a for a, b in levels[lvl]):
                lvl += 1
            if lvl == len(levels):
                levels.append([])
            levels[lvl].append((span[0] - 0.03, span[1] + 0.03))
            bar(x1, x2, base + pt(_SV_BAR_STEP * lvl), p_text(p, _SV_ALPHA), bcol[kind])

    if show_time_legend:
        ax.legend(handles=[Patch(facecolor=tuple(tcol[t]), edgecolor=tuple(tcol[t] * 0.8), lw=_SV_VIOLIN_LW, label=t)
                           for t in times], loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
                  handlelength=1, handleheight=0.8, handletextpad=0.4, borderaxespad=0,
                  prop={"size": fs_p, "weight": "bold"})

    saved = save_plot(fig, out_base, formats, dpi, svg_postprocess=svg_mm_postprocess)
    plt.close(fig)
    return saved
