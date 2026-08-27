"""Main two-panel figure: workflow schematic + risk/testing Pareto frontier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "risktriage"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = {
    "crc": ("RAC / CRC", "#1f4e79", 2.8, "-"),
    "risktriage": ("RiskTriage-Bayes", "#2a9d8f", 2.1, "--"),
    "uncertainty": ("Uncertainty sampling", "#e07a3d", 2.3, "-"),
    "split_cp": ("Split conformal", "#6b7280", 2.1, "-."),
}

R_STAR = 0.10


def t_at_r_from_dense(dense: pd.DataFrame, method: str, r_grid: np.ndarray) -> np.ndarray:
    rows = []
    for seed, g in dense[dense.method == method].groupby("seed"):
        pt = g["p_test"].to_numpy()
        rk = g["triage_risk"].to_numpy()
        tvals = []
        for r in r_grid:
            ok = rk <= r + 1e-12
            tvals.append(float(np.min(pt[ok])) if np.any(ok) else np.nan)
        rows.append(tvals)
    return np.asarray(rows, dtype=float)


def _roundbox(ax, x, y, w, h, fc, ec="#1f2937", lw=1.25, rs=0.12, z=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rs}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, x1, y1, x2, y2, lw=1.5, ms=13, color="#374151"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=ms),
        zorder=3,
    )


ICON_S = 360


def _badge(ax, cx, cy, fc, ec, s=ICON_S):
    """Round badge in display space so it stays circular."""
    ax.scatter([cx], [cy], s=s, c=fc, edgecolors=ec, linewidths=1.2, zorder=5)


def icon_molecule(ax, cx, cy):
    _badge(ax, cx, cy, "#dbeafe", "#1d4ed8")
    d, h = 0.16, 0.20
    pts = [(cx, cy + h), (cx - d, cy - 0.08), (cx + d, cy - 0.08)]
    xs, ys = zip(*pts)
    ax.plot([pts[0][0], pts[1][0], pts[2][0], pts[0][0]],
            [pts[0][1], pts[1][1], pts[2][1], pts[0][1]],
            color="#1d4ed8", lw=1.35, zorder=6)
    ax.scatter(xs, ys, s=28, c="#1d4ed8", zorder=7, edgecolors="white", linewidths=0.4)


def icon_model(ax, cx, cy):
    _badge(ax, cx, cy, "#dbeafe", "#1d4ed8")
    bars = [(-0.16, 0.10), (0.0, 0.22), (0.16, 0.16)]
    for dx, hh in bars:
        ax.add_patch(Rectangle((cx + dx - 0.045, cy - 0.16), 0.09, hh,
                               facecolor="#1d4ed8", edgecolor="none", zorder=6))


def icon_funnel(ax, cx, cy):
    _badge(ax, cx, cy, "#dcfce7", "#166534")
    tri = Polygon(
        [(cx - 0.16, cy + 0.12), (cx + 0.16, cy + 0.12), (cx + 0.05, cy - 0.02), (cx - 0.05, cy - 0.02)],
        facecolor="#166534", edgecolor="none", zorder=6,
    )
    ax.add_patch(tri)
    ax.add_patch(Rectangle((cx - 0.03, cy - 0.14), 0.06, 0.12,
                           facecolor="#166534", edgecolor="none", zorder=6))


def icon_drop(ax, cx, cy):
    _badge(ax, cx, cy, "#fecaca", "#991b1b")
    ax.plot([cx - 0.09, cx + 0.09], [cy - 0.09, cy + 0.09], color="#991b1b", lw=2.0, zorder=6, solid_capstyle="round")
    ax.plot([cx - 0.09, cx + 0.09], [cy + 0.09, cy - 0.09], color="#991b1b", lw=2.0, zorder=6, solid_capstyle="round")


def icon_flask(ax, cx, cy):
    _badge(ax, cx, cy, "#fde68a", "#b45309")
    ax.add_patch(Rectangle((cx - 0.035, cy + 0.02), 0.07, 0.14,
                           facecolor="#b45309", edgecolor="none", zorder=6))
    body = Polygon(
        [(cx - 0.04, cy + 0.04), (cx + 0.04, cy + 0.04), (cx + 0.14, cy - 0.14), (cx - 0.14, cy - 0.14)],
        facecolor="#b45309", edgecolor="none", zorder=6,
    )
    ax.add_patch(body)


def icon_check(ax, cx, cy):
    _badge(ax, cx, cy, "#bbf7d0", "#166534")
    ax.plot([cx - 0.10, cx - 0.02, cx + 0.12], [cy + 0.01, cy - 0.10, cy + 0.12],
            color="#166534", lw=2.0, zorder=6, solid_capstyle="round")


def icon_lab(ax, cx, cy):
    _badge(ax, cx, cy, "#ffedd5", "#c2410c")
    # beaker
    ax.plot(
        [cx - 0.15, cx - 0.11, cx + 0.11, cx + 0.15],
        [cy + 0.14, cy - 0.14, cy - 0.14, cy + 0.14],
        color="#c2410c", lw=1.7, zorder=6, solid_capstyle="round",
    )
    ax.plot([cx - 0.09, cx + 0.09], [cy - 0.02, cy - 0.02], color="#ea580c", lw=2.0, zorder=6)
    # stir / bubble
    ax.scatter([cx - 0.03, cx + 0.04], [cy + 0.04, cy + 0.08], s=10, c="#c2410c", zorder=7)


def draw_workflow(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(2.35, 12.15)
    ax.axis("off")
    ax.set_title("(a)  Decision layer", loc="left", fontsize=12, fontweight="bold", pad=10)

    def row(y, h, text, fc, ec, icon, fs=8.6):
        _roundbox(ax, 0.40, y, 9.20, h, fc, ec=ec, lw=1.2, rs=0.14)
        icon(ax, 1.18, y + h / 2)
        ax.text(1.95, y + h / 2, text, ha="left", va="center",
                fontsize=fs, color="#111827", zorder=4, linespacing=1.25)

    row_h = 1.18
    rt_h = 4.13
    gap = 0.40
    y4 = 2.63
    y3 = y4 + row_h + gap
    y2 = y3 + rt_h + gap
    y1 = y2 + row_h + gap

    row(y1, row_h, "Computational candidate  $x$\nadsorption energies  +  composition",
        "#e8eef5", "#1e3a5f", icon_molecule)
    _arrow(ax, 5.0, y1, 5.0, y2 + row_h, lw=1.3, ms=11)
    row(y2, row_h, r"Predictor  $\hat{y}(x),\ \ p(x)=\Pr(G{=}1\mid x)$" + "\nLightGBM   $R^2 = 0.61$",
        "#dbeafe", "#1d4ed8", icon_model)
    _arrow(ax, 5.0, y2, 5.0, y3 + rt_h, lw=1.3, ms=11)

    _roundbox(ax, 0.25, y3, 9.50, rt_h, "#f0fdf4", ec="#166534", lw=1.7, rs=0.16)
    icon_funnel(ax, 1.18, y3 + rt_h - 0.46)
    ax.text(1.95, y3 + rt_h - 0.46, "RiskTriage", ha="left", va="center",
            fontsize=12.5, fontweight="bold", color="#14532d", zorder=4)

    def chip(x, y, w, h, title, value):
        _roundbox(ax, x, y, w, h, "#ffffff", ec="#86efac", lw=1.15, rs=0.10, z=3)
        ax.text(x + w / 2, y + 0.68 * h, title, ha="center", va="center",
                fontsize=7.4, color="#3f6212", zorder=5)
        ax.text(x + w / 2, y + 0.32 * h, value, ha="center", va="center",
                fontsize=10.2, fontweight="bold", color="#14532d", zorder=5)

    chips_y = y3 + 2.23
    chip(0.55, chips_y, 4.05, 0.88, "risk budget", r"$r^{\star}$")
    ax.plot([5.00, 5.00], [chips_y + 0.08, chips_y + 0.80], color="#166534", lw=2.0,
            solid_capstyle="round", zorder=5)
    chip(5.40, chips_y, 4.05, 0.88, "experiment cost", r"$c_E$")

    actions = [
        (0.50, "#fee2e2", "#991b1b", icon_drop, "low $p$", "DROP"),
        (3.58, "#fef3c7", "#92400e", icon_flask, "ambiguous", "TEST"),
        (6.66, "#dcfce7", "#166534", icon_check, "high $p$", "TRUST"),
    ]
    ay = y3 + 0.20
    for x, fc, ec, icon, sub, lab in actions:
        _roundbox(ax, x, ay, 2.84, 1.62, fc, ec=ec, lw=1.3, rs=0.12)
        icon(ax, x + 0.48, ay + 0.81)
        ax.text(x + 1.42, ay + 1.08, sub, ha="left", va="center",
                fontsize=7.0, fontweight="normal", color="#4b5563", zorder=4)
        ax.text(x + 1.42, ay + 0.62, lab, ha="left", va="center",
                fontsize=9.4, fontweight="bold", color="#111827", zorder=4)

    _arrow(ax, 5.0, y3, 5.0, y4 + row_h, lw=1.3, ms=11)
    row(y4, row_h, "Physical experiment\nsynthesis   ·   XRF/XRD   ·   electrolysis",
        "#fff7ed", "#c2410c", icon_lab)


def draw_frontier(ax, dense: pd.DataFrame):
    r_grid = np.linspace(0.03, 0.125, 40)
    order = ["split_cp", "uncertainty", "risktriage", "crc"]
    for method in order:
        name, color, lw, ls = METHODS[method]
        Ts = t_at_r_from_dense(dense, method, r_grid)
        mean = np.nanmean(Ts, axis=0) * 100
        lo = np.nanpercentile(Ts, 2.5, axis=0) * 100
        hi = np.nanpercentile(Ts, 97.5, axis=0) * 100
        ax.fill_betweenx(r_grid, lo, hi, color=color, alpha=0.14, linewidth=0)
        ax.plot(mean, r_grid, color=color, lw=lw, ls=ls, label=name, zorder=3)

    ax.axhline(R_STAR, color="#9ca3af", ls=":", lw=1.15, zorder=1)
    ax.text(78.5, R_STAR + 0.0028, r"$R=0.10$", fontsize=9.0, color="#6b7280", va="bottom", ha="right")

    pts = [
        (8.4, "crc", "o", 78),
        (8.9, "risktriage", "^", 70),
        (16.5, "uncertainty", "s", 62),
        (25.9, "split_cp", "D", 56),
    ]
    for x, m, mk, sz in pts:
        ax.scatter([x], [R_STAR], s=sz, marker=mk, color=METHODS[m][1],
                   zorder=5, edgecolors="white", linewidths=0.7)

    ax.annotate("8.4%", xy=(8.4, R_STAR), xytext=(0.6, 0.078),
                fontsize=9.2, fontweight="bold", color=METHODS["crc"][1],
                arrowprops=dict(arrowstyle="-", color=METHODS["crc"][1], lw=0.8))
    ax.annotate("8.9%", xy=(8.9, R_STAR), xytext=(2.6, 0.038),
                fontsize=9.2, fontweight="bold", color=METHODS["risktriage"][1],
                arrowprops=dict(arrowstyle="-", color=METHODS["risktriage"][1], lw=0.8))
    ax.annotate("16.5%", xy=(16.5, R_STAR), xytext=(20.5, 0.118),
                fontsize=9.2, fontweight="bold", color=METHODS["uncertainty"][1],
                arrowprops=dict(arrowstyle="-", color=METHODS["uncertainty"][1], lw=0.8))
    ax.annotate("25.9%", xy=(25.9, R_STAR), xytext=(34.0, 0.084),
                fontsize=9.2, fontweight="bold", color=METHODS["split_cp"][1],
                arrowprops=dict(arrowstyle="-", color=METHODS["split_cp"][1], lw=0.8))

    ax.set_xlim(0, 82)
    ax.set_ylim(0.025, 0.145)
    ax.set_xlabel(r"Physical tests   $P(\mathrm{TEST})$   (%)", fontsize=11)
    ax.set_ylabel("Decision risk   $R$", fontsize=11)
    ax.set_title("(b)  Experiments at matched scientific risk", loc="left",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8.8, loc="upper right", handlelength=2.4, borderaxespad=0.2)
    ax.tick_params(labelsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    dense = pd.read_csv(DATA / "exp3_dense_frontier.csv")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "mathtext.default": "regular",
    })
    fig = plt.figure(figsize=(13.2, 7.2), dpi=240)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=0.16,
                          left=0.03, right=0.985, top=0.92, bottom=0.10)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    draw_workflow(ax0)
    draw_frontier(ax1, dense)
    fig.savefig(OUT / "fig1_risktriage.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / "fig1_risktriage.png", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(DATA / "fig1_risktriage.png", bbox_inches="tight", pad_inches=0.08)
    print("wrote", OUT / "fig1_risktriage.png")


if __name__ == "__main__":
    main()
