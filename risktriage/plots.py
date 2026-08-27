"""Figures for RiskTriage experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pred_scatter(y, pred, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(y, pred, s=18, alpha=0.75, c="#1f4e79")
    lims = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lims, lims, ls="--", c="0.5", lw=1)
    ax.set_xlabel("Experimental $V_{50}$ (V vs SHE)")
    ax.set_ylabel("Predicted $V_{50}$")
    ax.set_title(title)
    _save(fig, path)


def plot_frontier(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for method, g in df.groupby("method"):
        g = g.sort_values("triage_risk")
        ax.plot(g["triage_risk"], g["p_test"], marker="o", ms=4, label=method)
    ax.set_xlabel("Decision risk (triage loss)")
    ax.set_ylabel("Physical test fraction")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Risk vs experiments")
    _save(fig, path)


def plot_discovery(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for method, g in df.groupby("method"):
        g = g.sort_values("B")
        ax.plot(g["B"], g["recall"], marker="o", ms=3, label=method)
    ax.set_xlabel("Physical experiments $B$")
    ax.set_ylabel("Recall of top-decile catalysts")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Discovery vs experimental budget")
    _save(fig, path)


def plot_cost_sensitivity(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method, g in df.groupby("method"):
        g = g.sort_values("c_e")
        ax.plot(g["c_e"], g["p_test"], marker="o", ms=4, label=method)
    ax.axvline(0.5, ls="--", c="0.6", lw=1, label="harmonic cap $c_E=1/2$")
    ax.set_xlabel("Experimental cost $c_E$")
    ax.set_ylabel("Physical test fraction")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Cost-aware action mix")
    _save(fig, path)


def plot_matched_risk(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    methods = [m for m in df["method"].unique()]
    rs = sorted(df["r"].unique())
    x = np.arange(len(rs))
    width = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        ys = []
        for r in rs:
            row = df[(df.method == m) & (df.r == r)]
            ys.append(float(row["T_mean"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + i * width, ys, width, label=m)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels([str(r) for r in rs])
    ax.set_xlabel("Max allowed decision risk $r$")
    ax.set_ylabel(r"$T(r)=\min P(\mathrm{TEST})$")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Matched-risk experimental efficiency")
    _save(fig, path)


def plot_ood_test_rate(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    methods = df["method"].unique()
    x = np.arange(len(methods))
    iid = [df[(df.method == m) & (df.split == "iid")]["p_test"].mean() for m in methods]
    ood = [df[(df.method == m) & (df.split == "ood")]["p_test"].mean() for m in methods]
    ax.bar(x - 0.18, iid, 0.35, label="IID")
    ax.bar(x + 0.18, ood, 0.35, label="OOD")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("$P$(TEST)")
    ax.legend(frameon=False)
    ax.set_title("Calibration conservatism under composition shift")
    _save(fig, path)
