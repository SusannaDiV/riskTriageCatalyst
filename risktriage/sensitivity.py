"""Threshold, predictor-strength, and paired method comparisons."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from risktriage.experiments import (
    RISK_LEVELS,
    _dense_frontier_points,
    _one_split_fit,
    _y_star,
)
from risktriage.data import TriageTask, load_her_task
from risktriage.splits import grouped_split
from risktriage.stats import bootstrap_mean_ci, min_test_at_risk, paired_bootstrap_delta, wilcoxon_paired
from risktriage.theory import Costs

METHODS = ["crc", "rac", "risktriage", "selective", "voi", "uncertainty", "split_cp"]
COMPETITORS = ["selective", "voi", "uncertainty", "split_cp"]
QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
STRENGTH = (("strong", -0.25), ("moderate", 0.0), ("weak", 0.65))

OPERATIONAL = {
    "point": "Always TRUST/DROP by ŷ vs y*; never tests. No risk control.",
    "selective": "Chow reject option: TEST iff t < p̂ < 1−t. Abstention is not calibrated to a risk budget.",
    "voi": "TEST the highest value-of-information candidates; remainder point-classified. Ranking, not risk control.",
    "uncertainty": "TEST highest predictive variance; remainder point-classified. Variance ≠ decision value.",
    "split_cp": "TEST iff a (1−α) interval straddles y*. Controls coverage, not experimental burden.",
    "crc": "Smallest nested interval width whose CRC bound on triage risk is ≤ r*. Risk-budget regime.",
    "rac": "Same nested family as CRC for this utility (robust max-min ≡ interval action).",
    "risktriage": "CRC/RAC risk-budget OR Bayes thresholds on p̂ that explicitly include c_E. Adds the TEST action as a costed physical experiment.",
}


def frontier_T_by_seed(frontier: pd.DataFrame, r: float = 0.10) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        sub = frontier[frontier.method == method]
        if sub.empty:
            continue
        for seed, g in sub.groupby("seed"):
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "r": r,
                    "T": min_test_at_risk(g["p_test"].to_numpy(), g["triage_risk"].to_numpy(), r),
                }
            )
    return pd.DataFrame(rows)


def paired_against_crc(t_seed: pd.DataFrame, r: float = 0.10) -> pd.DataFrame:
    """Δ = T_competitor − T_CRC; positive means CRC tests less."""
    rows = []
    at = t_seed[t_seed.r == r] if "r" in t_seed.columns else t_seed
    ref = at[at.method.isin(["crc", "rac"])]
    if ref.empty:
        return pd.DataFrame()
    ref_m = "crc" if (ref.method == "crc").any() else str(ref.method.iloc[0])
    pivot = at.pivot_table(index="seed", columns="method", values="T", aggfunc="mean")
    if ref_m not in pivot.columns:
        return pd.DataFrame()
    for m in [c for c in pivot.columns if c != ref_m]:
        a = pivot[m].to_numpy()
        b = pivot[ref_m].to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            continue
        boot = paired_bootstrap_delta(a[mask], b[mask], n_boot=2000, seed=0)
        wx = wilcoxon_paired(a[mask], b[mask])
        rows.append(
            {
                "r": r,
                "competitor": m,
                "reference": ref_m,
                "n": int(mask.sum()),
                "delta_mean": boot["mean"],
                "delta_ci_lo": boot["ci_lo"],
                "delta_ci_hi": boot["ci_hi"],
                "pr_competitor_higher_T": boot["p_gt0"],
                "wilcoxon_p": wx["pvalue"],
            }
        )
    return pd.DataFrame(rows)


def paired_from_by_seed_csv(path: Path, r: float = 0.10) -> pd.DataFrame:
    df = pd.read_csv(path)
    return paired_against_crc(df, r=r)


def _apply_mix(pred: np.ndarray, std: np.ndarray, scale: np.ndarray, mix: float):
    mu = float(np.mean(pred))
    pred_m = (1.0 - mix) * pred + mix * mu
    gain = 1.0 + max(mix, 0.0) * 2.0
    if mix < 0:
        # Pull toward a less noisy residual without using labels: shrink std.
        gain = 1.0 + mix
    return pred_m, np.clip(std * gain, 1e-4, None), np.clip(scale * max(gain, 0.25), 1e-6, None)


def experiment_wiley(
    task: TriageTask | None = None,
    out: Path | None = None,
    n_seeds: int | None = None,
    n_members: int = 5,
    smoke: bool = False,
) -> dict:
    task = task or load_her_task()
    out = Path(out or "results/wiley")
    out.mkdir(parents=True, exist_ok=True)
    n_seeds = n_seeds if n_seeds is not None else (3 if smoke else 20)
    n_members = 3 if smoke else n_members
    costs = Costs()
    qs = (0.50, 0.75, 0.90) if smoke else QUANTILES
    mixes = (("moderate", 0.0), ("weak", 0.65)) if smoke else STRENGTH
    dense = not smoke

    q_rows, s_rows = [], []
    for seed in range(n_seeds):
        split = grouped_split(task.groups, seed=seed)
        fitted, pred_ca, y_ca, scale_ca, *_ = _one_split_fit(task, split, n_members, seed, costs, 0.75)
        tr = split.train
        for q in qs:
            ys = _y_star(task.y[tr], q)
            fq = replace(fitted, y_star=ys, success=(fitted.y >= ys).astype(int))
            pts = _dense_frontier_points(fq, pred_ca, y_ca, scale_ca, seed, costs, dense)
            for ev in pts:
                ev["quantile"] = float(q)
                ev["success_rate"] = float(fq.success.mean())
            q_rows.extend(pts)
        for name, mix in mixes:
            pred_m, std_m, scale_m = _apply_mix(fitted.pred_mean, fitted.pred_std, scale_ca, mix)
            fs = replace(fitted, pred_mean=pred_m, pred_std=std_m)
            pred_ca_m, _, scale_ca_m = _apply_mix(pred_ca, np.ones_like(pred_ca), scale_ca, mix)
            pts = _dense_frontier_points(fs, pred_ca_m, y_ca, scale_ca_m, seed, costs, dense)
            for ev in pts:
                ev["regime"] = name
                ev["mix"] = float(mix)
            s_rows.extend(pts)

    qf = pd.DataFrame(q_rows)
    sf = pd.DataFrame(s_rows)
    qf.to_csv(out / "quantile_frontier.csv", index=False)
    sf.to_csv(out / "strength_frontier.csv", index=False)

    tq, ts = [], []
    for q, g in qf.groupby("quantile"):
        t = frontier_T_by_seed(g, r=0.10)
        t["quantile"] = float(q)
        tq.append(t)
    for regime, g in sf.groupby("regime"):
        t = frontier_T_by_seed(g, r=0.10)
        t["regime"] = str(regime)
        ts.append(t)
    tq_df = pd.concat(tq, ignore_index=True) if tq else pd.DataFrame()
    ts_df = pd.concat(ts, ignore_index=True) if ts else pd.DataFrame()
    tq_df.to_csv(out / "quantile_T_by_seed.csv", index=False)
    ts_df.to_csv(out / "strength_T_by_seed.csv", index=False)

    def _summ(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        rows = []
        if df.empty:
            return pd.DataFrame()
        for vals, g in df.groupby(keys + ["method"]):
            ci = bootstrap_mean_ci(g["T"].to_numpy(), n_boot=2000, seed=0)
            rec = {k: v for k, v in zip(keys + ["method"], vals)}
            rec.update({"T_mean": ci["mean"], "T_ci_lo": ci["ci_lo"], "T_ci_hi": ci["ci_hi"], "n": ci["n"]})
            rows.append(rec)
        return pd.DataFrame(rows)

    q_tab = _summ(tq_df, ["quantile"])
    s_tab = _summ(ts_df, ["regime"])
    q_tab.to_csv(out / "quantile_T_table.csv", index=False)
    s_tab.to_csv(out / "strength_T_table.csv", index=False)

    pair_q = []
    for q, g in tq_df.groupby("quantile"):
        p = paired_against_crc(g, r=0.10)
        p["quantile"] = float(q)
        pair_q.append(p)
    pair_s = []
    for regime, g in ts_df.groupby("regime"):
        p = paired_against_crc(g, r=0.10)
        p["regime"] = str(regime)
        pair_s.append(p)
    pq = pd.concat(pair_q, ignore_index=True) if pair_q else pd.DataFrame()
    ps = pd.concat(pair_s, ignore_index=True) if pair_s else pd.DataFrame()
    pq.to_csv(out / "quantile_paired.csv", index=False)
    ps.to_csv(out / "strength_paired.csv", index=False)

    extra = {}
    for label, pth in (
        ("co2r", Path("results/co2r/matched_risk_by_seed.csv")),
        ("enthalpy", Path("results/enthalpy/matched_risk_by_seed.csv")),
    ):
        if pth.exists():
            extra[label] = paired_from_by_seed_csv(pth, r=0.10).to_dict(orient="records")
            extra[f"{label}_r03"] = paired_from_by_seed_csv(pth, r=0.03).to_dict(orient="records")

    (out / "operational_comparison.json").write_text(json.dumps(OPERATIONAL, indent=2))
    summary = {
        "n_seeds": n_seeds,
        "quantiles": list(qs),
        "headline_r": 0.10,
        "extra_task_paired_vs_crc": extra,
    }
    (out / "wiley_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "DONE.txt").write_text("wiley sensitivity complete\n")
    return summary
