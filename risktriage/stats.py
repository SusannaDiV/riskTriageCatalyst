"""Paired bootstrap for discovery curves and AUDC."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = len(a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = float(np.mean(a[idx] - b[idx]))
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "mean": float(np.mean(deltas)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_gt0": float(np.mean(deltas > 0)),
    }


def min_test_at_risk(p_test: np.ndarray, risk: np.ndarray, r: float) -> float:
    """T(r) = min { P(TEST) : R <= r } on a realized frontier."""
    p_test = np.asarray(p_test, dtype=float)
    risk = np.asarray(risk, dtype=float)
    ok = np.isfinite(p_test) & np.isfinite(risk) & (risk <= r + 1e-12)
    if not np.any(ok):
        return float("nan")
    return float(np.min(p_test[ok]))


def min_risk_at_test_budget(p_test: np.ndarray, risk: np.ndarray, t: float) -> float:
    """R(t) = min { R : P(TEST) <= t }."""
    p_test = np.asarray(p_test, dtype=float)
    risk = np.asarray(risk, dtype=float)
    ok = np.isfinite(p_test) & np.isfinite(risk) & (p_test <= t + 1e-12)
    if not np.any(ok):
        return float("nan")
    return float(np.min(risk[ok]))


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank on paired differences a-b (NaNs dropped)."""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    d = d[np.abs(d) > 1e-15]
    n = int(len(d))
    if n < 6:
        return {"n": n, "stat": float("nan"), "pvalue": float("nan")}
    res = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"n": n, "stat": float(res.statistic), "pvalue": float(res.pvalue)}


def bootstrap_mean_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n": 0}
    n = len(x)
    means = np.array([float(np.mean(x[rng.integers(0, n, n)])) for _ in range(n_boot)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(np.mean(x)), "ci_lo": float(lo), "ci_hi": float(hi), "n": int(n)}
