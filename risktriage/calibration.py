"""Split conformal, conformal risk control, and RAC-style downstream calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risktriage.theory import (
    Costs,
    bayes_action_from_p,
    gaussian_success_prob,
    interval_action,
    robust_action,
    triage_loss,
)


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    q_level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


@dataclass
class IntervalResult:
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    lambda_: float


def residual_intervals(pred: np.ndarray, scale: np.ndarray, lam: float) -> IntervalResult:
    pred = np.asarray(pred, dtype=float)
    scale = np.clip(np.asarray(scale, dtype=float), 1e-6, None)
    half = lam * scale
    lo, hi = pred - half, pred + half
    return IntervalResult(lo, hi, 2.0 * half, float(lam))


def fit_split_conformal(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    scale_cal: np.ndarray,
    alpha: float = 0.1,
) -> float:
    scores = np.abs(np.asarray(y_cal) - np.asarray(pred_cal)) / np.clip(scale_cal, 1e-6, None)
    return split_conformal_quantile(scores, alpha)


def _lambda_grid(pred: np.ndarray, y: np.ndarray, scale: np.ndarray, n: int = 80) -> np.ndarray:
    scores = np.abs(np.asarray(y) - np.asarray(pred)) / np.clip(scale, 1e-6, None)
    hi = float(np.quantile(scores, 0.995)) if len(scores) else 1.0
    hi = max(hi, 1e-3)
    return np.concatenate([[0.0], np.geomspace(1e-3, hi * 1.5, n)])


def crc_lambda(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    scale_cal: np.ndarray,
    y_star: float,
    costs: Costs,
    r_star: float,
    use_robust: bool = False,
) -> float:
    """Smallest λ such that the finite-sample CRC bound on triage risk is ≤ r*."""
    n = len(y_cal)
    b = max(costs.c_fp, costs.c_fn)
    success = (np.asarray(y_cal) >= y_star).astype(int)
    best = None
    for lam in _lambda_grid(pred_cal, y_cal, scale_cal):
        iv = residual_intervals(pred_cal, scale_cal, lam)
        if use_robust:
            a = robust_action(iv.lower, iv.upper, y_star, costs)
        else:
            a = interval_action(iv.lower, iv.upper, y_star)
        rhat = float(np.mean(triage_loss(a, success, costs)))
        bound = (n / (n + 1.0)) * rhat + b / (n + 1.0)
        if bound <= r_star:
            best = lam
            break
    if best is None:
        return float(_lambda_grid(pred_cal, y_cal, scale_cal)[-1])
    return float(best)


def crc_lambda_bayes(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    scale_cal: np.ndarray,
    y_star: float,
    costs: Costs,
    r_star: float,
) -> float:
    """Smallest λ such that Bayes actions on N(pred, (λ σ)^2) control triage risk."""
    n = len(y_cal)
    b = max(costs.c_fp, costs.c_fn)
    success = (np.asarray(y_cal) >= y_star).astype(int)
    best = None
    for lam in _lambda_grid(pred_cal, y_cal, scale_cal):
        p = gaussian_success_prob(pred_cal, np.clip(scale_cal, 1e-6, None) * max(lam, 1e-6), y_star)
        a = bayes_action_from_p(p, costs)
        rhat = float(np.mean(triage_loss(a, success, costs)))
        bound = (n / (n + 1.0)) * rhat + b / (n + 1.0)
        if bound <= r_star:
            best = lam
            break
    if best is None:
        return float(_lambda_grid(pred_cal, y_cal, scale_cal)[-1])
    return float(best)


def rac_lambda(
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    scale_cal: np.ndarray,
    y_star: float,
    costs: Costs,
    r_star: float,
) -> float:
    """Calibrate nested max-min sets to control downstream triage risk.

    For the three-action catalyst utility, the robust action on an interval
    coincides with Theorem 1, so RAC reduces to CRC on the same nested family.
    """
    return crc_lambda(pred_cal, y_cal, scale_cal, y_star, costs, r_star, use_robust=True)


def decision_metrics(action: np.ndarray, success: np.ndarray, y: np.ndarray | None = None) -> dict:
    a = np.asarray(action, dtype=int)
    g = np.asarray(success, dtype=int)
    n = len(a)
    trust = a == 2
    drop = a == 0
    test = a == 1
    out = {
        "n": n,
        "p_trust": float(trust.mean()) if n else 0.0,
        "p_drop": float(drop.mean()) if n else 0.0,
        "p_test": float(test.mean()) if n else 0.0,
        "false_trust": float((g[trust] == 0).mean()) if trust.any() else np.nan,
        "false_drop": float((g[drop] == 1).mean()) if drop.any() else np.nan,
        "wrong_nontest": float((((a == 2) & (g == 0)) | ((a == 0) & (g == 1))).mean()) if n else 0.0,
    }
    return out
