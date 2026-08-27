"""Train/calibrate a split and emit decision policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

from risktriage.calibration import (
    IntervalResult,
    crc_lambda,
    decision_metrics,
    fit_split_conformal,
    rac_lambda,
    residual_intervals,
)
from risktriage.models import LGBMEnsemble, make_model
from risktriage.theory import (
    Costs,
    bayes_action_from_p,
    gaussian_success_prob,
    interval_action,
    loss_matrix,
    robust_action,
    triage_loss,
    voi_perfect_experiment,
)


def spearman(y, p) -> float:
    from scipy.stats import spearmanr

    if len(y) < 3:
        return float("nan")
    r = spearmanr(y, p, nan_policy="omit")
    return float(r.correlation)


@dataclass
class FittedSplit:
    pred_mean: np.ndarray
    pred_std: np.ndarray
    y: np.ndarray
    success: np.ndarray
    y_star: float
    costs: Costs
    residual_scale: float
    X: np.ndarray
    idx: np.ndarray
    sabatier: np.ndarray
    matched: np.ndarray


def fit_ensemble(X_train, y_train, X_eval, n_members: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ens = LGBMEnsemble(n_members=n_members, seed=seed).fit(X_train, y_train)
    dist = ens.predict_dist(X_eval)
    return dist.mean, np.maximum(dist.std, 1e-4)


def residual_mad(pred, y) -> float:
    return float(max(np.median(np.abs(np.asarray(y) - np.asarray(pred))), 1e-4))


def scaled_sigma(ens_std: np.ndarray, residual_scale: float) -> np.ndarray:
    return np.sqrt(np.clip(ens_std, 1e-6, None) ** 2 + residual_scale**2)


def prediction_metrics(y, pred) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "r2": float(r2_score(y, pred)) if len(y) > 1 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "spearman": spearman(y, pred),
        "n": int(len(y)),
    }


def coverage_of(iv: IntervalResult, y: np.ndarray) -> float:
    return float(np.mean((y >= iv.lower) & (y <= iv.upper)))


def policies_for_split(
    fitted: FittedSplit,
    pred_cal: np.ndarray,
    y_cal: np.ndarray,
    scale_cal: np.ndarray,
    alpha: float,
    r_star: float,
) -> dict[str, np.ndarray]:
    y_star, costs = fitted.y_star, fitted.costs
    scale_te = scaled_sigma(fitted.pred_std, fitted.residual_scale)
    lam_cp = fit_split_conformal(pred_cal, y_cal, scale_cal, alpha=alpha)
    iv_cp = residual_intervals(fitted.pred_mean, scale_te, lam_cp)
    lam_crc = crc_lambda(pred_cal, y_cal, scale_cal, y_star, costs, r_star, use_robust=False)
    iv_crc = residual_intervals(fitted.pred_mean, scale_te, lam_crc)
    lam_rac = rac_lambda(pred_cal, y_cal, scale_cal, y_star, costs, r_star)
    iv_rac = residual_intervals(fitted.pred_mean, scale_te, lam_rac)

    p = gaussian_success_prob(fitted.pred_mean, scale_te, y_star)
    uncal = bayes_action_from_p(p, costs)
    # uncalibrated point predictor: TRUST if pred >= y*, else DROP (no TEST)
    point = np.where(fitted.pred_mean >= y_star, 2, 0)

    return {
        "uncalibrated": uncal,
        "point": point,
        "split_cp": interval_action(iv_cp.lower, iv_cp.upper, y_star),
        "crc": interval_action(iv_crc.lower, iv_crc.upper, y_star),
        "rac": robust_action(iv_rac.lower, iv_rac.upper, y_star, costs),
        "_iv_cp": iv_cp,
        "_iv_crc": iv_crc,
        "_iv_rac": iv_rac,
        "_p": p,
        "_lam": {"cp": lam_cp, "crc": lam_crc, "rac": lam_rac},
        "_voi": voi_perfect_experiment(p, costs),
    }


def evaluate_actions(name: str, action: np.ndarray, fitted: FittedSplit) -> dict:
    m = decision_metrics(action, fitted.success)
    m["method"] = name
    m["triage_risk"] = float(np.mean(triage_loss(action, fitted.success, fitted.costs)))
    m["full_risk"] = float(np.mean(loss_matrix(action, fitted.success, fitted.costs)))
    return m
