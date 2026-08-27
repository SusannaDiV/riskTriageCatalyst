"""Retrospective experimental-budget replay on held-out OCx24 HER labels."""

from __future__ import annotations

import numpy as np


def topk_valuable(y: np.ndarray, frac: float = 0.1) -> np.ndarray:
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    order = np.argsort(-y)
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def ranking_metrics(y: np.ndarray, order: np.ndarray, budgets: np.ndarray, valuable: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    order = np.asarray(order, dtype=int)
    n_val = int(valuable.sum())
    y_star = float(np.max(y))
    rows = []
    for b in budgets:
        b = int(min(int(b), len(order)))
        chosen = order[:b]
        rec = float(valuable[chosen].sum() / max(n_val, 1))
        best = float(np.max(y[chosen])) if b else -np.inf
        regret = y_star - best
        rows.append({"B": b, "recall": rec, "best": best, "regret": regret})
    audc = float(np.mean([r["recall"] for r in rows])) if rows else 0.0
    # experiments to recover 80% of valuable catalysts
    ett = np.nan
    for r in rows:
        if r["recall"] >= 0.8:
            ett = float(r["B"])
            break
    return {"curve": rows, "audc": audc, "ett80": ett}


def policy_order(
    name: str,
    pred: np.ndarray,
    std: np.ndarray,
    sabatier: np.ndarray,
    p_success: np.ndarray,
    voi: np.ndarray,
    y_star: float,
    rng: np.random.Generator,
    interval_width: np.ndarray | None = None,
    rac_test: np.ndarray | None = None,
    rac_action: np.ndarray | None = None,
) -> np.ndarray:
    n = len(pred)
    if name == "random":
        return rng.permutation(n)
    if name == "sabatier":
        score = sabatier
    elif name == "ml_rank":
        score = pred
    elif name == "variance":
        score = std
    elif name == "ucb":
        score = pred + 1.0 * std
    elif name == "ei":
        # expected improvement toward high Y (less negative SHE voltage)
        z = (pred - y_star) / np.clip(std, 1e-6, None)
        from scipy.stats import norm

        score = (pred - y_star) * norm.cdf(z) + std * norm.pdf(z)
    elif name == "voi":
        score = voi
    elif name == "conformal_width":
        score = interval_width if interval_width is not None else std
    elif name == "risktriage":
        if rac_action is not None:
            # TRUST (2) first, then TEST (1), DROP (0) last; within stage rank by predicted Y.
            score = rac_action.astype(float) * 1000.0 + pred
        elif rac_test is not None:
            p = p_success
            stage = np.where(rac_test > 0.5, 1.0, np.where(p >= 0.5, 2.0, 0.0))
            score = stage * 1000.0 + pred
        else:
            score = pred
    else:
        raise ValueError(name)
    return np.argsort(-np.asarray(score, dtype=float))
