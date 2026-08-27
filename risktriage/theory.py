"""Bayes-optimal triage, VOI, and catalyst-specific decision theory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Costs:
    c_fp: float = 1.0
    c_fn: float = 1.0
    c_e: float = 0.2

    @property
    def harmonic_cap(self) -> float:
        return (self.c_fp * self.c_fn) / (self.c_fp + self.c_fn)

    @property
    def p_drop(self) -> float:
        return self.c_e / self.c_fn

    @property
    def p_trust(self) -> float:
        return 1.0 - self.c_e / self.c_fp


DROP, TEST, TRUST = 0, 1, 2
ACTION_NAMES = {DROP: "DROP", TEST: "TEST", TRUST: "TRUST"}


def has_testing_region(costs: Costs) -> bool:
    return costs.c_e < costs.harmonic_cap


def bayes_action_from_p(p: np.ndarray, costs: Costs) -> np.ndarray:
    """Proposition 1: DROP / EXPERIMENT / TRUST from success probability."""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, TEST, dtype=int)
    if not has_testing_region(costs):
        # experiment is never uniquely optimal; choose TRUST vs DROP
        out = np.where(p * costs.c_fn >= (1.0 - p) * costs.c_fp, TRUST, DROP)
        return out
    out = np.where(p <= costs.p_drop, DROP, out)
    out = np.where(p >= costs.p_trust, TRUST, out)
    return out


def expected_risks(p: np.ndarray, costs: Costs) -> dict[str, np.ndarray]:
    p = np.asarray(p, dtype=float)
    return {
        "TRUST": (1.0 - p) * costs.c_fp,
        "DROP": p * costs.c_fn,
        "TEST": np.full_like(p, costs.c_e, dtype=float),
    }


def voi_perfect_experiment(p: np.ndarray, costs: Costs) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.minimum(p * costs.c_fn, (1.0 - p) * costs.c_fp)


def loss_matrix(action: np.ndarray, success: np.ndarray, costs: Costs) -> np.ndarray:
    a = np.asarray(action, dtype=int)
    g = np.asarray(success, dtype=float)
    loss = np.zeros(len(a), dtype=float)
    loss[a == TRUST] = costs.c_fp * (1.0 - g[a == TRUST])
    loss[a == DROP] = costs.c_fn * g[a == DROP]
    loss[a == TEST] = costs.c_e
    return loss


def triage_loss(action: np.ndarray, success: np.ndarray, costs: Costs) -> np.ndarray:
    """Decision error only: TEST contributes 0 (pays money, not misclassification)."""
    a = np.asarray(action, dtype=int)
    g = np.asarray(success, dtype=float)
    loss = np.zeros(len(a), dtype=float)
    loss[a == TRUST] = costs.c_fp * (1.0 - g[a == TRUST])
    loss[a == DROP] = costs.c_fn * g[a == DROP]
    return loss


def interval_action(lower: np.ndarray, upper: np.ndarray, y_star: float) -> np.ndarray:
    """Theorem 1 mapping from a prediction interval to TRUST / DROP / TEST."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    a = np.full(lo.shape, TEST, dtype=int)
    a[lo >= y_star] = TRUST
    a[hi < y_star] = DROP
    return a


def robust_action(lower: np.ndarray, upper: np.ndarray, y_star: float, costs: Costs) -> np.ndarray:
    """Proposition 3 max-min action on an interval for the three catalyst utilities."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    a = np.empty(lo.shape, dtype=int)
    for i in range(len(lo)):
        # inf_y in C of u(TRUST), u(DROP), u(TEST)
        # TRUST worst case: if C can contain a failure (lo < y*), pay c_fp; else 0
        u_trust = 0.0 if lo[i] >= y_star else -costs.c_fp
        u_drop = 0.0 if hi[i] < y_star else -costs.c_fn
        u_test = -costs.c_e
        a[i] = [DROP, TEST, TRUST][int(np.argmax([u_drop, u_test, u_trust]))]
    return a


def cost_aware_interval_action(
    lower: np.ndarray,
    upper: np.ndarray,
    y_star: float,
    costs: Costs,
    pred: np.ndarray | None = None,
) -> np.ndarray:
    """Theorem 1 interval rule, but TEST is used only when it is Bayes-admissible.

    If c_E is above the harmonic cap, a straddling interval is resolved to TRUST
    or DROP using the point prediction rather than paying for an experiment that
    cannot be optimal in expectation.
    """
    a = interval_action(lower, upper, y_star)
    if has_testing_region(costs):
        return a
    pred = np.asarray(pred if pred is not None else 0.5 * (np.asarray(lower) + np.asarray(upper)))
    a = np.where(pred >= y_star, TRUST, DROP)
    return a


def gaussian_success_prob(mu: np.ndarray, sigma: np.ndarray, y_star: float) -> np.ndarray:
    from math import erf, sqrt

    mu = np.asarray(mu, dtype=float)
    sigma = np.clip(np.asarray(sigma, dtype=float), 1e-6, None)
    z = (mu - y_star) / (sigma * sqrt(2.0))
    # P(Y >= y*) = 1 - Phi((y*-mu)/sig) = 0.5 * erfc((y*-mu)/(sig sqrt2))
    p = 0.5 * (1.0 + np.vectorize(erf)(z))
    return np.clip(p, 1e-6, 1.0 - 1e-6)
