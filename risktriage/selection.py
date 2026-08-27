"""Optional IPW sensitivity for non-uniform electrochemical testing."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def propensity_weights(X: np.ndarray, tested: np.ndarray) -> np.ndarray:
    y = np.asarray(tested, dtype=int)
    if len(np.unique(y)) < 2:
        return np.ones(len(X))
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=400)),
        ]
    )
    pipe.fit(np.nan_to_num(X, nan=0.0), y)
    p = pipe.predict_proba(np.nan_to_num(X, nan=0.0))[:, 1]
    p = np.clip(p, 0.05, 0.95)
    w = np.zeros(len(X))
    w[y == 1] = 1.0 / p[y == 1]
    w = w / w[y == 1].mean()
    return w
