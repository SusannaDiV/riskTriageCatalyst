"""Boring predictors: linear, ridge, RF, LightGBM, small LGBM ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


def _pipe(model) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", model),
        ]
    )


def make_model(name: str, seed: int = 0):
    if name == "linear":
        return _pipe(LinearRegression())
    if name == "ridge":
        return _pipe(Ridge(alpha=1.0))
    if name == "rf":
        return _pipe(RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1))
    if name == "lgbm":
        if lgb is None:
            raise ImportError("lightgbm is required")
        return _pipe(
            lgb.LGBMRegressor(
                n_estimators=250,
                learning_rate=0.05,
                num_leaves=16,
                min_child_samples=5,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                verbosity=-1,
                force_row_wise=True,
            )
        )
    raise ValueError(name)


@dataclass
class EnsemblePred:
    mean: np.ndarray
    std: np.ndarray
    members: np.ndarray


class LGBMEnsemble:
    def __init__(self, n_members: int = 5, seed: int = 0):
        self.n_members = n_members
        self.seed = seed
        self.models: list = []

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_members):
            m = make_model("lgbm", seed=self.seed + 17 * i)
            m.fit(X, y)
            self.models.append(m)
        return self

    def predict_dist(self, X) -> EnsemblePred:
        members = np.column_stack([m.predict(X) for m in self.models])
        return EnsemblePred(members.mean(axis=1), members.std(axis=1), members)

    def predict(self, X) -> np.ndarray:
        return self.predict_dist(X).mean
