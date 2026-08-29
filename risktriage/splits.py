"""Grouped train / calibration / test splits without chemistry leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class Split:
    train: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def _assign_groups(groups: np.ndarray, frac_train: float, frac_cal: float, rng: np.random.Generator) -> dict:
    uniq = np.array(sorted(np.unique(groups), key=str))
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = max(1, int(round(frac_train * n)))
    n_cal = max(1, int(round(frac_cal * n)))
    if n_train + n_cal >= n:
        n_cal = max(1, n - n_train - 1)
    n_test = n - n_train - n_cal
    if n_test < 1:
        n_train = max(1, n - 2)
        n_cal = 1
        n_test = n - n_train - n_cal
    parts = {
        "train": set(uniq[:n_train]),
        "cal": set(uniq[n_train : n_train + n_cal]),
        "test": set(uniq[n_train + n_cal :]),
    }
    return parts


def grouped_split(
    groups: np.ndarray,
    seed: int = 0,
    frac_train: float = 0.65,
    frac_cal: float = 0.175,
) -> Split:
    rng = np.random.default_rng(seed)
    parts = _assign_groups(groups, frac_train, frac_cal, rng)
    idx = np.arange(len(groups))
    def take(name: str) -> np.ndarray:
        return idx[np.array([g in parts[name] for g in groups])]
    return Split(take("train"), take("cal"), take("test"))


def ood_cluster_split(
    X: np.ndarray,
    groups: np.ndarray,
    seed: int = 0,
    n_clusters: int = 8,
    holdout_clusters: int = 2,
    frac_cal_of_in: float = 0.25,
) -> Split:
    """Hold out entire composition clusters as OOD test; split remainder into train/cal."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # cluster group centroids
    cents = []
    for g in uniq:
        mask = groups == g
        cents.append(np.nanmean(np.nan_to_num(X[mask], nan=0.0), axis=0))
    cents = np.nan_to_num(np.vstack(cents), nan=0.0)
    k = min(n_clusters, len(uniq))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    lab = km.fit_predict(cents)
    cluster_ids = np.arange(k)
    rng.shuffle(cluster_ids)
    hold = set(cluster_ids[: max(1, min(holdout_clusters, k - 2))])
    ood_groups = {uniq[i] for i, c in enumerate(lab) if c in hold}
    in_groups = [g for g in uniq if g not in ood_groups]
    rng.shuffle(in_groups)
    n_cal_g = max(1, int(round(frac_cal_of_in * len(in_groups))))
    cal_g = set(in_groups[:n_cal_g])
    train_g = set(in_groups[n_cal_g:])
    idx = np.arange(len(groups))
    train = idx[np.array([g in train_g for g in groups])]
    cal = idx[np.array([g in cal_g for g in groups])]
    test = idx[np.array([g in ood_groups for g in groups])]
    return Split(train, cal, test)


def loco_folds(loco: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    for g in np.unique(loco):
        test = np.where(loco == g)[0]
        train = np.where(loco != g)[0]
        if len(test) == 0 or len(train) == 0:
            continue
        folds.append((train, test))
    return folds
