"""Load and join OCx24 HER experimental/computational tables."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

from risktriage.constants import ADSORBATES, DATA_FILES, ENERGY_AGGS, GITHUB_RAW
from risktriage.features import composition_distance, composition_feature_frame

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ocx24"


@dataclass
class HERTask:
    frame: pd.DataFrame
    feature_names: list[str]
    staged_feature_names: list[str]
    y_name: str = "voltage_she"
    meta: dict = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_names].to_numpy(dtype=float)

    @property
    def X_staged(self) -> np.ndarray:
        return self.frame[self.staged_feature_names].to_numpy(dtype=float)

    @property
    def y(self) -> np.ndarray:
        return self.frame[self.y_name].to_numpy(dtype=float)

    @property
    def groups(self) -> np.ndarray:
        return self.frame["group_id"].to_numpy()

    @property
    def loco_groups(self) -> np.ndarray:
        return self.frame["loco_cv"].to_numpy()

    @property
    def n(self) -> int:
        return int(len(self.frame))


def ensure_ocx24_data(data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    for rel in DATA_FILES.values():
        dest = data_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{GITHUB_RAW}/{rel}"
        urllib.request.urlretrieve(url, dest)
    return data_dir


def _energy_columns(frame: pd.DataFrame) -> list[str]:
    cols = []
    for ads in ADSORBATES:
        for agg in ENERGY_AGGS:
            name = f"{ads}_{agg}_energy"
            if name in frame.columns:
                cols.append(name)
    return cols


def load_her_task(
    data_dir: Path | None = None,
    matched_only: bool = False,
    y_star_quantile: float = 0.75,
) -> HERTask:
    data_dir = ensure_ocx24_data(data_dir)
    name = "her_matched" if matched_only else "her_all"
    df = pd.read_csv(data_dir / DATA_FILES[name])
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df = df[df["current density"].astype(float) == 50.0].copy()
    df = df.dropna(subset=["voltage_she"]).reset_index(drop=True)

    energy_cols = _energy_columns(df)
    xrf_feats = composition_feature_frame(df["xrf comp"], prefix="xrf_")
    nom_feats = composition_feature_frame(df["composition"], prefix="nom_")
    df = pd.concat([df.reset_index(drop=True), xrf_feats, nom_feats], axis=1)
    df["xrf_offset"] = [
        composition_distance(a, b) for a, b in zip(df["composition"], df["xrf comp"])
    ]
    df["matched_int"] = df["matched"].astype(bool).astype(float)
    df["rwp_filled"] = pd.to_numeric(df["rwp"], errors="coerce")
    df["q_score_filled"] = pd.to_numeric(df["q_score"], errors="coerce")
    df["rwp_missing"] = df["rwp_filled"].isna().astype(float)
    df["q_missing"] = df["q_score_filled"].isna().astype(float)
    df["rwp_filled"] = df["rwp_filled"].fillna(100.0)
    df["q_score_filled"] = df["q_score_filled"].fillna(0.0)
    df["sabatier_h"] = -np.abs(df["H_mean_energy"].to_numpy(dtype=float))

    if "sid" not in df.columns:
        df["sid"] = df["sample id"].astype(str).str.replace(r"_rep\d+$", "", regex=True)
    df["group_id"] = df["sid"].astype(str)
    if "loco_cv" not in df.columns:
        fam = df["elements"].astype(str)
        df["loco_cv"] = pd.factorize(fam)[0]

    computational = energy_cols + list(nom_feats.columns)
    staged = computational + list(xrf_feats.columns) + [
        "xrf_offset",
        "matched_int",
        "rwp_filled",
        "q_score_filled",
        "rwp_missing",
        "q_missing",
    ]
    computational = [c for c in computational if c in df.columns]
    staged = [c for c in staged if c in df.columns]

    y = df["voltage_she"].to_numpy(dtype=float)
    y_star = float(np.quantile(y, y_star_quantile))
    df["success"] = (y >= y_star).astype(int)
    df["y_star_global"] = y_star

    meta = {
        "n": int(len(df)),
        "n_matched": int(df["matched_int"].sum()),
        "n_groups": int(df["group_id"].nunique()),
        "n_loco": int(df["loco_cv"].nunique()),
        "y_star_global": y_star,
        "y_star_quantile": y_star_quantile,
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "success_rate": float(df["success"].mean()),
        "matched_only": matched_only,
    }
    return HERTask(df, computational, staged, meta=meta)


def load_her_candidates(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = ensure_ocx24_data(data_dir)
    return pd.read_csv(data_dir / DATA_FILES["her_candidates"])
