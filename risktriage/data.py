"""Load OCx24 electrochemistry and experimental formation-enthalpy tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

from risktriage.constants import ADSORBATES, DATA_FILES, ENERGY_AGGS, ENTHALPY_FIGSHARE, GITHUB_RAW
from risktriage.features import composition_distance, composition_feature_frame, parse_formula

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "ocx24"
ENTHALPY_DIR = ROOT / "data" / "enthalpy"


@dataclass
class TriageTask:
    frame: pd.DataFrame
    feature_names: list[str]
    staged_feature_names: list[str]
    y_name: str = "voltage_she"
    meta: dict = field(default_factory=dict)
    has_staged: bool = True

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_names].to_numpy(dtype=float)

    @property
    def X_staged(self) -> np.ndarray:
        cols = self.staged_feature_names or self.feature_names
        return self.frame[cols].to_numpy(dtype=float)

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


HERTask = TriageTask


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


def _finalize_ocx(df: pd.DataFrame, y_col: str, y_star_quantile: float, task_name: str) -> TriageTask:
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df = df.dropna(subset=[y_col]).reset_index(drop=True)

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
    h_col = "H_mean_energy" if "H_mean_energy" in df.columns else energy_cols[0]
    df["sabatier_h"] = -np.abs(df[h_col].to_numpy(dtype=float))

    if "sid" not in df.columns or df["sid"].isna().all():
        df["sid"] = df["sample id"].astype(str).str.replace(r"_rep\d+$", "", regex=True)
    df["sid"] = df["sid"].astype(str)
    df["group_id"] = df["sid"]
    if "loco_cv" not in df.columns:
        fam = df["elements"].astype(str) if "elements" in df.columns else df["composition"].astype(str)
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

    y = df[y_col].to_numpy(dtype=float)
    y_star = float(np.quantile(y, y_star_quantile))
    df["success"] = (y >= y_star).astype(int)
    df["y_star_global"] = y_star

    meta = {
        "task": task_name,
        "n": int(len(df)),
        "n_matched": int(df["matched_int"].sum()),
        "n_groups": int(df["group_id"].nunique()),
        "n_loco": int(pd.Series(df["loco_cv"]).nunique()),
        "y_name": y_col,
        "y_star_global": y_star,
        "y_star_quantile": y_star_quantile,
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "success_rate": float(df["success"].mean()),
    }
    return TriageTask(df, computational, staged, y_name=y_col, meta=meta, has_staged=True)


def load_her_task(
    data_dir: Path | None = None,
    matched_only: bool = False,
    y_star_quantile: float = 0.75,
) -> TriageTask:
    data_dir = ensure_ocx24_data(data_dir)
    name = "her_matched" if matched_only else "her_all"
    df = pd.read_csv(data_dir / DATA_FILES[name])
    df = df[df["current density"].astype(float) == 50.0].copy()
    return _finalize_ocx(df, "voltage_she", y_star_quantile, "ocx24_her")


def load_co2r_task(
    data_dir: Path | None = None,
    matched_only: bool = False,
    y_star_quantile: float = 0.75,
) -> TriageTask:
    """CO2RR selectivity: non-H2 Faradaic efficiency (higher is better)."""
    data_dir = ensure_ocx24_data(data_dir)
    name = "co2r_matched" if matched_only else "co2r_all"
    df = pd.read_csv(data_dir / DATA_FILES[name])
    fe_h2 = pd.to_numeric(df["fe_h2"], errors="coerce")
    df = df.assign(fe_co2rr=100.0 - fe_h2)
    df = df.dropna(subset=["fe_co2rr"]).copy()
    return _finalize_ocx(df, "fe_co2rr", y_star_quantile, "ocx24_co2r")


def load_her_candidates(data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = ensure_ocx24_data(data_dir)
    return pd.read_csv(data_dir / DATA_FILES["her_candidates"])


def _energy_from_pairs(obj) -> float:
    if obj is None:
        return np.nan
    if isinstance(obj, (int, float)):
        return float(obj)
    if isinstance(obj, dict):
        for k in ("eV/atom", "eV", "kJ/mol"):
            if k in obj and obj[k] is not None:
                v = float(obj[k])
                return v / 96.485 if k.startswith("kJ") else v
        return np.nan
    if not isinstance(obj, list) or not obj:
        return np.nan

    def _as_pair(item):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        a, b = item[0], item[1]
        if isinstance(a, (int, float)) and isinstance(b, str):
            return float(a), str(b)
        if isinstance(b, (int, float)) and isinstance(a, str):
            return float(b), str(a)
        return None

    prefer = {"ev/atom": 0, "ev": 1, "kj/mol of atom": 2, "kj/mol": 3}
    best = None
    rank = 99
    for item in obj:
        pair = _as_pair(item)
        if pair is None:
            continue
        val, unit = pair
        u = unit.lower().replace(" ", "")
        r = 50
        if "ev/atom" in u:
            r = 0
        elif u == "ev":
            r = 1
        elif "kj" in u:
            r = 2
        if r < rank:
            rank = r
            best = val / 96.485 if "kj" in u else val
    return float(best) if best is not None else np.nan


def ensure_enthalpy_data(data_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir) if data_dir else ENTHALPY_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / "enthalpy_formation.json"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    urllib.request.urlretrieve(ENTHALPY_FIGSHARE, dest)
    return dest


def load_enthalpy_task(
    data_dir: Path | None = None,
    y_star_quantile: float = 0.75,
    require_dft: bool = True,
) -> TriageTask:
    """DFT formation energy + composition → experimental formation enthalpy.

    Y is −ΔH_f^expt (eV/atom) so that more stable compounds score higher, matching
    the HER convention that larger Y is the desired outcome.
    """
    path = ensure_enthalpy_data(data_dir)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", list(raw.values()))
    rows = []
    for rec in raw:
        if not isinstance(rec, dict):
            continue
        mp = rec.get("materials_project") or {}
        oq = rec.get("oqmd") or {}
        y_expt = _energy_from_pairs(
            rec.get("standard_enthalpy_formation") or rec.get("enthalpy_formation")
        )
        e_mp = _energy_from_pairs(mp.get("mp_formation_energy") or rec.get("mp_formation_energy"))
        e_oqmd = _energy_from_pairs(oq.get("oqmd_formation_energy") or rec.get("oqmd_formation_energy"))
        formula = rec.get("formula") or rec.get("reduced_formula") or ""
        if not formula or not np.isfinite(y_expt):
            continue
        if require_dft and not (np.isfinite(e_mp) or np.isfinite(e_oqmd)):
            continue
        rows.append(
            {
                "formula": str(formula),
                "e_form_expt": float(y_expt),
                "e_form_mp": float(e_mp) if np.isfinite(e_mp) else np.nan,
                "e_form_oqmd": float(e_oqmd) if np.isfinite(e_oqmd) else np.nan,
                "mpid": mp.get("mp_id") or rec.get("mp_id") or rec.get("mpid") or "",
                "oqmdid": oq.get("oqmd_id") or rec.get("oqmd_id") or rec.get("oqmdid") or "",
                "space_group": rec.get("space_group") or rec.get("space group") or "",
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No usable experimental formation-enthalpy rows.")
    df = df.drop_duplicates(subset=["formula", "e_form_expt"]).reset_index(drop=True)
    # More negative experimental enthalpy = more stable → flip sign.
    df["neg_e_form_expt"] = -df["e_form_expt"]
    nom_feats = composition_feature_frame(df["formula"], prefix="nom_")
    df = pd.concat([df, nom_feats], axis=1)
    df["has_mp"] = df["e_form_mp"].notna().astype(float)
    df["has_oqmd"] = df["e_form_oqmd"].notna().astype(float)
    df["matched_int"] = ((df["has_mp"] + df["has_oqmd"]) > 0).astype(float)
    df["sabatier_h"] = -np.abs(df["e_form_mp"].fillna(df["e_form_oqmd"]))
    df["group_id"] = df["formula"].astype(str)
    systems = []
    for f in df["formula"]:
        els = tuple(sorted(parse_formula(str(f))))
        systems.append("-".join(els) if els else str(f))
    df["chem_system"] = systems
    df["loco_cv"] = pd.factorize(df["chem_system"])[0]

    feats = ["e_form_mp", "e_form_oqmd"] + list(nom_feats.columns)
    feats = [c for c in feats if c in df.columns]
    y = df["neg_e_form_expt"].to_numpy(dtype=float)
    y_star = float(np.quantile(y, y_star_quantile))
    df["success"] = (y >= y_star).astype(int)
    df["y_star_global"] = y_star
    meta = {
        "task": "expt_formation_enthalpy",
        "n": int(len(df)),
        "n_with_mp": int(df["has_mp"].sum()),
        "n_with_oqmd": int(df["has_oqmd"].sum()),
        "n_groups": int(df["group_id"].nunique()),
        "n_loco": int(df["loco_cv"].nunique()),
        "y_name": "neg_e_form_expt",
        "y_star_global": y_star,
        "y_star_quantile": y_star_quantile,
        "y_mean": float(np.mean(y)),
        "y_std": float(np.std(y)),
        "success_rate": float(df["success"].mean()),
        "require_dft": require_dft,
        "source": "Kim et al., Sci. Data 2017 (Figshare 5193229 / matminer expt_formation_enthalpy)",
    }
    return TriageTask(df, feats, feats, y_name="neg_e_form_expt", meta=meta, has_staged=False)


def load_task(name: str, **kwargs) -> TriageTask:
    name = name.lower().replace("-", "_")
    if name in {"her", "ocx24_her"}:
        return load_her_task(**kwargs)
    if name in {"co2r", "co2rr", "ocx24_co2r"}:
        return load_co2r_task(**kwargs)
    if name in {"enthalpy", "formation", "expt_formation_enthalpy"}:
        return load_enthalpy_task(**kwargs)
    raise ValueError(f"Unknown task {name}")
