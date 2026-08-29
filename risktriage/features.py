"""Composition parsing and OCx24-style elemental descriptors."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from risktriage.constants import COMP_FEATURE_NAMES, ELEMENT_PROPS

_TOKEN = re.compile(r"([A-Z][a-z]?)[-_]?([0-9]*\.?[0-9]+)")


_FORMULA_EL = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> dict[str, float]:
    """Parse stoichiometric formulas (Fe2O3, NaCl) into mole fractions."""
    s = formula.replace(" ", "")
    if not s:
        return {}
    # strip simple parentheses by expanding once
    while "(" in s:
        s = re.sub(
            r"\(([^\(\)]+)\)(\d*\.?\d*)",
            lambda m: "".join(
                f"{el}{float(n or 1) * float(m.group(2) or 1):g}"
                for el, n in _FORMULA_EL.findall(m.group(1))
            ),
            s,
            count=1,
        )
    counts: dict[str, float] = {}
    for el, n in _FORMULA_EL.findall(s):
        counts[el] = counts.get(el, 0.0) + (float(n) if n else 1.0)
    tot = sum(counts.values())
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in counts.items()}


def parse_composition(comp: str | float | None) -> dict[str, float]:
    if not isinstance(comp, str) or not comp.strip():
        return {}
    s = comp.replace(" ", "")
    pairs = _TOKEN.findall(s)
    if pairs and ("-" in s or "_" in s or any("." in p[1] for p in pairs)):
        out = {el: float(frac) for el, frac in pairs if frac}
        tot = sum(out.values())
        if tot > 0:
            return {k: v / tot for k, v in out.items()}
    parsed = parse_formula(s)
    if parsed:
        return parsed
    els = re.findall(r"[A-Z][a-z]?", s)
    if not els:
        return {}
    w = 1.0 / len(els)
    return {e: w for e in els}


def composition_features(comp: str | float | None) -> np.ndarray:
    fracs = parse_composition(comp)
    vec = np.zeros(len(COMP_FEATURE_NAMES), dtype=float)
    if not fracs:
        vec[:] = np.nan
        return vec
    acc = np.zeros(6, dtype=float)
    wtot = 0.0
    for el, w in fracs.items():
        if el not in ELEMENT_PROPS:
            continue
        acc += w * np.asarray(ELEMENT_PROPS[el], dtype=float)
        wtot += w
    if wtot <= 0:
        vec[:] = np.nan
        return vec
    acc /= wtot
    vec[:6] = acc
    vec[6] = float(len(fracs))
    return vec


def composition_feature_frame(series: pd.Series, prefix: str = "comp_") -> pd.DataFrame:
    mats = np.vstack([composition_features(v) for v in series.tolist()])
    cols = [prefix + n for n in COMP_FEATURE_NAMES]
    return pd.DataFrame(mats, columns=cols, index=series.index)


def composition_distance(a: str, b: str) -> float:
    fa, fb = parse_composition(a), parse_composition(b)
    keys = sorted(set(fa) | set(fb))
    if not keys:
        return np.nan
    va = np.array([fa.get(k, 0.0) for k in keys])
    vb = np.array([fb.get(k, 0.0) for k in keys])
    return float(np.linalg.norm(va - vb))
