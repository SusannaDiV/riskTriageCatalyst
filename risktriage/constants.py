"""Elemental property tables and OCx24 download locations."""

from __future__ import annotations

GITHUB_RAW = "https://raw.githubusercontent.com/facebookresearch/fairchem/main/src/fairchem/applications/ocx/data"

DATA_FILES = {
    "her_all": "processed_data/HER_40_70_all.csv",
    "her_matched": "processed_data/HER_40_70_matched.csv",
    "co2r_all": "processed_data/CO2R_40_70_all.csv",
    "co2r_matched": "processed_data/CO2R_40_70_matched.csv",
    "expt": "experimental_data/ExpDataDump_241113_clean.csv",
    "xrf": "experimental_data/XRFDataDump-241113.csv",
    "xrd": "experimental_data/XRDDataDump-241113.csv",
    "her_candidates": "computational_data/her_candidates.csv",
}

ENTHALPY_FIGSHARE = "https://ndownloader.figshare.com/files/9218491"

ADSORBATES = ["H", "OH", "CO", "C", "CHO", "COCOH"]
ENERGY_AGGS = ["mean", "wulff", "boltz"]

# (period, group, atomic_mass, atomic_radius_pm, Pauling_X, Mendeleev_no)
ELEMENT_PROPS: dict[str, tuple[float, float, float, float, float, float]] = {
    "H": (1, 1, 1.008, 31, 2.20, 92),
    "Li": (2, 1, 6.94, 167, 0.98, 12),
    "Be": (2, 2, 9.012, 112, 1.57, 17),
    "B": (2, 13, 10.81, 87, 2.04, 86),
    "C": (2, 14, 12.011, 67, 2.55, 87),
    "N": (2, 15, 14.007, 56, 3.04, 88),
    "O": (2, 16, 15.999, 48, 3.44, 89),
    "F": (2, 17, 18.998, 42, 3.98, 90),
    "Na": (3, 1, 22.990, 190, 0.93, 11),
    "Mg": (3, 2, 24.305, 145, 1.31, 18),
    "Al": (3, 13, 26.982, 118, 1.61, 80),
    "Si": (3, 14, 28.085, 111, 1.90, 85),
    "P": (3, 15, 30.974, 98, 2.19, 86),
    "S": (3, 16, 32.06, 88, 2.58, 88),
    "Cl": (3, 17, 35.45, 79, 3.16, 89),
    "K": (4, 1, 39.098, 243, 0.82, 10),
    "Ca": (4, 2, 40.078, 194, 1.00, 19),
    "Sc": (4, 3, 44.956, 184, 1.36, 20),
    "Ti": (4, 4, 47.867, 176, 1.54, 21),
    "V": (4, 5, 50.942, 171, 1.63, 22),
    "Cr": (4, 6, 51.996, 166, 1.66, 23),
    "Mn": (4, 7, 54.938, 161, 1.55, 24),
    "Fe": (4, 8, 55.845, 156, 1.83, 25),
    "Co": (4, 9, 58.933, 152, 1.88, 26),
    "Ni": (4, 10, 58.693, 149, 1.91, 27),
    "Cu": (4, 11, 63.546, 145, 1.90, 28),
    "Zn": (4, 12, 65.38, 142, 1.65, 29),
    "Ga": (4, 13, 69.723, 136, 1.81, 81),
    "Ge": (4, 14, 72.63, 125, 2.01, 84),
    "As": (4, 15, 74.922, 114, 2.18, 85),
    "Se": (4, 16, 78.971, 103, 2.55, 87),
    "Y": (5, 3, 88.906, 212, 1.22, 20),
    "Zr": (5, 4, 91.224, 206, 1.33, 21),
    "Nb": (5, 5, 92.906, 198, 1.60, 22),
    "Mo": (5, 6, 95.95, 190, 2.16, 23),
    "Ru": (5, 8, 101.07, 178, 2.20, 25),
    "Rh": (5, 9, 102.91, 173, 2.28, 26),
    "Pd": (5, 10, 106.42, 169, 2.20, 27),
    "Ag": (5, 11, 107.87, 165, 1.93, 28),
    "Cd": (5, 12, 112.41, 161, 1.69, 29),
    "In": (5, 13, 114.82, 156, 1.78, 81),
    "Sn": (5, 14, 118.71, 145, 1.96, 83),
    "Sb": (5, 15, 121.76, 133, 2.05, 84),
    "Te": (5, 16, 127.60, 123, 2.10, 86),
    "Hf": (6, 4, 178.49, 208, 1.30, 21),
    "Ta": (6, 5, 180.95, 200, 1.50, 22),
    "W": (6, 6, 183.84, 193, 2.36, 23),
    "Re": (6, 7, 186.21, 188, 1.90, 24),
    "Os": (6, 8, 190.23, 185, 2.20, 25),
    "Ir": (6, 9, 192.22, 180, 2.20, 26),
    "Pt": (6, 10, 195.08, 177, 2.28, 27),
    "Au": (6, 11, 196.97, 174, 2.54, 28),
    "Hg": (6, 12, 200.59, 171, 2.00, 29),
    "Tl": (6, 13, 204.38, 156, 1.62, 81),
    "Pb": (6, 14, 207.2, 154, 2.33, 82),
    "Bi": (6, 15, 208.98, 143, 2.02, 83),
}

COMP_FEATURE_NAMES = [
    "mean_period",
    "mean_group",
    "mean_mass",
    "mean_radius",
    "mean_en",
    "mean_mendeleev",
    "n_elements",
]
