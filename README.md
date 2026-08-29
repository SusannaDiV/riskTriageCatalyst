# RiskTriage

Calibrating when physical validation is needed in computational catalyst discovery (OCx24 HER).

Manuscript, code, and OCx24 HER data to reproduce [RiskTriage](https://github.com/SusannaDiV/riskTriageCatalyst).

## Reproduce

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m risktriage --task her
python -m risktriage --task co2r
python -m risktriage --task enthalpy
python -m risktriage --task her --stage wiley
```

- `her` — OCx24 HER (main paper)
- `co2r` — OCx24 CO₂RR, same platform, target = non-H₂ Faradaic efficiency
- `enthalpy` — Kim et al. experimental formation enthalpies with matched MP/OQMD DFT (independent thermodynamics replication)

OCx24 tables live in `data/ocx24/` (from [fairchem](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/applications/ocx/data)). The calorimetry file is fetched from Figshare into `data/enthalpy/` on first enthalpy run. CLI output goes to `results/<task>/` and is gitignored.

## Paper

- `paper/risktriage_neurips.tex` — NeurIPS 2026 workshop manuscript
- `paper/references.bib`
- `paper/figures/fig1_risktriage.png`
- `scripts/make_fig1_risktriage.py` — regenerates the main figure after a full run
