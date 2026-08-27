# RiskTriage

Calibrating when physical validation is needed in computational catalyst discovery (OCx24 HER).

Manuscript, code, and OCx24 HER data to reproduce [RiskTriage](https://github.com/SusannaDiV/riskTriageCatalyst).

## Reproduce

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m risktriage
```

Data are in `data/ocx24/` (re-downloaded from [fairchem OCx24](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/applications/ocx/data) if missing). Running the CLI writes `results/risktriage/` locally; those outputs are not part of the paper repository.

## Paper

- `paper/risktriage_neurips.tex` — NeurIPS 2026 workshop manuscript
- `paper/references.bib`
- `paper/figures/fig1_risktriage.png`
- `scripts/make_fig1_risktriage.py` — regenerates the main figure after a full run
