# RiskTriage

Calibrating when physical validation is needed in computational catalyst discovery (OCx24 HER).

Code, paper sources, OCx24 HER data, and experiment outputs for [github.com/SusannaDiV/riskTriageCatalyst](https://github.com/SusannaDiV/riskTriageCatalyst).

## Reproduce

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m risktriage --smoke
python -m risktriage
```

Data live in `data/ocx24/` (also fetched from [fairchem OCx24](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/applications/ocx/data) on first run if missing). Results are written to `results/risktriage/`.

## Paper

- `paper/risktriage_neurips.tex` — NeurIPS 2026 workshop draft
- `paper/figures/fig1_risktriage.png` — main figure
- `scripts/make_fig1_risktriage.py` — figure generator

Primary endpoint: physical-testing fraction at matched decision risk. Secondary: retrospective discovery replay vs experimental budget.
