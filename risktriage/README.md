# RiskTriage: risk-averse simulation-to-experiment decisions (OCx24 HER)

Reproduce the OCx24 HER joined table (179 targets at 50 mA cm$^{-2}$) and run decision-calibrated uncertainty experiments.

```bash
python -m risktriage --smoke
python -m risktriage
```

Data are fetched from [fairchem OCx24](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/applications/ocx/data) into `data/ocx24/` on first run.

Primary endpoint: physical test fraction at controlled scientific decision risk.
Secondary: recall of experimentally top-decile catalysts vs budget.
