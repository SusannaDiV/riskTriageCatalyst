"""CLI: python -m risktriage [--task her|co2r|enthalpy] [--stage ...]"""

from __future__ import annotations

import argparse
from pathlib import Path

from risktriage.experiments import (
    experiment1_prediction,
    experiment2_calibration,
    experiment3_frontier,
    experiment4_replay,
    experiment5_staged,
    experiment6_ood,
    experiment_cost_sensitivity,
    experiment_matched_efficiency,
    run_all,
    run_core,
)
from risktriage.data import load_her_task, load_task
from risktriage.theory import Costs
import numpy as np


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="RiskTriage: simulation-to-experiment decisions")
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "pred", "cal", "replay", "staged", "ood", "efficiency", "wiley"],
    )
    p.add_argument(
        "--task",
        default="her",
        choices=["her", "co2r", "enthalpy"],
        help="her = OCx24 HER (paper); co2r = OCx24 CO2RR; enthalpy = Kim calorimetry + MP/OQMD DFT",
    )
    p.add_argument("--outdir", default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=None)
    args = p.parse_args(argv)
    defaults = {"her": "results/risktriage", "co2r": "results/co2r", "enthalpy": "results/enthalpy"}
    out = Path(args.outdir or ("results/wiley" if args.stage == "wiley" else defaults[args.task]))
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "her" and args.stage == "all":
        run_all(out, smoke=args.smoke, n_seeds=args.seeds)
        return
    if args.stage == "all":
        run_core(load_task(args.task), out, smoke=args.smoke, n_seeds=args.seeds)
        return

    costs = Costs()
    n_seeds = args.seeds if args.seeds is not None else (3 if args.smoke else 20)
    n_members = 3 if args.smoke else 5
    alpha, r_star, q = 0.1, 0.15, 0.75
    task = load_her_task() if args.task == "her" else load_task(args.task)
    if args.stage == "data":
        (out / "task_meta.json").write_text(__import__("json").dumps(task.meta, indent=2))
        print(task.meta)
        return
    if args.stage == "pred":
        experiment1_prediction(task, out, smoke=args.smoke)
        return
    if args.stage == "cal":
        experiment2_calibration(task, out, n_seeds, n_members, costs, alpha, r_star, q)
        r_grid = np.array([0.05, 0.1, 0.2, 0.3])
        experiment3_frontier(task, out, n_seeds, n_members, costs, q, r_grid, np.array([0.1, 0.2]))
        return
    if args.stage == "replay":
        experiment4_replay(task, out, n_seeds, n_members, costs, alpha, r_star, q, [5, 10, 20, 30])
        return
    if args.stage == "staged":
        if not task.has_staged:
            raise SystemExit(f"{args.task} has no characterization stage")
        experiment5_staged(task, out, n_seeds, n_members, costs, alpha, r_star, q)
        return
    if args.stage == "ood":
        experiment6_ood(task, out, n_seeds, n_members, costs, alpha, r_star, q)
        return
    if args.stage == "efficiency":
        experiment_cost_sensitivity(task, out, n_seeds, n_members, q, r_star, smoke=args.smoke)
        experiment_matched_efficiency(task, out, n_seeds, n_members, costs, q, smoke=args.smoke)
        return
    if args.stage == "wiley":
        from risktriage.sensitivity import experiment_wiley

        experiment_wiley(load_task(args.task) if args.task != "her" else load_her_task(), out, n_seeds=n_seeds, n_members=n_members, smoke=args.smoke)
        return
