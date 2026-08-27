"""CLI: python -m risktriage [--smoke] [--stage all|data|pred|cal|replay|staged|ood]"""

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
)
from risktriage.data import load_her_task
from risktriage.theory import Costs
import numpy as np


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="RiskTriage on OCx24 HER")
    p.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "pred", "cal", "replay", "staged", "ood", "efficiency"],
    )
    p.add_argument("--outdir", default="results/risktriage")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=None)
    args = p.parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        run_all(out, smoke=args.smoke, n_seeds=args.seeds)
        return
    costs = Costs()
    n_seeds = args.seeds if args.seeds is not None else (3 if args.smoke else 20)
    n_members = 3 if args.smoke else 5
    alpha, r_star, q = 0.1, 0.15, 0.75
    task = load_her_task()
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
        experiment5_staged(task, out, n_seeds, n_members, costs, alpha, r_star, q)
        return
    if args.stage == "ood":
        experiment6_ood(task, out, n_seeds, n_members, costs, alpha, r_star, q)
        return
    if args.stage == "efficiency":
        experiment_cost_sensitivity(task, out, n_seeds, n_members, q, r_star, smoke=args.smoke)
        experiment_matched_efficiency(task, out, n_seeds, n_members, costs, q, smoke=args.smoke)
        return
