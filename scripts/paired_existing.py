from pathlib import Path
from risktriage.sensitivity import paired_from_by_seed_csv

root = Path("results")
out = Path("results/wiley")
out.mkdir(parents=True, exist_ok=True)
frames = []
for task, r in [("co2r", 0.10), ("enthalpy", 0.10), ("enthalpy", 0.03)]:
    p = Path("results") / task / "matched_risk_by_seed.csv"
    df = paired_from_by_seed_csv(p, r=r)
    df["task"] = task
    frames.append(df)
    print(task, r)
    print(df.to_string(index=False))
import pandas as pd
pd.concat(frames, ignore_index=True).to_csv(out / "existing_task_paired.csv", index=False)
