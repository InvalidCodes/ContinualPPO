"""Learning-speed and forward-transfer metrics.

Final success rates saturate at 1.0 for several methods, so the evaluation
matrix alone cannot show *how fast* a task was learned or whether starting from
a previous morphology helped. These metrics read the per-task training curves in
``train_stats.csv`` (the ``update`` column is per-task local) to quantify speed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _per_seed_speed(sub: pd.DataFrame, smooth: int = 5) -> dict | None:
    """AUC and time-to-threshold for one (method, desc, seed, task) curve."""

    sub = sub.dropna(subset=["episodic_success"]).sort_values("update")
    if sub.empty:
        return None
    succ = sub["episodic_success"].to_numpy(dtype=float)
    updates = sub["update"].to_numpy(dtype=int)
    smoothed = pd.Series(succ).rolling(smooth, min_periods=1).mean().to_numpy()
    first = next((int(u) for u, v in zip(updates, succ) if v > 0.0), np.nan)
    t08 = next((int(u) for u, v in zip(updates, smoothed) if v >= 0.8), np.nan)
    return {
        "auc": float(succ.mean()),  # area under the (0..1) learning curve
        "time_to_first": first,  # first PPO update with any success
        "time_to_0.8": t08,  # first update with smoothed success >= 0.8
    }


def compute_learning_speed(
    train_csv: str | Path, out_dir: str | Path
) -> pd.DataFrame:
    """Aggregate per-task learning-speed metrics across seeds; write a CSV."""

    df = pd.read_csv(train_csv)
    per_seed = []
    keys = ["method", "use_morphology", "seed", "task"]
    for (method, desc, seed, task), sub in df.groupby(keys):
        speed = _per_seed_speed(sub)
        if speed is None:
            continue
        per_seed.append(
            {"method": method, "use_morphology": desc, "seed": seed, "task": task, **speed}
        )
    per_seed = pd.DataFrame(per_seed)
    if per_seed.empty:
        return per_seed

    summary = (
        per_seed.groupby(["method", "use_morphology", "task"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            time_to_first_mean=("time_to_first", "mean"),
            time_to_0_8_mean=("time_to_0.8", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary[["auc_std"]] = summary[["auc_std"]].fillna(0.0)

    # Forward transfer on the new task: does starting from a previous morphology
    # learn faster than reset (from scratch)? Positive = helpful transfer.
    ft_rows = []
    for desc in summary["use_morphology"].unique():
        d = summary[summary["use_morphology"] == desc]
        for task in d["task"].unique():
            ref = d[(d["method"] == "reset") & (d["task"] == task)]["auc_mean"]
            if ref.empty:
                continue
            reset_auc = float(ref.iloc[0])
            for method in ("finetune", "kl"):
                row = d[(d["method"] == method) & (d["task"] == task)]["auc_mean"]
                if row.empty:
                    continue
                ft_rows.append(
                    {
                        "use_morphology": desc,
                        "task": task,
                        "method": method,
                        "auc": float(row.iloc[0]),
                        "reset_auc": reset_auc,
                        "forward_transfer": float(row.iloc[0]) - reset_auc,
                    }
                )
    forward_transfer = pd.DataFrame(ft_rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "learning_speed.csv", index=False)
    if not forward_transfer.empty:
        forward_transfer.to_csv(out_dir / "forward_transfer.csv", index=False)
    return summary
