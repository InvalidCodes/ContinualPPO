"""CSV aggregation and plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def aggregate_metrics(metrics_csv: str | Path, out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    group_cols = ["method", "use_morphology", "stage", "eval_task"]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            success_mean=("success_rate", "mean"),
            success_std=("success_rate", "std"),
            return_mean=("return_mean", "mean"),
            return_std=("return_mean", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    summary["success_std"] = summary["success_std"].fillna(0.0)
    summary["return_std"] = summary["return_std"].fillna(0.0)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def plot_matrix(summary: pd.DataFrame, out_path: str | Path) -> None:
    """Render the success matrix as an annotated grid (rows x stage->task)."""

    if summary.empty:
        return
    df = summary.copy()
    df["row"] = df.apply(
        lambda r: f"{r['method']} | {'desc' if bool(r['use_morphology']) else 'no-desc'}",
        axis=1,
    )
    df["col"] = df["stage"].astype(str) + " -> " + df["eval_task"].astype(str)
    mean_grid = df.pivot_table(index="row", columns="col", values="success_mean")
    std_grid = df.pivot_table(index="row", columns="col", values="success_std")

    fig, ax = plt.subplots(
        figsize=(1.4 * len(mean_grid.columns) + 3.0, 0.6 * len(mean_grid.index) + 2.0)
    )
    ax.imshow(mean_grid.to_numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(mean_grid.columns)))
    ax.set_xticklabels(mean_grid.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(mean_grid.index)))
    ax.set_yticklabels(mean_grid.index, fontsize=8)
    for i in range(len(mean_grid.index)):
        for j in range(len(mean_grid.columns)):
            mean = mean_grid.iat[i, j]
            if pd.isna(mean):
                continue
            std = std_grid.iat[i, j]
            text = f"{mean:.2f}" if pd.isna(std) else f"{mean:.2f}\n±{std:.2f}"
            ax.text(
                j, i, text, ha="center", va="center", fontsize=8,
                color="white" if mean > 0.55 else "black",
            )
    ax.set_title("Success rate (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(
    train_csv: str | Path, out_path: str | Path, smooth: int = 5
) -> None:
    path = Path(train_csv)
    if not path.exists():
        return
    df = pd.read_csv(path)
    if df.empty or "episodic_success" not in df:
        return
    df = df.dropna(subset=["episodic_success"])
    if df.empty:
        return

    # ``update`` is per-task local (each task is a fresh train_ppo call), so it
    # aligns curves across methods regardless of accumulated global_step.
    curve = (
        df.groupby(["method", "use_morphology", "task", "update"])
        .agg(
            success_mean=("episodic_success", "mean"),
            success_std=("episodic_success", "std"),
        )
        .reset_index()
    )
    curve["success_std"] = curve["success_std"].fillna(0.0)

    tasks = sorted(curve["task"].unique())
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(5.0 * len(tasks), 4.2), sharey=True, squeeze=False
    )
    for ax, task in zip(axes[0], tasks):
        for (method, desc), sub in curve[curve["task"] == task].groupby(
            ["method", "use_morphology"]
        ):
            sub = sub.sort_values("update")
            y = sub["success_mean"].rolling(smooth, min_periods=1).mean().to_numpy()
            s = sub["success_std"].rolling(smooth, min_periods=1).mean().to_numpy()
            x = sub["update"].to_numpy()
            label = f"{method} {'desc' if desc else 'no-desc'}"
            ax.plot(x, y, label=label)
            ax.fill_between(x, np.clip(y - s, 0, 1), np.clip(y + s, 0, 1), alpha=0.12)
        ax.set_title(f"task {task}")
        ax.set_xlabel("PPO update (per task)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("Episode success (smoothed)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
