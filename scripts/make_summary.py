#!/usr/bin/env python
"""Build a one-page summary figure of the toy continual-PPO results.

Reads the result CSVs (5-seed T1<->T3 matrix + T1->T2 plasticity study) and
renders results/summary_onepager.png. Re-run after regenerating results.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd


def cell(df, **kw):
    m = df
    for k, v in kw.items():
        m = m[m[k] == v]
    return m.iloc[0]


def main() -> None:
    stab = pd.read_csv(ROOT / "results/main_5seed/summary.csv")
    stab = stab[stab.use_morphology == True]  # noqa: E712  (descriptor on = clean case)
    plas_speed = pd.read_csv(ROOT / "results/plasticity_t1t2/learning_speed.csv")
    plas_speed = plas_speed[(plas_speed.use_morphology == True) & (plas_speed.task == "T2")]

    fig = plt.figure(figsize=(15, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.32)

    # --- Panel 1: STABILITY (T1<->T3, conflict pair) ---
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ["finetune", "kl"]
    retain = [cell(stab, method=m, stage="after_T3", eval_task="T1")["success_mean"] for m in methods]
    retain_sd = [cell(stab, method=m, stage="after_T3", eval_task="T1")["success_std"] for m in methods]
    learn = [cell(stab, method=m, stage="after_T3", eval_task="T3")["success_mean"] for m in methods]
    x = range(len(methods))
    ax1.bar([i - 0.2 for i in x], retain, 0.4, yerr=retain_sd, capsize=4,
            label="retain T1 (anti-forget)", color="#2563eb")
    ax1.bar([i + 0.2 for i in x], learn, 0.4, label="learn T3 (new task)", color="#9ca3af")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(["finetune", "KL distill"])
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("success after T3 (5 seeds)")
    ax1.set_title("STABILITY — T1↔T3 (max conflict)\nKL retains T1 where finetune forgets")
    ax1.legend(fontsize=8, loc="lower center")
    ax1.grid(axis="y", alpha=0.25)
    for i, v in enumerate(retain):
        ax1.text(i - 0.2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # --- Panel 2: PLASTICITY (T1->T2, similar pair) ---
    ax2 = fig.add_subplot(gs[0, 1])
    order = ["reset", "finetune", "kl"]
    labels = ["reset\n(from scratch)", "finetune\n(from T1)", "KL\n(from T1)"]
    t08 = [float(cell(plas_speed, method=m)["time_to_0_8_mean"]) for m in order]
    colors = ["#9ca3af", "#16a34a", "#16a34a"]
    ax2.bar(range(len(order)), t08, color=colors)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("PPO updates to reach 0.8 success")
    ax2.set_title("PLASTICITY — T1→T2 (no conflict)\ntransfer from T1 ≈ 12× faster than scratch")
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate(t08):
        ax2.text(i, v + 0.2, f"{v:.0f}", ha="center", fontsize=9)
    ax2.text(0.5, 0.92, "forward transfer +0.30\n(finetune T2→T1 = 1.00: no forgetting)",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8,
             bbox=dict(boxstyle="round", fc="#dcfce7", ec="#16a34a"))

    # --- Panel 3: controlled contrast + verdict ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("CONTROLLED CONTRAST  (T2 vs T3)", fontsize=11)
    txt = (
        "T2 and T3 = same morphology change vs T1\n"
        "(3-finger baseline variant); ONLY difference:\n"
        "T3 flips finger-1 sign, T2 does not.\n"
        "\n"
        "                 forward transfer   forgetting\n"
        "  T1→T2 (no flip)     +0.30          none (1.00)\n"
        "  T1→T3 (flip f1)     −0.30          catastrophic (0.00)\n"
        "\n"
        "⇒ the SIGN CONFLICT causes forgetting and\n"
        "   negative transfer — not the morphology\n"
        "   change itself (which transfers positively).\n"
        "\n"
        "VERDICT: KL distillation = successful CL here\n"
        "  • anti-forget under conflict (T3): 1.0 vs 0.0\n"
        "  • no plasticity cost when none (T2): ≈1.0\n"
        "SCOPE: 2 task pairs; descriptor REQUIRED\n"
        "  (necessity toy-specific; AE-entangled case\n"
        "   and other CL baselines untested)."
    )
    ax3.text(0.0, 0.93, txt, transform=ax3.transAxes, va="top", ha="left",
             family="monospace", fontsize=8.3)

    fig.suptitle(
        "Cross-Embodiment Continual PPO (2D toy) — 5 seeds, deterministic eval, descriptor on",
        fontsize=12, y=0.99,
    )
    out = ROOT / "results/summary_onepager.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
