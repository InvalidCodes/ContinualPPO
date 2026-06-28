"""Lightweight 2D visualisation of the toy grasping environment.

Kept separate from ``envs.py`` so the environment itself stays free of any
plotting dependency. Used by ``scripts/render_env.py`` to produce filmstrip
previews of each morphology.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from continual_ppo.envs import ToyDexGraspEnv

_FINGER_COLORS = ("#ef4444", "#22c55e", "#3b82f6")
_FINGER_LABELS = ("finger 1", "finger 2", "finger 3")


def draw_state(ax, env: ToyDexGraspEnv, legend: bool = True) -> None:
    """Draw the current state of ``env`` onto a matplotlib Axes.

    Palm = hand base (grey bar); coloured dots = fingertips (one hand, up to 3
    fingers); a grey x marks an inactive finger; the yellow square is the cube.
    """

    obj = env.object_xy
    palm = env._palm_xy
    tips = env._finger_tips()
    mask = env.task.mask

    # Cube (semi-transparent so fingertips touching its faces stay visible).
    ax.add_patch(
        Rectangle((obj[0] - 0.05, obj[1] - 0.05), 0.10, 0.10,
                  facecolor="#fcd34d", edgecolor="#b45309", alpha=0.55, zorder=1)
    )
    # Palm as a short bar.
    ax.plot([palm[0] - 0.12, palm[0] + 0.12], [palm[1], palm[1]],
            color="#374151", linewidth=5, solid_capstyle="round", zorder=2,
            label="palm (hand base)")

    for idx in range(3):
        active = mask[idx] > 0.5
        color = _FINGER_COLORS[idx] if active else "#9ca3af"
        label = _FINGER_LABELS[idx] + ("" if active else " (inactive)")
        ax.plot([palm[0], tips[idx, 0]], [palm[1], tips[idx, 1]],
                color=color, linewidth=2, alpha=0.9 if active else 0.35, zorder=2)
        ax.scatter(tips[idx, 0], tips[idx, 1], s=70 if active else 60, color=color,
                   edgecolor="black" if active else None,
                   marker="o" if active else "x", zorder=3, label=label)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0.25, 1.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        ax.legend(loc="upper right", fontsize=6, framealpha=0.9)


def render_task_filmstrip(
    task: str,
    out_path: str | Path,
    frame_steps: tuple[int, ...] = (0, 5, 11, 22),
    use_morphology: bool = True,
    seed: int = 0,
) -> Path:
    """Roll out a scripted grasp and save a row of snapshots for one task."""

    env = ToyDexGraspEnv(task, use_morphology=use_morphology, seed=seed)
    env.reset(seed=seed)
    # Action that closes the active fingers under this morphology's sign map.
    close_action = np.sign(env.task.action_sign).astype(np.float32)

    fig, axes = plt.subplots(1, len(frame_steps), figsize=(3.0 * len(frame_steps), 3.2))
    info: dict = {}
    next_capture = 0
    mask = "".join(str(int(m)) for m in env.task.mask)
    # Keep stepping past success/termination so the filmstrip shows the full
    # close-then-lift trajectory rather than stopping at first success.
    for step in range(max(frame_steps) + 1):
        if step in frame_steps:
            ax = axes[next_capture]
            draw_state(ax, env, legend=(next_capture == 0))
            ax.set_title(
                f"t={step}  lift={env._lift:.2f}\n"
                f"success={info.get('success', False)}",
                fontsize=9,
            )
            next_capture += 1
        _, _, _, _, info = env.step(close_action)

    fig.suptitle(
        f"{task}  mask=[{mask}]  sign={env.task.action_sign.tolist()}", fontsize=11
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
