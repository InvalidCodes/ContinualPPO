#!/usr/bin/env python
"""Render filmstrip previews of each toy morphology to results/env_preview/."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_ppo.envs import TASKS
from continual_ppo.render import render_task_filmstrip


def main() -> None:
    out_dir = ROOT / "results" / "env_preview"
    for task in ("T1", "T3"):
        if task not in TASKS:
            continue
        path = render_task_filmstrip(task, out_dir / f"{task}.png")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
