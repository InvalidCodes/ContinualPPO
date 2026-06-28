#!/usr/bin/env python
"""Run the toy continual PPO experiment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from continual_ppo.experiment import main


if __name__ == "__main__":
    main()
