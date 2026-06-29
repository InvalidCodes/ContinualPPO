# Results layout

Each directory is one study. The **integrated** result is `sequence_t1t2t3`;
the others isolate a single phenomenon (kept separate on purpose, not by accident).

| dir | study | what it answers |
|---|---|---|
| `sequence_t1t2t3/` | **T1→T2→T3 continual chain**, 5 methods × {desc 0,1} × 5 seeds | **integrated headline**: after the full chain, does KL retain both old tasks and learn the new one? (KL desc=1: T1 1.00 / T2 1.00 / T3 0.93; finetune ~0.60 on all) |
| `main_5seed/` | T1↔T3 only (max-conflict pair), 5 seeds | stability in isolation + starting point for the λ sweep |
| `plasticity_t1t2/` | T1→T2 only (no-conflict pair), 5 seeds | plasticity in isolation: forward transfer (+0.30, ~12× faster than scratch) |
| `multitask_10seed/` | joint training only, 10 seeds × 2× budget | is the "upper bound" stable? (no: desc=1 T1 0.64±0.40 — joint training is itself degraded by conflict) |
| `lambda_sweep/` | KL coefficient sweep, desc=0 | stability-plasticity frontier vs λ (no sweet spot without a descriptor) → `frontier.png` |
| `env_preview/` | rendered morphologies | sanity-check the 2D toy geometry (`T1.png`, `T3.png`) |
| `summary_onepager.png` | — | one-page summary (regenerate with `scripts/make_summary.py`) |

Each study dir contains: `summary.csv` (mean±std), `metrics.csv` (per-seed),
`train_stats.csv`, `learning_speed.csv` (AUC / time-to-threshold),
`forward_transfer.csv`, `success_matrix.png`, `training_curves.png`, `checkpoints/`.
