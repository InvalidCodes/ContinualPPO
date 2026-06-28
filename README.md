# Cross-Embodiment Continual PPO Toy Project

This directory implements the v1.2 plan in `plan.md` as a lightweight 2D toy
benchmark:

- fixed 15D observation and 3D continuous action;
- T1 base 3-finger task and T3 2-finger task;
- T3 deactivates finger 3 and flips **only finger 1's** action-to-effect sign
  (finger 2 transfers), so T3 conflicts with T1 yet stays learnable from a T1 init;
- lift is gated behind the weakest active finger's contact, so success
  (all active fingers grasping + cube lifted) aligns with the dense reward;
- morphology descriptor can be enabled or zeroed (the `use_morphology` switch);
- separated actor/critic PPO with a configurable `--log-std-init`;
- closed-form diagonal Gaussian KL distillation with actor snapshots and old-state buffers;
- `single`, `multitask`, `finetune`, `kl`, and `reset` experiment methods;
- CSV summaries and plots with seed aggregation, plus learning-speed metrics
  (AUC, time-to-threshold);
- **deterministic (mean-action) evaluation by default**; pass `--stochastic-eval`
  to sample from the Gaussian policy instead;
- 2D environment rendering: `env.render()` (`rgb_array`) and
  `scripts/render_env.py` for per-morphology filmstrip previews.

## Install

```bash
cd /home/ge/Desktop/ContinualPPO
python -m pip install -r requirements.txt
```

## Environment Smoke Test

```bash
python -m pytest tests/test_env.py
```

## Short Training Smoke Run

```bash
python scripts/run_experiment.py \
  --methods finetune,kl \
  --seeds 0 \
  --use-morphology 1 \
  --total-timesteps 4096 \
  --num-envs 4 \
  --num-steps 64 \
  --eval-episodes 5 \
  --out-dir results/smoke
```

This verifies the full T1 -> T3 pipeline and should produce nonzero
`distill_kl` values during the KL method's T3 stage. It is too short to judge
success rates.

## Single-Seed First-Pass Matrix

```bash
python scripts/run_experiment.py \
  --methods finetune,kl,multitask,reset \
  --seeds 0 \
  --use-morphology 1,0 \
  --total-timesteps 20000 \
  --eval-episodes 20 \
  --out-dir results/first_pass_seed0
```

Use this as a pipeline sanity check only. The project plan's statistical claim
requires 3-5 seeds and enough steps for both T1 and T3 to converge.

## Full First-Pass Matrix

```bash
python scripts/run_experiment.py \
  --methods finetune,kl,multitask,reset,single \
  --seeds 0,1,2,3,4 \
  --use-morphology 1,0 \
  --total-timesteps 30000 \
  --kl-coef 1.0 \
  --log-std-init -0.5 \
  --out-dir results/full_t1_t3
```

Evaluation uses the deterministic (mean) action by default; pass
`--stochastic-eval` to sample from the learned PPO policy instead.

Outputs:

- `metrics.csv`: one row per method/descriptor/seed/stage/eval task;
- `train_stats.csv`: PPO update diagnostics, including distillation KL and `log_std`;
- `summary.csv`: mean +/- std aggregation across seeds;
- `learning_speed.csv`: per-task AUC, time-to-first-success, time-to-0.8 success
  (forward-transfer evidence when final success saturates);
- `success_matrix.png`: evaluation matrix plot;
- `training_curves.png`: per-task aggregated training success curves;
- `checkpoints/`: actor/critic/optimizer checkpoints after each stage.

## Environment Preview

```bash
python scripts/render_env.py   # writes results/env_preview/{T1,T3}.png
```
