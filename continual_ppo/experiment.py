"""Experiment scheduler for the cross-embodiment continual PPO toy project."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import pandas as pd

from continual_ppo.envs import TASK_SEQUENCE, ToyDexGraspEnv, make_env
from continual_ppo.metrics import compute_learning_speed
from continual_ppo.plotting import aggregate_metrics, plot_matrix, plot_training_curves
from continual_ppo.ppo import (
    DistillReference,
    PPOAgent,
    PPOConfig,
    evaluate_agent,
    train_ppo,
)
from continual_ppo.state_buffer import StateBuffer


METHODS = ("single", "multitask", "finetune", "kl", "reset")


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def bool_flag(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean flag, got {value!r}")


def build_vector_env(
    task: str,
    use_morphology: bool,
    num_envs: int,
    seed: int,
    max_steps: int,
):
    env_fns = [
        make_env(task, use_morphology, seed + idx, max_steps=max_steps)
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def build_multitask_vector_env(
    tasks: tuple[str, ...],
    use_morphology: bool,
    num_envs: int,
    seed: int,
    max_steps: int,
):
    # Assign each parallel env a fixed task round-robin so every rollout batch
    # is balanced across morphologies. This is the standard low-variance way to
    # batch multi-task PPO; per-episode random task sampling made the joint
    # upper bound swing wildly across seeds.
    env_fns = [
        make_env(tasks[idx % len(tasks)], use_morphology, seed + idx, max_steps=max_steps)
        for idx in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(env_fns)


def eval_all(
    agent: PPOAgent,
    tasks: tuple[str, ...],
    use_morphology: bool,
    seed: int,
    episodes: int,
    max_steps: int,
    deterministic_eval: bool,
) -> dict[str, dict[str, float]]:
    results = {}
    for task in tasks:
        results[task] = evaluate_agent(
            agent,
            make_env(task, use_morphology, seed + 10_000, max_steps=max_steps),
            episodes=episodes,
            deterministic=deterministic_eval,
        )
    return results


def append_eval_rows(
    rows: list[dict],
    method: str,
    use_morphology: bool,
    seed: int,
    stage: str,
    results: dict[str, dict[str, float]],
) -> None:
    for eval_task, metrics in results.items():
        rows.append(
            {
                "method": method,
                "use_morphology": use_morphology,
                "seed": seed,
                "stage": stage,
                "eval_task": eval_task,
                **metrics,
            }
        )


def append_train_rows(
    rows: list[dict],
    method: str,
    use_morphology: bool,
    seed: int,
    stats,
) -> None:
    for stat in stats:
        rows.append(
            {
                "method": method,
                "use_morphology": use_morphology,
                "seed": seed,
                **dataclasses.asdict(stat),
            }
        )


def run_single_seed_method(
    method: str,
    seed: int,
    use_morphology: bool,
    cfg: PPOConfig,
    tasks: tuple[str, ...],
    out_dir: Path,
    eval_episodes: int,
    max_steps: int,
    deterministic_eval: bool,
) -> tuple[list[dict], list[dict]]:
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    obs_dim = ToyDexGraspEnv(tasks[0], use_morphology=use_morphology).observation_space.shape[0]
    action_dim = ToyDexGraspEnv(tasks[0], use_morphology=use_morphology).action_space.shape[0]
    run_tag = f"{method}_desc{int(use_morphology)}_seed{seed}"
    ckpt_dir = out_dir / "checkpoints" / run_tag

    if method == "single":
        for task_idx, task in enumerate(tasks):
            local_cfg = dataclasses.replace(cfg, seed=seed + task_idx)
            agent = PPOAgent(obs_dim, action_dim, local_cfg)
            envs = build_vector_env(
                task, use_morphology, local_cfg.num_envs, seed + task_idx * 1000, max_steps
            )
            buffer = StateBuffer(local_cfg.state_buffer_capacity, obs_dim, seed=seed)
            agent, stats = train_ppo(
                agent, envs, local_cfg, task, collect_state_buffer=buffer
            )
            envs.close()
            append_train_rows(train_rows, method, use_morphology, seed, stats)
            agent.save(ckpt_dir / f"{task}.pt", extra={"task": task})
            append_eval_rows(
                eval_rows,
                method,
                use_morphology,
                seed,
                f"after_{task}",
                {
                    task: eval_all(
                        agent,
                        (task,),
                        use_morphology,
                        seed,
                        eval_episodes,
                        max_steps,
                        deterministic_eval,
                    )[task]
                },
            )
        return eval_rows, train_rows

    if method == "reset":
        for task_idx, task in enumerate(tasks):
            local_cfg = dataclasses.replace(cfg, seed=seed + task_idx)
            agent = PPOAgent(obs_dim, action_dim, local_cfg)
            envs = build_vector_env(
                task, use_morphology, local_cfg.num_envs, seed + task_idx * 1000, max_steps
            )
            agent, stats = train_ppo(agent, envs, local_cfg, task)
            envs.close()
            append_train_rows(train_rows, method, use_morphology, seed, stats)
            agent.save(ckpt_dir / f"reset_{task}.pt", extra={"task": task})
            append_eval_rows(
                eval_rows,
                method,
                use_morphology,
                seed,
                f"after_reset_{task}",
                {
                    task: eval_all(
                        agent,
                        (task,),
                        use_morphology,
                        seed,
                        eval_episodes,
                        max_steps,
                        deterministic_eval,
                    )[task]
                },
            )
        return eval_rows, train_rows

    if method == "multitask":
        # Joint training sees every task at once, so give it the same total
        # budget as the sequential methods (which spend cfg.total_timesteps per
        # task). Otherwise the "upper bound" is starved relative to finetune/kl.
        mt_cfg = dataclasses.replace(
            cfg, seed=seed, total_timesteps=cfg.total_timesteps * len(tasks)
        )
        agent = PPOAgent(obs_dim, action_dim, mt_cfg)
        envs = build_multitask_vector_env(
            tasks, use_morphology, cfg.num_envs, seed, max_steps
        )
        agent, stats = train_ppo(agent, envs, mt_cfg, "joint")
        envs.close()
        append_train_rows(train_rows, method, use_morphology, seed, stats)
        agent.save(ckpt_dir / "joint.pt", extra={"tasks": tasks})
        append_eval_rows(
            eval_rows,
            method,
            use_morphology,
            seed,
            "after_joint",
            eval_all(
                agent,
                tasks,
                use_morphology,
                seed,
                eval_episodes,
                max_steps,
                deterministic_eval,
            ),
        )
        return eval_rows, train_rows

    if method not in {"finetune", "kl"}:
        raise ValueError(f"Unknown method {method!r}")

    agent = PPOAgent(obs_dim, action_dim, dataclasses.replace(cfg, seed=seed))
    global_step = 0
    distill_refs: list[DistillReference] = []
    for task_idx, task in enumerate(tasks):
        task_cfg = dataclasses.replace(cfg, seed=seed + task_idx)
        envs = build_vector_env(
            task, use_morphology, task_cfg.num_envs, seed + task_idx * 1000, max_steps
        )
        old_task_buffer = StateBuffer(
            task_cfg.state_buffer_capacity,
            obs_dim,
            seed=seed + task_idx * 17,
        )
        active_refs = distill_refs if method == "kl" and task_idx > 0 else []
        train_cfg = task_cfg if method == "kl" else dataclasses.replace(task_cfg, kl_coef=0.0)
        agent, stats = train_ppo(
            agent,
            envs,
            train_cfg,
            task,
            start_global_step=global_step,
            distill_refs=active_refs,
            collect_state_buffer=old_task_buffer,
        )
        envs.close()
        if stats:
            global_step = stats[-1].global_step
        append_train_rows(train_rows, method, use_morphology, seed, stats)

        agent.save(ckpt_dir / f"after_{task}.pt", extra={"task": task})
        stage = f"after_{task}"
        if task_idx == 0:
            append_eval_rows(
                eval_rows,
                method,
                use_morphology,
                seed,
                stage,
                eval_all(
                    agent,
                    (task,),
                    use_morphology,
                    seed,
                    eval_episodes,
                    max_steps,
                    deterministic_eval,
                ),
            )
        else:
            append_eval_rows(
                eval_rows,
                method,
                use_morphology,
                seed,
                stage,
                eval_all(
                    agent,
                    tasks[: task_idx + 1],
                    use_morphology,
                    seed,
                    eval_episodes,
                    max_steps,
                    deterministic_eval,
                ),
            )

        distill_refs.append(
            DistillReference(
                task=task,
                actor=agent.snapshot_actor(),
                states=old_task_buffer,
            )
        )
    return eval_rows, train_rows


def run_experiment(args: argparse.Namespace) -> Path:
    methods = parse_csv_strings(args.methods)
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise ValueError(f"Unknown methods {invalid}. Valid methods: {METHODS}")
    seeds = parse_csv_ints(args.seeds)
    descriptors = [bool_flag(item) for item in parse_csv_strings(args.use_morphology)]
    tasks = tuple(parse_csv_strings(args.tasks))
    if not tasks:
        tasks = TASK_SEQUENCE

    cfg = PPOConfig(
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        seed=seeds[0] if seeds else 0,
        kl_coef=args.kl_coef,
        distill_batch_size=args.distill_batch_size,
        state_buffer_capacity=args.state_buffer_capacity,
        log_std_init=args.log_std_init,
        cuda=args.cuda,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_eval_rows: list[dict] = []
    all_train_rows: list[dict] = []
    for use_morphology in descriptors:
        for method in methods:
            for seed in seeds:
                eval_rows, train_rows = run_single_seed_method(
                    method=method,
                    seed=seed,
                    use_morphology=use_morphology,
                    cfg=cfg,
                    tasks=tasks,
                    out_dir=out_dir,
                    eval_episodes=args.eval_episodes,
                    max_steps=args.max_steps,
                    deterministic_eval=not args.stochastic_eval,
                )
                all_eval_rows.extend(eval_rows)
                all_train_rows.extend(train_rows)

    metrics_path = out_dir / "metrics.csv"
    train_path = out_dir / "train_stats.csv"
    pd.DataFrame(all_eval_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(all_train_rows).to_csv(train_path, index=False)
    summary = aggregate_metrics(metrics_path, out_dir)
    plot_matrix(summary, out_dir / "success_matrix.png")
    plot_training_curves(train_path, out_dir / "training_curves.png")
    compute_learning_speed(train_path, out_dir)
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="finetune,kl")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--use-morphology", default="1")
    parser.add_argument("--tasks", default="T1,T3")
    parser.add_argument("--out-dir", default="results/toy_run")
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=6)
    parser.add_argument("--kl-coef", type=float, default=0.5)
    parser.add_argument("--log-std-init", type=float, default=-0.5)
    parser.add_argument("--distill-batch-size", type=int, default=256)
    parser.add_argument("--state-buffer-capacity", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument(
        "--stochastic-eval",
        action="store_true",
        help="Evaluate by sampling actions instead of using the Gaussian mean. "
        "Default is deterministic (mean-action) evaluation.",
    )
    parser.add_argument("--cuda", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    out_dir = run_experiment(args)
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
