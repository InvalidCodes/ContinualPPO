"""CleanRL-style PPO with optional closed-form Gaussian KL distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from continual_ppo.state_buffer import StateBuffer


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.distributions import Normal
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for PPO training. Install dependencies with "
            "`python -m pip install -r requirements.txt` from the project root."
        ) from exc
    return torch, nn, Normal


@dataclass
class PPOConfig:
    total_timesteps: int = 50_000
    learning_rate: float = 3e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 6
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    hidden_sizes: tuple[int, ...] = (64, 64)
    log_std_init: float = -0.5
    seed: int = 0
    torch_deterministic: bool = True
    cuda: bool = False
    kl_coef: float = 0.0
    distill_batch_size: int = 256
    state_buffer_capacity: int = 20_000

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


@dataclass
class TrainStats:
    task: str
    global_step: int
    update: int
    episodic_return: float | None = None
    episodic_success: float | None = None
    policy_loss: float | None = None
    value_loss: float | None = None
    entropy: float | None = None
    approx_kl: float | None = None
    clipfrac: float | None = None
    distill_kl: float | None = None
    distill_std_kl: float | None = None
    log_std_mean: float | None = None


class Actor:
    """Factory wrapper so torch is imported lazily."""

    @staticmethod
    def build(
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...],
        log_std_init: float = 0.0,
    ):
        torch, nn, Normal = _require_torch()

        class _Actor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                in_dim = obs_dim
                for hidden in hidden_sizes:
                    layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
                    in_dim = hidden
                layers.append(nn.Linear(in_dim, action_dim))
                self.net = nn.Sequential(*layers)
                self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))
                self.apply(_orthogonal_init)

            def forward(self, obs):
                return self.net(obs)

            def dist(self, obs):
                mean = self.forward(obs)
                std = torch.exp(self.log_std).expand_as(mean)
                return Normal(mean, std)

            def clone_frozen(self):
                clone = _Actor()
                clone.load_state_dict(self.state_dict())
                clone.eval()
                for param in clone.parameters():
                    param.requires_grad_(False)
                return clone

        return _Actor()


class Critic:
    """Factory wrapper so torch is imported lazily."""

    @staticmethod
    def build(obs_dim: int, hidden_sizes: tuple[int, ...]):
        torch, nn, _ = _require_torch()

        class _Critic(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                layers: list[nn.Module] = []
                in_dim = obs_dim
                for hidden in hidden_sizes:
                    layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
                    in_dim = hidden
                layers.append(nn.Linear(in_dim, 1))
                self.net = nn.Sequential(*layers)
                self.apply(_orthogonal_init)

            def forward(self, obs):
                return self.net(obs).squeeze(-1)

        return _Critic()


def _orthogonal_init(module: Any) -> None:
    torch, nn, _ = _require_torch()
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.constant_(module.bias, 0.0)


@dataclass
class DistillReference:
    task: str
    actor: Any
    states: StateBuffer


@dataclass
class PPOAgent:
    obs_dim: int
    action_dim: int
    config: PPOConfig
    actor: Any = field(init=False)
    critic: Any = field(init=False)
    optimizer: Any = field(init=False)
    device: Any = field(init=False)

    def __post_init__(self) -> None:
        torch, _, _ = _require_torch()
        self.device = torch.device(
            "cuda" if self.config.cuda and torch.cuda.is_available() else "cpu"
        )
        torch.manual_seed(self.config.seed)
        if self.config.torch_deterministic:
            torch.backends.cudnn.deterministic = True
        self.actor = Actor.build(
            self.obs_dim,
            self.action_dim,
            self.config.hidden_sizes,
            self.config.log_std_init,
        ).to(self.device)
        self.critic = Critic.build(self.obs_dim, self.config.hidden_sizes).to(
            self.device
        )
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.config.learning_rate,
            eps=1e-5,
        )

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        torch, _, _ = _require_torch()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            dist = self.actor.dist(obs_t)
            action = dist.mean if deterministic else dist.sample()
        return action.cpu().numpy()

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch, _, _ = _require_torch()
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "extra": extra or {},
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, config_override: PPOConfig | None = None):
        torch, _, _ = _require_torch()
        payload = torch.load(path, map_location="cpu")
        config = config_override or PPOConfig(**payload["config"])
        agent = cls(payload["obs_dim"], payload["action_dim"], config)
        agent.actor.load_state_dict(payload["actor"])
        agent.critic.load_state_dict(payload["critic"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        return agent, payload.get("extra", {})

    def snapshot_actor(self):
        actor = self.actor.clone_frozen().to(self.device)
        actor.eval()
        return actor


def diagonal_gaussian_kl(old_actor, new_actor, obs, direction: str = "forward"):
    """Closed-form KL between two diagonal Gaussian actors.

    ``direction='forward'`` computes KL(old || new). Both means and standard
    deviations contribute; this is intentionally not MSE-on-means.
    """

    torch, _, _ = _require_torch()
    with torch.no_grad():
        old_mean = old_actor(obs)
        old_log_std = old_actor.log_std.expand_as(old_mean)
    new_mean = new_actor(obs)
    new_log_std = new_actor.log_std.expand_as(new_mean)

    if direction == "reverse":
        mean_p, log_std_p = new_mean, new_log_std
        mean_q, log_std_q = old_mean.detach(), old_log_std.detach()
    elif direction == "forward":
        mean_p, log_std_p = old_mean.detach(), old_log_std.detach()
        mean_q, log_std_q = new_mean, new_log_std
    else:
        raise ValueError("direction must be 'forward' or 'reverse'")

    var_p = torch.exp(2.0 * log_std_p)
    var_q = torch.exp(2.0 * log_std_q)
    mean_term = ((mean_q - mean_p) ** 2) / (2.0 * var_q)
    std_term = log_std_q - log_std_p + (var_p / (2.0 * var_q)) - 0.5
    per_dim = std_term + mean_term
    return per_dim.sum(dim=-1).mean(), std_term.sum(dim=-1).mean()


def train_ppo(
    agent: PPOAgent,
    envs,
    config: PPOConfig,
    task_name: str,
    start_global_step: int = 0,
    distill_refs: list[DistillReference] | None = None,
    collect_state_buffer: StateBuffer | None = None,
) -> tuple[PPOAgent, list[TrainStats]]:
    """Train PPO on a Gymnasium vector env."""

    torch, _, _ = _require_torch()
    np.random.seed(config.seed)
    distill_refs = distill_refs or []
    device = agent.device

    obs = torch.zeros((config.num_steps, config.num_envs, agent.obs_dim), device=device)
    actions = torch.zeros(
        (config.num_steps, config.num_envs, agent.action_dim), device=device
    )
    logprobs = torch.zeros((config.num_steps, config.num_envs), device=device)
    rewards = torch.zeros((config.num_steps, config.num_envs), device=device)
    dones = torch.zeros((config.num_steps, config.num_envs), device=device)
    values = torch.zeros((config.num_steps, config.num_envs), device=device)

    next_obs_np, _ = envs.reset(seed=config.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

    num_updates = max(config.total_timesteps // config.batch_size, 1)
    global_step = start_global_step
    stats: list[TrainStats] = []
    latest_return: float | None = None
    latest_success: float | None = None

    for update in range(1, num_updates + 1):
        if config.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lr_now = frac * config.learning_rate
            for param_group in agent.optimizer.param_groups:
                param_group["lr"] = lr_now

        for step in range(config.num_steps):
            global_step += config.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                dist = agent.actor.dist(next_obs)
                action = dist.sample()
                logprob = dist.log_prob(action).sum(dim=-1)
                value = agent.critic(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            clipped_action = torch.clamp(action, -1.0, 1.0).cpu().numpy()
            next_obs_np, reward_np, term_np, trunc_np, infos = envs.step(clipped_action)
            done_np = np.logical_or(term_np, trunc_np)
            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.as_tensor(done_np, dtype=torch.float32, device=device)

            if collect_state_buffer is not None:
                collect_state_buffer.add(obs[step].detach().cpu().numpy())

            if "episode" in infos and "_episode" in infos:
                episode_mask = np.asarray(infos["_episode"], dtype=bool)
                if episode_mask.any():
                    episode = infos["episode"]
                    returns = np.asarray(episode["r"], dtype=np.float32)[episode_mask]
                    success_values = np.asarray(
                        infos.get("success", np.zeros(config.num_envs)),
                        dtype=np.float32,
                    )[episode_mask]
                    latest_return = float(np.mean(returns))
                    latest_success = float(np.mean(success_values))
            elif "final_info" in infos:
                returns = []
                successes = []
                for info in infos["final_info"]:
                    if info is None:
                        continue
                    episode = info.get("episode")
                    if episode is not None:
                        returns.append(float(episode["r"]))
                    successes.append(float(info.get("success", False)))
                if returns:
                    latest_return = float(np.mean(returns))
                if successes:
                    latest_success = float(np.mean(successes))

        with torch.no_grad():
            next_value = agent.critic(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + config.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1, agent.obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, agent.action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(config.batch_size)
        clipfracs = []
        last_policy_loss = None
        last_value_loss = None
        last_entropy = None
        last_approx_kl = None
        last_distill_kl = None
        last_distill_std_kl = None

        for _epoch in range(config.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
                mb_inds = b_inds[start:end]

                dist = agent.actor.dist(b_obs[mb_inds])
                newlogprob = dist.log_prob(b_actions[mb_inds]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                newvalue = agent.critic(b_obs[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > config.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.clip_coef,
                        config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                distill_kl = torch.zeros((), dtype=torch.float32, device=device)
                distill_std_kl = torch.zeros((), dtype=torch.float32, device=device)
                if config.kl_coef > 0.0 and distill_refs:
                    for ref in distill_refs:
                        sample_np = ref.states.sample(config.distill_batch_size)
                        sample = torch.as_tensor(
                            sample_np, dtype=torch.float32, device=device
                        )
                        ref_kl, ref_std_kl = diagonal_gaussian_kl(
                            ref.actor, agent.actor, sample, direction="forward"
                        )
                        distill_kl = distill_kl + ref_kl / len(distill_refs)
                        distill_std_kl = distill_std_kl + ref_std_kl / len(distill_refs)

                loss = (
                    pg_loss
                    - config.ent_coef * entropy
                    + config.vf_coef * v_loss
                    + config.kl_coef * distill_kl
                )

                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.actor.parameters()) + list(agent.critic.parameters()),
                    config.max_grad_norm,
                )
                agent.optimizer.step()

                last_policy_loss = float(pg_loss.detach().cpu())
                last_value_loss = float(v_loss.detach().cpu())
                last_entropy = float(entropy.detach().cpu())
                last_approx_kl = float(approx_kl.detach().cpu())
                last_distill_kl = float(distill_kl.detach().cpu())
                last_distill_std_kl = float(distill_std_kl.detach().cpu())

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        stats.append(
            TrainStats(
                task=task_name,
                global_step=global_step,
                update=update,
                episodic_return=latest_return,
                episodic_success=latest_success,
                policy_loss=last_policy_loss,
                value_loss=last_value_loss,
                entropy=last_entropy,
                approx_kl=last_approx_kl,
                clipfrac=float(np.mean(clipfracs)) if clipfracs else None,
                distill_kl=last_distill_kl,
                distill_std_kl=last_distill_std_kl,
                log_std_mean=float(agent.actor.log_std.detach().mean().cpu()),
            )
        )

    return agent, stats


def evaluate_agent(
    agent: PPOAgent,
    env_factory,
    episodes: int = 20,
    deterministic: bool = False,
) -> dict[str, float]:
    env = env_factory()
    successes = []
    returns = []
    lengths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_len = 0
        last_success = False
        while not done:
            action = agent.act(obs[None, :], deterministic=deterministic)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            episode_len += 1
            last_success = bool(info.get("success", False)) or last_success
            done = bool(terminated or truncated)
        successes.append(float(last_success))
        returns.append(episode_return)
        lengths.append(episode_len)
    env.close()
    return {
        "success_rate": float(np.mean(successes)),
        "return_mean": float(np.mean(returns)),
        "episode_length_mean": float(np.mean(lengths)),
    }
