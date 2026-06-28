"""Lightweight 2D grasping environments for continual PPO experiments.

The environment intentionally keeps physics simple. The point of this toy
project is to stress the continual-learning pipeline: T1 and T3 share the same
observation/action shape, while T3 flips the active action-to-effect mapping so
that fine-tuning has a real source of conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces


TASK_SEQUENCE = ("T1", "T3")


@dataclass(frozen=True)
class MorphologyTask:
    """Task-level morphology and action-effect configuration."""

    name: str
    mask: np.ndarray
    action_sign: np.ndarray
    scalars: np.ndarray
    finger_lengths: np.ndarray


def _arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float32)


TASKS: dict[str, MorphologyTask] = {
    # Descriptor scalars are a 2D morphology fingerprint: scalar[0] = finger-1
    # action sign/gain (resolves the T3 conflict), scalar[1] = finger-length
    # scale (distinguishes the T2 variant). The two task pairs span the
    # conflict spectrum: T1->T2 is medium-similarity / no-conflict (for the
    # forward-transfer story); T1<->T3 is maximum conflict (for the forgetting
    # story). T2 and T3 share the same morphology change relative to T1, so the
    # only thing that differs is the sign flip -> isolates conflict as the cause
    # of negative transfer.
    "T1": MorphologyTask(
        name="T1",
        mask=_arr([1.0, 1.0, 1.0]),
        action_sign=_arr([1.0, 1.0, 1.0]),
        scalars=_arr([1.0, 1.0]),
        finger_lengths=_arr([1.0, 1.0, 1.0]),
    ),
    "T2": MorphologyTask(
        name="T2",
        # 3-finger variant: shorter fingers (different reach), same action
        # directions as T1 -> medium similarity, no conflict, should transfer.
        mask=_arr([1.0, 1.0, 1.0]),
        action_sign=_arr([1.0, 1.0, 1.0]),
        scalars=_arr([1.0, 0.8]),
        finger_lengths=_arr([0.8, 0.8, 0.8]),
    ),
    "T3": MorphologyTask(
        name="T3",
        mask=_arr([1.0, 1.0, 0.0]),
        # Only finger 1 flips sign: finger 2 transfers from T1, so T3 stays
        # learnable from a T1 init while finger 1 is a genuine conflict.
        action_sign=_arr([-1.0, 1.0, 0.0]),
        scalars=_arr([-1.0, 1.0]),
        finger_lengths=_arr([1.0, 1.0, 0.0]),
    ),
    "T4": MorphologyTask(
        name="T4",
        mask=_arr([1.0, 1.0, 0.0]),
        action_sign=_arr([-0.75, 1.0, 0.0]),
        scalars=_arr([-0.75, 1.0]),
        finger_lengths=_arr([0.8, 1.1, 0.0]),
    ),
}


class ToyDexGraspEnv(gym.Env):
    """A compact continuous-control grasping task.

    Observation is always 15D:

    ``object_xy, palm_xy, fingertip1_xy, fingertip2_xy, fingertip3_xy,
    mask_123, morphology_scalar_12``.

    When ``use_morphology=False``, the final five descriptor dimensions are
    zeroed. The action sign flip is still active in the environment, so T1 and
    T3 require opposite finger-1 actions from overlapping initial observations.

    Known limitation: finger 3's tip is part of the observation and behaves
    differently across morphologies (it stays open under T3). A no-descriptor
    policy can therefore still partially infer the task from finger-3 motion,
    so the no-descriptor conflict is strong but not perfectly sealed.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        task: str = "T1",
        use_morphology: bool = True,
        max_steps: int = 64,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if task not in TASKS:
            raise ValueError(f"Unknown task {task!r}. Valid tasks: {sorted(TASKS)}")

        self.task_name = task
        self.task = TASKS[task]
        self.use_morphology = use_morphology
        self.max_steps = max_steps

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._base_object_xy = np.zeros(2, dtype=np.float32)
        self._lift = 0.0
        self._palm_xy = np.zeros(2, dtype=np.float32)
        self._closure = np.zeros(3, dtype=np.float32)
        self._stable_success_steps = 0
        self._last_action = np.zeros(3, dtype=np.float32)
        self.contact_threshold = 0.11
        self.success_lift_threshold = 0.16

        # Contact points sit on the cube faces (half-size 0.05): finger 1 on
        # the left face, finger 2 on the right, finger 3 on the top. Distance at
        # full closure is 0.05, comfortably under contact_threshold (0.11).
        self._contact_offsets = np.asarray(
            [[-0.05, 0.0], [0.05, 0.0], [0.0, 0.05]], dtype=np.float32
        )
        self._open_offsets = np.asarray(
            [[-0.22, -0.055], [0.22, -0.055], [0.0, 0.22]], dtype=np.float32
        )

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step = 0
        self._base_object_xy = self._rng.uniform(
            low=[-0.08, 0.56], high=[0.08, 0.72]
        ).astype(np.float32)
        palm_noise = self._rng.normal(loc=0.0, scale=[0.015, 0.01]).astype(np.float32)
        self._palm_xy = self._base_object_xy + np.asarray(
            [0.0, -0.16], dtype=np.float32
        ) + palm_noise

        # Overlapping initial observations make the no-descriptor conflict
        # concrete: T1 needs positive active actions, T3 needs negative ones.
        self._closure = self._rng.uniform(0.0, 0.08, size=3).astype(np.float32)
        self._closure *= self.task.mask
        self._lift = 0.0
        self._stable_success_steps = 0
        self._last_action = np.zeros(3, dtype=np.float32)

        obs = self._get_obs()
        return obs, self._info(success=False, reward_terms={})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        active_action = action * self.task.mask

        effect = self.task.action_sign * active_action
        self._closure = np.clip(self._closure + 0.09 * effect, 0.0, 1.05)
        self._closure *= self.task.mask

        reward_terms = self._reward_terms(action)
        # Gate lift behind the *weakest* active finger so the cube only rises
        # once every active finger is grasping. Gating on the mean contact
        # would let a partial 2/3 grasp harvest most of the lift reward and
        # never close the last finger (reward/success misalignment).
        grip = reward_terms["weakest_active_contact"]
        close_quality = reward_terms["close_quality"]
        lift_delta = 0.035 * grip * close_quality
        lift_decay = 0.004 * (1.0 - grip)
        self._lift = float(np.clip(self._lift + lift_delta - lift_decay, 0.0, 0.35))

        reward_terms = self._reward_terms(action)
        reward = (
            2.00 * reward_terms["lift"]
            + 1.25 * reward_terms["contact_bonus"]
            - 0.02 * reward_terms["action_smoothness"]
        )

        success = self._is_success()
        self._stable_success_steps = (
            self._stable_success_steps + 1 if success else 0
        )
        terminated = self._stable_success_steps >= 3
        self._step += 1
        truncated = self._step >= self.max_steps
        self._last_action = active_action

        return (
            self._get_obs(),
            float(reward),
            bool(terminated),
            bool(truncated),
            self._info(success=success, reward_terms=reward_terms),
        )

    def render(self) -> np.ndarray:
        """Return an RGB array of the current state (gym ``rgb_array`` mode).

        Drawing lives in ``continual_ppo.render`` (imported lazily) so the core
        environment stays free of a matplotlib import.
        """
        import matplotlib.pyplot as plt

        from continual_ppo.render import draw_state

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        draw_state(ax, self)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return rgb

    @property
    def object_xy(self) -> np.ndarray:
        return self._base_object_xy + np.asarray([0.0, self._lift], dtype=np.float32)

    @property
    def active_finger_count(self) -> float:
        return float(max(np.sum(self.task.mask), 1.0))

    def _finger_tips(self) -> np.ndarray:
        closure = np.clip(self._closure * self.task.finger_lengths, 0.0, 1.05)
        tips = []
        for idx in range(3):
            target = self.object_xy + self._contact_offsets[idx]
            open_pos = self.object_xy + self._open_offsets[idx]
            tip = target * closure[idx] + open_pos * (1.0 - closure[idx])
            if self.task.mask[idx] < 0.5:
                tip = self.object_xy + self._open_offsets[idx]
            tips.append(tip.astype(np.float32))
        return np.stack(tips, axis=0)

    def _get_obs(self) -> np.ndarray:
        fingertips = self._finger_tips().reshape(-1)
        if self.use_morphology:
            descriptor = np.concatenate([self.task.mask, self.task.scalars])
        else:
            descriptor = np.zeros(5, dtype=np.float32)
        obs = np.concatenate(
            [self.object_xy, self._palm_xy, fingertips, descriptor]
        ).astype(np.float32)
        if obs.shape != (15,):
            raise RuntimeError(f"Expected 15D observation, got {obs.shape}")
        return obs

    def _reward_terms(self, action: np.ndarray) -> dict[str, float]:
        # In the 2D toy, lift is a scalar success proxy rather than real
        # vertical geometry; it is accumulated in ``step`` only while the active
        # fingers are in contact (see the gating there).
        fingertips = self._finger_tips()
        distances = np.linalg.norm(fingertips - self.object_xy[None, :], axis=1)
        soft_contacts = np.exp(-((distances / 0.10) ** 2)).astype(np.float32)
        active_contacts = soft_contacts * self.task.mask
        active_soft = soft_contacts[self.task.mask > 0.5]
        weakest_active_contact = float(np.min(active_soft)) if active_soft.size else 0.0
        mean_active_contact = float(np.sum(active_contacts) / self.active_finger_count)
        active_closure = np.clip(self._closure, 0.0, 1.0)[self.task.mask > 0.5]
        weakest_active_closure = (
            float(np.min(active_closure)) if active_closure.size else 0.0
        )
        contact_bonus = 0.5 * weakest_active_contact + 0.5 * weakest_active_closure
        contact_fraction = float(
            np.sum((distances < self.contact_threshold).astype(np.float32) * self.task.mask)
            / self.active_finger_count
        )
        close_quality = float(
            np.sum(np.clip(self._closure, 0.0, 1.0) * self.task.mask)
            / self.active_finger_count
        )
        action_smoothness = float(np.mean((action * self.task.mask) ** 2))
        action_delta = float(np.mean(((action - self._last_action) * self.task.mask) ** 2))

        return {
            "contact_bonus": contact_bonus,
            "contact_fraction": contact_fraction,
            "close_quality": close_quality,
            "weakest_active_contact": weakest_active_contact,
            "weakest_active_closure": weakest_active_closure,
            "action_smoothness": action_smoothness,
            "action_delta": action_delta,
            "lift": float(self._lift),
        }

    def _is_success(self) -> bool:
        terms = self._reward_terms(self._last_action)
        return bool(
            terms["contact_fraction"] >= 0.999
            and self._lift >= self.success_lift_threshold
        )

    def _info(self, success: bool, reward_terms: dict[str, float]) -> dict:
        info = {
            "task": self.task_name,
            "use_morphology": self.use_morphology,
            "success": bool(success),
            "lift": float(self._lift),
            "closure": self._closure.copy(),
            "action_sign": self.task.action_sign.copy(),
            "mask": self.task.mask.copy(),
        }
        info.update(reward_terms)
        return info


def make_env(
    task: str,
    use_morphology: bool,
    seed: int,
    max_steps: int = 64,
) -> Callable[[], gym.Env]:
    """Factory compatible with Gymnasium vector environments."""

    def thunk() -> gym.Env:
        env = ToyDexGraspEnv(
            task=task, use_morphology=use_morphology, max_steps=max_steps, seed=seed
        )
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk
