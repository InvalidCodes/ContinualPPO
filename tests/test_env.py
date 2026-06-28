from __future__ import annotations

import numpy as np

from continual_ppo.envs import ToyDexGraspEnv


def test_observation_shape_and_descriptor_toggle() -> None:
    env_desc = ToyDexGraspEnv("T3", use_morphology=True, seed=0)
    obs_desc, _ = env_desc.reset()
    assert obs_desc.shape == (15,)
    assert np.allclose(obs_desc[10:13], [1.0, 1.0, 0.0])
    assert np.allclose(obs_desc[13:15], [-1.0, 1.0])

    env_no_desc = ToyDexGraspEnv("T3", use_morphology=False, seed=0)
    obs_no_desc, _ = env_no_desc.reset()
    assert obs_no_desc.shape == (15,)
    assert np.allclose(obs_no_desc[10:15], np.zeros(5))


def test_t3_action_remap_conflict() -> None:
    t1 = ToyDexGraspEnv("T1", use_morphology=False, seed=123)
    t3 = ToyDexGraspEnv("T3", use_morphology=False, seed=123)
    t1.reset()
    t3.reset()

    positive_action = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    negative_action = -positive_action

    _, _, _, _, info_t1_pos = t1.step(positive_action)
    _, _, _, _, info_t3_pos = t3.step(positive_action)
    assert info_t1_pos["closure"][0] > 0.05
    assert info_t3_pos["closure"][0] <= 0.08

    t3.reset(seed=123)
    _, _, _, _, info_t3_neg = t3.step(negative_action)
    assert info_t3_neg["closure"][0] > info_t3_pos["closure"][0]


def test_active_only_contact_and_success() -> None:
    env = ToyDexGraspEnv("T3", use_morphology=True, seed=4)
    env.reset()
    done = False
    info = {}
    # T3 sign is [-1, 1, 0]: finger 1 closes on -1, finger 2 closes on +1.
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(
            np.asarray([-1.0, 1.0, 1.0], dtype=np.float32)
        )
        done = terminated or truncated
        if done:
            break
    assert info["mask"][2] == 0.0
    assert info["contact_fraction"] <= 1.0
    assert info["success"] is True
