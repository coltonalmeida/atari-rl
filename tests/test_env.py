"""Env construction tests: build, reset/step shapes, and determinism basics."""

import sys

import numpy as np
import pytest

from atari_rl.env import ENV_ID, make_env

FRAME_STACK = 4


@pytest.fixture(scope="module")
def env():
    e = make_env(ENV_ID, n_envs=1, seed=0, frame_stack=FRAME_STACK)
    yield e
    e.close()


def test_observation_space_shape(env):
    assert env.observation_space.shape == (FRAME_STACK, 84, 84)


def test_action_space_is_discrete(env):
    assert env.action_space.n >= 4  # Space Invaders has 6 actions


def test_reset_shape(env):
    obs = env.reset()
    assert obs.shape == (1, FRAME_STACK, 84, 84)
    assert obs.dtype == np.uint8


def test_step_shapes(env):
    env.reset()
    obs, reward, dones, infos = env.step(np.array([0]))
    assert obs.shape == (1, FRAME_STACK, 84, 84)
    assert reward.shape == (1,)
    assert dones.shape == (1,)
    assert isinstance(infos, (list, tuple)) and len(infos) == 1


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="n_envs>1 uses SubprocVecEnv, which spawns on Windows; ALE env registration "
    "doesn't propagate to spawned workers. Training and CI run on Linux (fork).",
)
def test_vectorized_envs():
    e = make_env(ENV_ID, n_envs=2, seed=0)
    try:
        obs = e.reset()
        assert obs.shape == (2, FRAME_STACK, 84, 84)
    finally:
        e.close()
