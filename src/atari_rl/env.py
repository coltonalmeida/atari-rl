"""Environment construction. All wrapper logic lives here — never duplicate it elsewhere.

Uses SB3's make_atari_env, which applies the standard Atari preprocessing stack:
grayscale, resize to 84x84, frame-skip (max-pooled), episodic life handling,
and reward clipping during training. Frame-stacking is added on top.
"""

from __future__ import annotations

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

gym.register_envs(ale_py)

ENV_ID = "ALE/SpaceInvaders-v5"


def make_env(
    env_id: str = ENV_ID,
    n_envs: int = 1,
    seed: int | None = None,
    frame_stack: int = 4,
    eval_mode: bool = False,
):
    """Build a vectorized, preprocessed Atari env ready for SB3 CnnPolicy.

    Returns a VecEnv with observations of shape (frame_stack, 84, 84).
    eval_mode disables reward clipping and life-loss termination so episode
    rewards reflect the true game score.
    """
    wrapper_kwargs = {"clip_reward": False, "terminal_on_life_loss": False} if eval_mode else None
    # SubprocVecEnv runs each env in its own process so multiple CPU cores are used during
    # training (n_envs > 1). DummyVecEnv (sequential, single core) is fine for n_envs == 1.
    vec_env_cls = SubprocVecEnv if n_envs > 1 else None
    vec_env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        wrapper_kwargs=wrapper_kwargs,
        vec_env_cls=vec_env_cls,
    )
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)
    vec_env = VecTransposeImage(vec_env)
    return vec_env


def make_render_env(env_id: str = ENV_ID, seed: int | None = None):
    """Single raw env with rgb_array rendering, for video recording."""
    env = gym.make(env_id, render_mode="rgb_array")
    if seed is not None:
        env.reset(seed=seed)
    return env
