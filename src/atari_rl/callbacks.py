"""Evaluation + checkpoint callbacks shared by all training runs."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from atari_rl.env import make_env


def build_callbacks(config: dict, run_name: str) -> CallbackList:
    """Standard callbacks: periodic eval (saves best model) + periodic checkpoints."""
    models_dir = Path("models") / run_name
    eval_cfg = config["eval"]

    eval_env = make_env(
        config["env_id"],
        n_envs=1,
        seed=config["seed"] + 1000,
        frame_stack=config.get("frame_stack", 4),
        eval_mode=True,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(models_dir / "best"),
        log_path=str(models_dir / "eval_logs"),
        eval_freq=eval_cfg["freq"],
        n_eval_episodes=eval_cfg["episodes"],
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=str(models_dir / "checkpoints"),
        name_prefix=config["algo"],
    )
    return CallbackList([eval_callback, checkpoint_callback])
