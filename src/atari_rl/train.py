"""Config-driven training entry point. Every run is driven by a Hydra config.

Usage:
    python -m atari_rl.train --config-name=ppo
    python -m atari_rl.train --config-name=ppo total_timesteps=10000 wandb.enabled=false
"""

from __future__ import annotations

import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from atari_rl.callbacks import build_callbacks
from atari_rl.env import make_env

ALGOS = {"ppo": PPO}


def linear_schedule(initial: float):
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return schedule


def load_model(model_path: str, env):
    """Load a saved SB3 model, inferring the algorithm from the zip."""
    for algo_cls in ALGOS.values():
        try:
            return algo_cls.load(model_path, env=env)
        except (ValueError, KeyError, AssertionError):
            continue
    raise ValueError(f"Could not load {model_path} as any of {list(ALGOS)}")


def _run_name_from_checkpoint(resume_from: str) -> str:
    """Recover the original run dir name (models/<run_name>/...) from a checkpoint path.

    Lets a resumed run drop its new checkpoints alongside the originals so the
    full learning timelapse stays in one place.
    """
    run_dir = Path(resume_from).resolve().parent
    while run_dir.name in {"checkpoints", "best"}:
        run_dir = run_dir.parent
    return run_dir.name


def train(config: dict) -> Path:
    algo_name = config["algo"].lower()
    resume_from = config.get("resume_from")

    if resume_from:
        run_name = _run_name_from_checkpoint(resume_from)
    else:
        run_name = f"{algo_name}_{int(time.time())}"

    env = make_env(
        config["env_id"],
        n_envs=config["n_envs"],
        seed=config["seed"],
        frame_stack=config.get("frame_stack", 4),
    )

    hyperparams = dict(config["hyperparams"])
    if algo_name == "ppo" and "learning_rate" in hyperparams:
        hyperparams["learning_rate"] = linear_schedule(float(hyperparams["learning_rate"]))

    wandb_run = None
    callbacks = [build_callbacks(config, run_name)]
    if config.get("wandb", {}).get("enabled", False):
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            name=run_name,
            config=config,
            sync_tensorboard=True,
        )
        callbacks.append(WandbCallback())

    total_timesteps = int(config["total_timesteps"])

    if resume_from:
        # Restore policy weights + optimizer state from the checkpoint and pick up
        # where the run stopped. reset_num_timesteps=False keeps the global step
        # counter, so checkpoint filenames and the LR schedule continue correctly.
        model = ALGOS[algo_name].load(resume_from, env=env)
        model.tensorboard_log = f"runs/{run_name}"
        remaining = total_timesteps - model.num_timesteps
        if remaining <= 0:
            print(
                f"Checkpoint is already at {model.num_timesteps} steps "
                f"(>= total_timesteps={total_timesteps}); nothing to train."
            )
            env.close()
            if wandb_run is not None:
                wandb_run.finish()
            return Path("models") / run_name / "final_model"
        print(
            f"Resuming {run_name} from {resume_from} at {model.num_timesteps} steps; "
            f"training {remaining} more to reach {total_timesteps}."
        )
        model.learn(total_timesteps=remaining, callback=callbacks, reset_num_timesteps=False)
    else:
        model = ALGOS[algo_name](
            config["policy"],
            env,
            seed=config["seed"],
            tensorboard_log=f"runs/{run_name}",
            verbose=1,
            **hyperparams,
        )
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

    final_path = Path("models") / run_name / "final_model"
    final_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_path)

    env.close()
    if wandb_run is not None:
        wandb_run.finish()
    return final_path


@hydra.main(version_base=None, config_path="../../configs", config_name="ppo")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True)
    path = train(config)
    print(f"Saved final model to {path}.zip")


if __name__ == "__main__":
    main()
