"""Evaluate a trained model or a random-action baseline over N episodes.

Usage:
    python -m atari_rl.evaluate --random --episodes 30 --out reports/random_baseline.json
    python -m atari_rl.evaluate --model models/ppo_best/best_model.zip --episodes 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atari_rl.env import ENV_ID, make_env

DEFAULT_BASELINE = "reports/random_baseline.json"


def load_baseline(path: str) -> dict | None:
    """Load a previously recorded random-agent baseline, if present."""
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def run_random_baseline(env_id: str, episodes: int, seed: int = 42) -> tuple[float, float]:
    """Random-agent sanity check: env builds, steps, and yields a baseline reward."""
    env = make_env(env_id, n_envs=1, seed=seed, eval_mode=True)
    rng = np.random.default_rng(seed)
    rewards = []
    for _ in range(episodes):
        env.reset()
        total, done = 0.0, False
        while not done:
            action = np.array([rng.integers(env.action_space.n)])
            _, reward, dones, _ = env.step(action)
            total += float(reward[0])
            done = bool(dones[0])
        rewards.append(total)
    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))


def run_model(model_path: str, env_id: str, episodes: int, seed: int = 42) -> tuple[float, float]:
    from stable_baselines3.common.evaluation import evaluate_policy

    from atari_rl.train import load_model

    env = make_env(env_id, n_envs=1, seed=seed, eval_mode=True)
    model = load_model(model_path, env)
    mean, std = evaluate_policy(model, env, n_eval_episodes=episodes)
    env.close()
    return float(mean), float(std)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default=None, help="Path to a saved SB3 model .zip")
    parser.add_argument("--random", action="store_true", help="Evaluate a random-action baseline")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE,
        help="Random-baseline JSON to compare a model against (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.random == (args.model is not None):
        parser.error("Provide exactly one of --random or --model")

    if args.random:
        label = "random"
        mean, std = run_random_baseline(args.env_id, args.episodes, args.seed)
    else:
        label = args.model
        mean, std = run_model(args.model, args.env_id, args.episodes, args.seed)

    result = {"agent": label, "episodes": args.episodes, "mean_reward": mean, "std_reward": std}
    print(json.dumps(result, indent=2))

    if not args.random:
        baseline = load_baseline(args.baseline)
        if baseline is None:
            print(f"\n(no baseline found at {args.baseline} — skipping comparison)")
        else:
            base_mean = baseline["mean_reward"]
            factor = mean / base_mean if base_mean else float("inf")
            print(
                f"\nvs random baseline ({args.baseline}):\n"
                f"  random: {base_mean:.1f} ± {baseline['std_reward']:.1f} "
                f"({baseline['episodes']} eps)\n"
                f"  model:  {mean:.1f} ± {std:.1f} ({args.episodes} eps)\n"
                f"  improvement: {factor:.2f}x ({mean - base_mean:+.1f} reward)"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
