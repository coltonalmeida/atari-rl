"""Record a gameplay video of a trained agent, optionally uploading it to W&B.

Usage:
    python -m atari_rl.record --model models/ppo_best/best_model.zip
    python -m atari_rl.record --model checkpoints/ppo_500000_steps.zip \\
        --step 500000 --wandb --wandb-project atari-rl
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import gymnasium as gym
import imageio_ffmpeg
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from atari_rl.env import ENV_ID
from atari_rl.train import load_model


def _make_recorded_env(env_id: str, out_dir: str, episodes: int):
    """Raw env wrapped with RecordVideo (full-color frames), then Atari preprocessing
    so the model sees the same observations as during training.

    The visual env runs with frameskip=1 so RecordVideo captures *every* emulator frame.
    Atari games (Space Invaders, Asteroids, …) draw some sprites on alternating frames;
    sampling only one frame per skip lands on a single flicker phase, so those sprites never
    make it into the video. Capturing every frame keeps every phase (see deflicker()).
    AtariWrapper(skip=4) still gives the model the correct max-pooled observations.

    terminal_on_life_loss is disabled so the recording shows a full multi-life game
    (a compelling demo) rather than ending at the first life lost. Only the first
    `episodes` episodes are recorded, so the vec-env auto-reset after the final
    episode doesn't leave a spurious near-empty clip behind.
    """

    def _thunk():
        try:
            env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
        except (TypeError, ValueError, gym.error.Error):
            # Some env ids reject a frameskip override; fall back to the no-frameskip ROM.
            env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
        env = RecordVideo(env, video_folder=out_dir, episode_trigger=lambda ep: ep < episodes)
        return AtariWrapper(env, terminal_on_life_loss=False)

    return _thunk


def deflicker(video_path: Path) -> None:
    """Replace each frame with the per-pixel max ("lighten") over a short sliding window of
    consecutive frames, so Atari's sprite flicker disappears in the recording. A sprite
    drawn in any frame of the window stays visible at full brightness — the video analogue
    of the max-pooling the training preprocessing already applies to the agent's
    observations.

    Three chained tblend filters give a bounded 4-frame max window, which clears the
    2-to-4-phase flicker heavier scenes produce (a single 2-frame blend was not enough)
    while keeping motion trails to ~3 frames.
    """
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = video_path.with_suffix(".deflicker.mp4")
    lighten = "tblend=all_mode=lighten"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"{lighten},{lighten},{lighten}",
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ],
        check=True,
        capture_output=True,
    )
    tmp.replace(video_path)


def record(
    model_path: str,
    env_id: str,
    out_dir: str,
    episodes: int,
    seed: int,
    deflicker_enabled: bool = True,
) -> list[Path]:
    vec = DummyVecEnv([_make_recorded_env(env_id, out_dir, episodes)])
    vec.seed(seed)
    vec = VecFrameStack(vec, n_stack=4)
    vec = VecTransposeImage(vec)
    model = load_model(model_path, vec)

    for _ in range(episodes):
        obs = vec.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec.step(np.asarray(action))
            done = bool(dones[0])
    vec.close()

    videos = sorted(Path(out_dir).glob("*.mp4"))
    if deflicker_enabled:
        for video in videos:
            deflicker(video)
    print(f"Saved {len(videos)} video(s) to {Path(out_dir).resolve()}")
    return videos


def upload_to_wandb(videos: list[Path], project: str, step: int) -> None:
    """Log recorded gameplay clips to W&B as video artifacts (see CLAUDE.md §7)."""
    if not videos:
        print("No videos to upload.")
        return
    import wandb

    run = wandb.init(project=project, job_type="record", config={"step": step})
    for i, path in enumerate(videos):
        key = f"gameplay/step_{step}" if len(videos) == 1 else f"gameplay/step_{step}_ep{i}"
        run.log({key: wandb.Video(str(path), format="mp4")})
    run.finish()
    print(f"Uploaded {len(videos)} clip(s) to W&B project '{project}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env-id", type=str, default=ENV_ID)
    parser.add_argument("--out-dir", type=str, default="videos")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true", help="Upload the clip(s) to W&B")
    parser.add_argument("--wandb-project", type=str, default="atari-rl")
    parser.add_argument(
        "--step", type=int, default=0, help="Training step label for the W&B video key"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip de-flicker; keep the video exactly as the game renders (sprites blink)",
    )
    args = parser.parse_args()
    videos = record(
        args.model,
        args.env_id,
        args.out_dir,
        args.episodes,
        args.seed,
        deflicker_enabled=not args.raw,
    )
    if args.wandb:
        upload_to_wandb(videos, args.wandb_project, args.step)


if __name__ == "__main__":
    main()
