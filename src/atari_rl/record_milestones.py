"""Record gameplay videos from PPO checkpoints at fixed step intervals and log them
all to a single W&B run — the progression montage (see CLAUDE.md §7).

Reuses ``record()`` and the ``wandb.Video`` logging pattern from ``record.py`` rather
than reimplementing either. Each milestone is written to its own folder so nothing
overwrites, and every clip lands in one W&B run instead of ten scattered ones.

Usage (run this in the RunPod terminal):
    python -m atari_rl.record_milestones                              # auto-detect run, 1M interval, upload
    python -m atari_rl.record_milestones --interval 1000000 --no-wandb  # local only
    python -m atari_rl.record_milestones --run-dir models/ppo_1782338992
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from atari_rl.env import ENV_ID
from atari_rl.record import record

_CKPT_RE = re.compile(r"_(\d+)_steps\.zip$")


def find_run_dir(models_root: Path) -> Path:
    """Newest ``models/ppo_*`` whose ``checkpoints/`` actually holds checkpoint zips.

    Run-dir timestamps differ between machines (local vs RunPod), so the run is
    discovered rather than hardcoded.
    """
    candidates = sorted(models_root.glob("ppo_*/checkpoints"), key=lambda p: p.stat().st_mtime)
    for ckpt_dir in reversed(candidates):
        if any(ckpt_dir.glob("ppo_*_steps.zip")):
            return ckpt_dir.parent
    raise SystemExit(f"No PPO checkpoints found under {models_root.resolve()}")


def select_checkpoints(ckpt_dir: Path, interval: int) -> list[tuple[int, Path]]:
    """Checkpoints whose step count is a multiple of ``interval`` (e.g. 1M, 2M, …),
    sorted ascending. Robust to the underlying 200k save cadence."""
    selected: list[tuple[int, Path]] = []
    for zip_path in ckpt_dir.glob("ppo_*_steps.zip"):
        match = _CKPT_RE.search(zip_path.name)
        if match and int(match.group(1)) % interval == 0:
            selected.append((int(match.group(1)), zip_path))
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="PPO run dir (e.g. models/ppo_1782338992). Default: auto-detect newest.",
    )
    parser.add_argument("--models-root", type=str, default="models")
    parser.add_argument("--interval", type=int, default=1_000_000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-project", type=str, default="atari-rl")
    parser.add_argument("--no-wandb", action="store_true", help="Skip the W&B upload")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Skip de-flicker; keep the video exactly as the game renders (sprites blink)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else find_run_dir(Path(args.models_root))
    ckpt_dir = run_dir / "checkpoints"
    milestones = select_checkpoints(ckpt_dir, args.interval)
    if not milestones:
        raise SystemExit(
            f"No checkpoints divisible by {args.interval:,} found in {ckpt_dir.resolve()}"
        )

    print(f"Run: {run_dir.name}  |  {len(milestones)} milestone(s) at {args.interval:,}-step intervals")

    recorded: list[tuple[int, list[Path]]] = []
    for step, ckpt_path in milestones:
        out_dir = Path("videos") / run_dir.name / f"step_{step}"
        print(f"\n=== Recording step {step:,} -> {out_dir} ===")
        videos = record(
            str(ckpt_path),
            ENV_ID,
            str(out_dir),
            args.episodes,
            args.seed,
            deflicker_enabled=not args.raw,
        )
        recorded.append((step, videos))

    if args.no_wandb:
        print(f"\nDone. {len(recorded)} milestone clip(s) saved locally under videos/{run_dir.name}/.")
        return

    import wandb

    wandb_run = wandb.init(
        project=args.wandb_project,
        job_type="record",
        name=f"{run_dir.name}_milestones",
        config={"interval": args.interval, "run": run_dir.name},
    )
    for step, videos in recorded:
        for i, path in enumerate(videos):
            key = f"gameplay/step_{step}" if len(videos) == 1 else f"gameplay/step_{step}_ep{i}"
            wandb_run.log({key: wandb.Video(str(path), format="mp4")})
    wandb_run.finish()
    print(
        f"\nDone. Uploaded {sum(len(v) for _, v in recorded)} clip(s) to "
        f"W&B project '{args.wandb_project}' as run '{run_dir.name}_milestones'."
    )


if __name__ == "__main__":
    main()
