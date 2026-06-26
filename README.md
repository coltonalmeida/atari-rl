# space-invaders-rl

End-to-end deep reinforcement learning pipeline that trains an agent to play
**Atari Space Invaders** (`ALE/SpaceInvaders-v5`) and beats a random-play baseline
by a wide margin. A **PPO** agent is trained, evaluated, and compared against a
random-play baseline with tracked experiments, saved checkpoints, and recorded gameplay.

<video src="https://github.com/user-attachments/assets/b0d9951d-6dbe-4e43-b6e3-461de39d4caa" controls width="600">
  <!-- Fallback for viewers that don't render the video tag (npm, some IDEs) -->
  <a href="https://www.youtube.com/watch?v=yJduFZuogFA">Watch the gameplay demo on YouTube</a>
</video>

## Results

Mean episode reward over 30 evaluation episodes:

| Agent  | Mean episode reward | Std | Improvement vs random |
|--------|---------------------|-----|-----------------------|
| Random | 465                 | 278 | 1.0×                  |
| PPO    | 1,820               | 610 | 3.9×                  |

The PPO agent was trained for 10M steps on a cloud GPU and significantly
outperforms random play.

## What this project demonstrates

- **Deep RL**: PPO trained from raw pixels with `CnnPolicy`
- **Modern RL stack**: Gymnasium + Arcade Learning Environment, Stable-Baselines3 (PyTorch)
- **Reproducible experiments**: every run driven by a Hydra config file, never hardcoded params
- **Experiment tracking**: live reward curves and checkpoint videos logged to Weights & Biases
- **Evaluation & reporting**: scripted N-episode evaluation with mean/std and improvement factor
- **Engineering practices**: pytest, ruff/black, and a working Dockerfile

## Setup

```bash
uv venv
uv sync
```

ROMs are installed automatically via AutoROM (`accept-rom-license` extra).

## Usage

```bash
# Random-agent baseline (sanity check + baseline reward)
python -m atari_rl.evaluate --random --episodes 30 --out reports/random_baseline.json

# Train PPO (Hydra config; override any value on the CLI)
python -m atari_rl.train --config-name=ppo

# Evaluate a trained model (prints improvement factor vs the random baseline)
python -m atari_rl.evaluate --model models/<run>/best/best_model.zip --episodes 30

# Record gameplay video (add --wandb to upload the clip to Weights & Biases)
python -m atari_rl.record --model models/<run>/best/best_model.zip --step 10000000 --wandb
```

## Development

```bash
uv run pytest
uv run ruff check src tests
uv run black --check src tests
```

## Stack

Python 3.10+ · Gymnasium + ALE-py · Stable-Baselines3 (PyTorch) · Weights & Biases ·
Hydra configs · pytest · ruff/black · Docker
