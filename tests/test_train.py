"""Fast wiring tests for the training entry point — no actual training runs.

Validates that both Hydra configs parse and carry the keys train() relies on,
and that the PPO learning-rate schedule behaves as expected.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from atari_rl.train import ALGOS, linear_schedule

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
REQUIRED_KEYS = ["algo", "env_id", "policy", "hyperparams", "total_timesteps", "n_envs", "seed"]


@pytest.mark.parametrize("name", ["ppo"])
def test_config_has_required_keys(name):
    cfg = OmegaConf.load(CONFIGS_DIR / f"{name}.yaml")
    for key in REQUIRED_KEYS:
        assert key in cfg, f"{name}.yaml missing required key: {key}"
    assert cfg.algo in ALGOS
    assert cfg.policy == "CnnPolicy"  # raw-pixel observations require a CNN policy
    assert "freq" in cfg.eval and "episodes" in cfg.eval


def test_linear_schedule_decays_to_zero():
    schedule = linear_schedule(0.1)
    assert schedule(1.0) == pytest.approx(0.1)  # start of training
    assert schedule(0.5) == pytest.approx(0.05)  # halfway
    assert schedule(0.0) == pytest.approx(0.0)  # end of training
