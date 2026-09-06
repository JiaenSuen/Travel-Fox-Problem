from dataclasses import FrozenInstanceError
import pytest

from tfp.training.ppo import PPOConfig


def test_ppo_config_is_immutable_after_run_snapshot():
    cfg = PPOConfig(action_mask_mode='task')
    with pytest.raises(FrozenInstanceError):
        cfg.action_mask_mode = 'valid'  # type: ignore[misc]
