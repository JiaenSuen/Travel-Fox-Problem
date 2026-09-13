from dataclasses import FrozenInstanceError
import pytest

from tfp.training.ppo import PPOConfig, TrainingControl, _protocol_fingerprint


def test_ppo_config_is_immutable_after_run_snapshot():
    cfg = PPOConfig(action_mask_mode='task')
    with pytest.raises(FrozenInstanceError):
        cfg.action_mask_mode = 'valid'  # type: ignore[misc]


def test_training_control_changes_execution_state_not_protocol():
    cfg = PPOConfig(action_mask_mode="task", lr=2.5e-4, seed=77)
    fingerprint = _protocol_fingerprint(cfg)
    control = TrainingControl()
    control.pause()
    assert control.paused and not control.stop_requested
    control.resume()
    assert not control.paused
    control.request_stop()
    assert control.stop_requested and not control.paused
    assert cfg.action_mask_mode == "task"
    assert cfg.lr == 2.5e-4 and cfg.seed == 77
    assert _protocol_fingerprint(cfg) == fingerprint
