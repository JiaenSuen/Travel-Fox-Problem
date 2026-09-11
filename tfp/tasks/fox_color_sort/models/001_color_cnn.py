from __future__ import annotations

from torch import nn

from tfp.models._simple_cnn_core import SimpleCNNActorCritic

MODEL_SPEC = {
    "key": "001_color_cnn",
    "display_name": "001 · Color CNN",
    "algorithm": "ppo",
    "recurrent": False,
    "description": "Feed-forward PPO baseline over color-separated local object/goal channels and target-direction cues.",
}


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return SimpleCNNActorCritic(observation_shape, action_count)
