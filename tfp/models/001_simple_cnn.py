from __future__ import annotations

from torch import nn
from ._simple_cnn_core import SimpleCNNActorCritic

MODEL_SPEC = {
    "key": "001_simple_cnn",
    "display_name": "001 · Simple CNN",
    "algorithm": "ppo",
    "recurrent": False,
    "description": "Small feed-forward CNN baseline for TFP-FoxTransport-Local.",
}

SimpleCNN001 = SimpleCNNActorCritic


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return SimpleCNNActorCritic(observation_shape, action_count)
