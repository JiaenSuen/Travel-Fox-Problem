from __future__ import annotations

from torch import nn
from ._recurrent_core import CompactRecurrentActorCritic

MODEL_SPEC = {
    "key": "007_ppo_gru_action_memory",
    "display_name": "007 · PPO-GRU + Action Memory",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru_plus_action_history",
    "intrinsic_module": "none",
    "description": "Compact PPO-GRU conditioned on explicit recent executed actions in addition to recurrent visual state.",
}


def create_model(observation_shape, action_count) -> nn.Module:
    return CompactRecurrentActorCritic(observation_shape, action_count, cell_type="gru", action_memory=True)
