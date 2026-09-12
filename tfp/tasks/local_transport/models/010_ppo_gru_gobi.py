from __future__ import annotations

from torch import nn
from tfp.models._recurrent_core import CompactRecurrentActorCritic

MODEL_SPEC = {
    "key": "010_ppo_gru_gobi",
    "display_name": "010 · PPO-GRU + GoBI",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru",
    "intrinsic_module": "001_gobi_compact",
    "description": "Compact PPO-GRU paired with a tiny latent-world-model GoBI-style reachability-expansion bonus.",
}


def create_model(observation_shape, action_count) -> nn.Module:
    return CompactRecurrentActorCritic(observation_shape, action_count, cell_type="gru")
