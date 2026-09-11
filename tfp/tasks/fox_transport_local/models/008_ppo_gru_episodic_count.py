from __future__ import annotations

from torch import nn
from tfp.models._recurrent_core import CompactRecurrentActorCritic

MODEL_SPEC = {
    "key": "008_ppo_gru_episodic_count",
    "display_name": "008 · PPO-GRU + Episodic Count",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru",
    "intrinsic_module": "001_episodic_count",
    "description": "Compact PPO-GRU paired with per-episode observation-count novelty for broader local coverage.",
}


def create_model(observation_shape, action_count) -> nn.Module:
    return CompactRecurrentActorCritic(observation_shape, action_count, cell_type="gru")
