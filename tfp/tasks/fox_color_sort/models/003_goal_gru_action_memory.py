from __future__ import annotations

from torch import nn

from tfp.models._recurrent_core import CompactRecurrentActorCritic

MODEL_SPEC = {
    "key": "003_goal_gru_action_memory",
    "display_name": "003 · Goal GRU + Action Memory",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru_plus_action_history",
    "description": "Compact recurrent PPO model for local-view aliasing, augmented with explicit recent-action memory.",
}


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return CompactRecurrentActorCritic(observation_shape, action_count, cell_type="gru", action_memory=True)
