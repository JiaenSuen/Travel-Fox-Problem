from __future__ import annotations
from ._dynamic_core import InterceptGRUActionMemoryActorCritic

MODEL_SPEC = {
    "key": "003_intercept_gru_memory",
    "display_name": "003 · Intercept GRU + Action Memory",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru+action_history",
    "description": "Recurrent dynamic-scene model that tracks carrier phase and hazard motion while retaining explicit executed-action memory.",
}

def create_model(observation_shape, action_count):
    return InterceptGRUActionMemoryActorCritic(observation_shape, action_count)
