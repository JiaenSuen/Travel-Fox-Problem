from __future__ import annotations
from ._dynamic_core import InterceptActionMemoryActorCritic

MODEL_SPEC = {
    "key": "002_intercept_action_memory",
    "display_name": "002 · Intercept + Action Memory",
    "algorithm": "ppo",
    "memory_type": "action_history",
    "description": "Intercept-safety policy with compact executed-action history for timing, pursuit oscillation, and evasive maneuver context.",
}

def create_model(observation_shape, action_count):
    return InterceptActionMemoryActorCritic(observation_shape, action_count)
