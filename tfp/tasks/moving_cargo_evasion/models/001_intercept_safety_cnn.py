from __future__ import annotations
from ._dynamic_core import InterceptSafetyActorCritic

MODEL_SPEC = {
    "key": "001_intercept_safety_cnn",
    "display_name": "001 · Intercept-Safety CNN",
    "algorithm": "ppo",
    "memory_type": "none",
    "description": "Local visual/context encoder with residual interception and near-field predator-avoidance action prior.",
}

def create_model(observation_shape, action_count):
    return InterceptSafetyActorCritic(observation_shape, action_count)
