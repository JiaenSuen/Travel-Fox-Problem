from __future__ import annotations

from torch import nn

from ._route_core import RouteFusionActorCritic

MODEL_SPEC = {
    "key": "001_route_prior_cnn",
    "display_name": "001 · Route-Prior CNN",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "structured_context",
    "description": (
        "Separates local geometry from task context and learns PPO residual logits on top of a high-level doorway-route prior."
    ),
}


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return RouteFusionActorCritic(observation_shape, action_count)
