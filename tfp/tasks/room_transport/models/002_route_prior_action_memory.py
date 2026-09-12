from __future__ import annotations

from torch import nn

from ._route_core import RouteActionMemoryActorCritic

MODEL_SPEC = {
    "key": "002_route_prior_action_memory",
    "display_name": "002 · Route-Prior + Action Memory",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "structured_context_plus_action_history",
    "description": (
        "Doorway-route residual PPO with a learnable 10-step executed-action history for retries, backtracking, and short cycles."
    ),
}


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return RouteActionMemoryActorCritic(observation_shape, action_count)
