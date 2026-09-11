from __future__ import annotations

from torch import nn

from ._route_core import RouteGRUActionMemoryActorCritic

MODEL_SPEC = {
    "key": "003_route_prior_gru_memory",
    "display_name": "003 · Route-Prior GRU + Action Memory",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "structured_context_plus_gru_plus_action_history",
    "description": (
        "Doorway-route residual PPO combining explicit action history with GRU room/door memory for long-horizon aliasing."
    ),
}


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return RouteGRUActionMemoryActorCritic(observation_shape, action_count)
