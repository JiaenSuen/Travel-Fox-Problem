from torch import nn
from ._keyed_core import DependencyFiLMShieldActorCritic

MODEL_SPEC = {
    "key": "001_dependency_film_shield",
    "display_name": "001 · Dependency-FiLM Shield",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "symbolic_context",
    "description": "FiLM-fuses local geometry with key/cargo/access state and learns a bounded PPO residual over a dependency-route and soft hazard-safety prior.",
}

def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return DependencyFiLMShieldActorCritic(observation_shape, action_count)
