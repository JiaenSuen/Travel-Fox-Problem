from torch import nn
from ._keyed_core import DualTimescaleGRUActorCritic

MODEL_SPEC = {
    "key": "003_dual_timescale_gru_shield",
    "display_name": "003 · Dual-Timescale GRU Shield",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "task_gru+hazard_gru+action_memory",
    "description": "Separate long-horizon task-progress and short-horizon hazard memories with explicit action history and a bounded safety-aware residual policy.",
}

def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return DualTimescaleGRUActorCritic(observation_shape, action_count)
