from __future__ import annotations

from torch import nn
from ._action_memory_core import ActionMemoryCNN

MODEL_SPEC = {
    "key": "003_action_memory_cnn",
    "display_name": "003 · Action-Memory CNN",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "learnable_action_history",
    "description": (
        "Compact CNN with a learnable short action-memory layer. The environment still emits one "
        "5x5 observation; the model internally embeds recent executed actions and gates the visual feature."
    ),
}

ActionMemoryCNN003 = ActionMemoryCNN


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return ActionMemoryCNN(observation_shape, action_count)
