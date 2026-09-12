from torch import nn
from ._keyed_core import EventMemoryTransformerActorCritic

MODEL_SPEC = {
    "key": "002_event_memory_transformer",
    "display_name": "002 · Event-Memory Transformer",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "compact_event_transformer",
    "description": "Ten-step attention over compact access/progress/hazard state, avoiding expensive visual-frame memory while retaining key, delivery, and risk transitions.",
}

def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return EventMemoryTransformerActorCritic(observation_shape, action_count)
