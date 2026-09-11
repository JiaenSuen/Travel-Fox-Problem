from __future__ import annotations

from torch import nn
from tfp.models._recurrent_core import CompactRecurrentActorCritic

MODEL_SPEC = {
    "key": "006_ppo_gru_bootstrap",
    "display_name": "006 · PPO-GRU Bootstrap",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru_bootstrap_from_cnn",
    "intrinsic_module": "none",
    "bootstrap_from": "001_simple_cnn",
    "description": (
        "GRU policy initialized from the mature 001 Simple CNN visual encoder and actor/value heads. "
        "The recurrent residual starts near zero and is introduced gradually instead of switching a random recurrent branch on mid-run."
    ),
}


def create_model(observation_shape, action_count) -> nn.Module:
    return CompactRecurrentActorCritic(
        observation_shape,
        action_count,
        cell_type="gru",
        staged=False,
        action_memory=False,
        bootstrap_mode=True,
    )
