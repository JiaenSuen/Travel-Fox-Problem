from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ._action_memory_core import ActionMemoryCNN
from .tabux_controller import TabuXController

MODEL_SPEC = {
    "key": "005_action_memory_tabux_cnn",
    "display_name": "005 · Action-Memory CNN + TabuX",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "learnable_action_history_plus_inference_tabux",
    "description": (
        "Action-Memory CNN 003 with the same learnable eight-action neural context during PPO "
        "training plus an independent inference-only TabuX local-basin controller at deployment."
    ),
}


class ActionMemoryTabuXCNN005(ActionMemoryCNN):
    """Learnable action-memory policy augmented by inference-only TabuX.

    Neural training is exactly the Action-Memory CNN path: recent executed action IDs
    are embedded and fused into visual features. TabuX is separate and
    non-differentiable; PPO never uses its rewritten actions. During greedy deployment,
    TabuX adds short-term local-basin memory to suppress repeated state/action loops and
    immediate re-entry after an escape move.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__(observation_shape, action_count)
        self._tabux = TabuXController(action_count)

    def reset_behavior_memory(self, env_ids: Sequence[int]) -> None:
        self._tabux.reset(env_ids)

    def observe_behavior(
        self,
        env_ids: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        self._tabux.observe(env_ids, actions, rewards, dones)

    def rewrite_actions(
        self,
        proposed: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
        env_ids: Sequence[int],
    ) -> torch.Tensor:
        return self._tabux.rewrite_actions(proposed, logits, mask, env_ids)

    def cycle_event_count(self, env_id: int) -> int:
        return self._tabux.cycle_event_count(env_id)


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return ActionMemoryTabuXCNN005(observation_shape, action_count)
