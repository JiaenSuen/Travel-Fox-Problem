from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from ._recurrent_core import CompactRecurrentActorCritic
from .tabux_controller import TabuXController

MODEL_SPEC = {
    "key": "009_ppo_gru_action_memory_tabux",
    "display_name": "009 · PPO-GRU + Action Memory + TabuX",
    "algorithm": "ppo",
    "recurrent": True,
    "memory_type": "gru_plus_action_history_plus_inference_tabux",
    "intrinsic_module": "none",
    "description": (
        "The 007 recurrent action-memory policy with an independent inference-only TabuX "
        "local-basin escape controller for deployment/evaluation ablations."
    ),
}


class PPOGRUActionMemoryTabuX009(CompactRecurrentActorCritic):
    """GRU + explicit action history with inference-only TabuX escape control.

    Neural training is intentionally the same as 007: PPO sees the GRU and explicit
    six-action context, while TabuX is inactive during rollout collection. During
    greedy evaluation, TabuX tracks short-term behavior coordinates/transitions and
    may rewrite a cycle-continuing action. Keeping the controller outside PPO makes
    007 vs 009 an interpretable deployment-controller ablation rather than a hidden
    change to the learned recurrent policy.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__(
            observation_shape,
            action_count,
            cell_type="gru",
            action_memory=True,
        )
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
    return PPOGRUActionMemoryTabuX009(observation_shape, action_count)
