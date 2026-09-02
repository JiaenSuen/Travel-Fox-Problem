from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.distributions import Categorical

from .policy_api import (
    apply_model_action_rewrite,
    notify_model_action_memory,
    notify_model_inference_behavior,
    reset_model_memory,
)

POLICY_SPEC = {
    "key": "ppo_categorical_001",
    "display_name": "PPO Categorical 001",
    "algorithm": "ppo",
    "description": (
        "Categorical sampling during PPO training and masked greedy actions during evaluation. "
        "Model-owned learnable action memory is updated in both phases, while optional action "
        "rewrites such as Tabu are applied only at inference time."
    ),
}


class PPOCategoricalPolicy001:
    """Default PPO action policy with a clean train/inference separation.

    ``sample`` is the behavior policy used to collect PPO rollouts. It never applies
    non-differentiable Tabu action rewrites. ``greedy`` is the evaluation/deployment
    path and may invoke a model-owned inference controller. Learnable action-memory
    hooks still receive executed actions in both modes because they are part of the
    neural context rather than a post-hoc controller.
    """

    def __init__(self) -> None:
        self._last_mode = "train"

    @staticmethod
    def _dist(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        return Categorical(logits=logits.masked_fill(~mask, -1e9))

    def sample(self, model: nn.Module, logits: torch.Tensor, mask: torch.Tensor, env_ids: Sequence[int]):
        self._last_mode = "train"
        dist = self._dist(logits, mask)
        actions = dist.sample()
        # Important: no inference-only action rewrite during PPO rollout collection.
        logp = dist.log_prob(actions)
        return actions, logp, dist.entropy()

    def greedy(self, model: nn.Module, logits: torch.Tensor, mask: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor:
        self._last_mode = "inference"
        masked = logits.masked_fill(~mask, -1e9)
        proposed = masked.argmax(dim=1)
        return apply_model_action_rewrite(model, proposed, logits, mask, env_ids)

    def observe(
        self,
        model: nn.Module,
        env_ids: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        # Learnable action memory is neural context and therefore updates in training
        # and inference. The Tabu/behavior controller is inference-only in TFP.
        notify_model_action_memory(model, env_ids, actions, rewards, dones)
        if self._last_mode == "inference":
            notify_model_inference_behavior(model, env_ids, actions, rewards, dones)

    def reset(self, model: nn.Module, env_ids: Sequence[int]) -> None:
        reset_model_memory(model, env_ids)


def create_policy() -> PPOCategoricalPolicy001:
    return PPOCategoricalPolicy001()
