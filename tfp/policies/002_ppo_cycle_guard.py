from __future__ import annotations

from collections import defaultdict, deque
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
    "key": "002_ppo_cycle_guard",
    "display_name": "002 · PPO Cycle Guard",
    "algorithm": "ppo",
    "description": (
        "PPO categorical policy with the same action-history guard in rollout collection and evaluation. "
        "It prioritizes a legal pickup and suppresses the next action that would continue a persistent opposite-action ABAB oscillation."
    ),
}


class PPOCycleGuard002:
    """Train/eval-consistent anti-oscillation behavior policy.

    Unlike the legacy model-owned Tabu wrappers, this guard changes the categorical
    *mask before sampling*. Therefore the action probability/log-probability stored by
    PPO is the probability of the behavior actually executed, avoiding an off-policy
    mismatch. The rule uses only executed actions and the environment-provided legal
    mask; it never reads hidden map state or oracle distance.
    """

    OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}

    def __init__(self) -> None:
        self._last_mode = "train"
        self._history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=8))

    @staticmethod
    def _dist(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        return Categorical(logits=logits.masked_fill(~mask, -1e9))

    def _guard_mask(self, mask: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor:
        guarded = mask.clone()
        for row, raw_env_id in enumerate(env_ids):
            env_id = int(raw_env_id)
            legal = guarded[row]
            if not bool(legal.any()):
                continue

            # Under valid-mask evaluation, PICKUP becomes legal only on the object
            # cell. Match the task-mask training semantics so an agent cannot orbit
            # the object forever merely because its raw pickup logit is lower.
            if legal.numel() > 4 and bool(legal[4]):
                forced = torch.zeros_like(legal)
                forced[4] = True
                guarded[row] = forced
                continue

            history = list(self._history[env_id])
            if len(history) < 4:
                continue
            a, b, c, d = history[-4:]
            if a == c and b == d and self.OPPOSITE.get(a) == b:
                expected = a  # ABAB would continue with A.
                if 0 <= expected < legal.numel() and bool(legal[expected]):
                    candidate = legal.clone()
                    candidate[expected] = False
                    # Never create an invalid/deadlocked distribution in corridors.
                    if bool(candidate.any()):
                        guarded[row] = candidate
        return guarded

    def sample(self, model: nn.Module, logits: torch.Tensor, mask: torch.Tensor, env_ids: Sequence[int]):
        self._last_mode = "train"
        guarded = self._guard_mask(mask, env_ids)
        dist = self._dist(logits, guarded)
        actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy(), guarded

    def greedy(self, model: nn.Module, logits: torch.Tensor, mask: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor:
        self._last_mode = "inference"
        guarded = self._guard_mask(mask, env_ids)
        proposed = logits.masked_fill(~guarded, -1e9).argmax(dim=1)
        return apply_model_action_rewrite(model, proposed, logits, guarded, env_ids)

    def observe(
        self,
        model: nn.Module,
        env_ids: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        notify_model_action_memory(model, env_ids, actions, rewards, dones)
        if self._last_mode == "inference":
            notify_model_inference_behavior(model, env_ids, actions, rewards, dones)
        for env_id, action, done in zip(env_ids, actions, dones):
            key = int(env_id)
            self._history[key].append(int(action))
            if done:
                self._history.pop(key, None)

    def reset(self, model: nn.Module, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._history.pop(int(env_id), None)
        reset_model_memory(model, env_ids)


def create_policy() -> PPOCycleGuard002:
    return PPOCycleGuard002()
