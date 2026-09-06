from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .tabux_controller import TabuXController

MODEL_SPEC = {
    "key": "004_simple_cnn_tabux",
    "display_name": "004 · Simple CNN + TabuX",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "inference_only_tabux_local_basin",
    "description": (
        "Simple CNN 001 with an inference-only TabuX controller. TabuX keeps short-term "
        "behavior coordinates, recent states, and transition tenure so the robot not only "
        "breaks ABAB loops but also avoids immediately re-entering the same 2-4 state basin."
    ),
}


class SimpleCNNTabuX004(nn.Module):
    """Simple CNN baseline plus an inference-only local-basin Tabu Search controller."""

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * height * width, 128),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(128, action_count)
        self.value = nn.Linear(128, 1)
        self.action_count = int(action_count)
        self._tabux = TabuXController(action_count)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.policy(features), self.value(features).squeeze(-1)

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
    return SimpleCNNTabuX004(observation_shape, action_count)
