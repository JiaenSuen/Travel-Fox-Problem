from __future__ import annotations

import torch
from torch import nn

MODEL_SPEC = {
    "key": "002_goal_conditioned_cnn",
    "display_name": "002 · Goal-Conditioned CNN",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "goal_conditioning",
    "description": "Dual-path CNN that fuses local spatial features with pooled carried/target-color context before actor/value heads.",
}


class GoalConditionedCNN(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(48 * height * width, 128), nn.ReLU(inplace=True),
        )
        self.context = nn.Sequential(
            nn.Linear(channels, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 64), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(nn.Linear(192, 160), nn.ReLU(inplace=True))
        self.policy = nn.Linear(160, action_count)
        self.value = nn.Linear(160, 1)

    def forward(self, x: torch.Tensor):
        spatial = self.spatial(x)
        context = self.context(x.mean(dim=(-2, -1)))
        z = self.fusion(torch.cat((spatial, context), dim=-1))
        return self.policy(z), self.value(z).squeeze(-1)


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return GoalConditionedCNN(observation_shape, action_count)
