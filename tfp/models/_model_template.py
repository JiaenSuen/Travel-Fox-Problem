"""Copy this file to a new name such as ``my_cnn_002.py`` and remove the leading underscore."""
from __future__ import annotations

import torch
from torch import nn

MODEL_SPEC = {
    "key": "my_cnn_002",
    "display_name": "My CNN 002",
    "algorithm": "ppo",
    "recurrent": False,
    "description": "Describe the research idea here.",
}


class MyModel(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * height * width, 64),
            nn.ReLU(inplace=True),
        )
        self.policy = nn.Linear(64, action_count)
        self.value = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.policy(features), self.value(features).squeeze(-1)


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return MyModel(observation_shape, action_count)
