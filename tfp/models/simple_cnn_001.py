from __future__ import annotations

import torch
from torch import nn

MODEL_SPEC = {
    "key": "simple_cnn_001",
    "display_name": "Simple CNN 001",
    "algorithm": "ppo",
    "recurrent": False,
    "description": "Small feed-forward CNN baseline for TFP-FoxTransport-Local.",
}


class SimpleCNN001(nn.Module):
    """Compact shared visual encoder with policy and value heads."""

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.policy(features), self.value(features).squeeze(-1)


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return SimpleCNN001(observation_shape, action_count)
