from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from torch import nn


class RunningScale:
    """Small online RMS normalizer used only to stabilize intrinsic reward magnitudes."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.count = eps
        self.mean = 0.0
        self.m2 = 1.0

    def update_and_scale(self, values: np.ndarray, clip: float = 5.0) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        for x in values.reshape(-1):
            self.count += 1.0
            delta = float(x) - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (float(x) - self.mean)
        std = math.sqrt(max(self.m2 / max(1.0, self.count - 1.0), 1e-8))
        scaled = values / max(std, 1e-3)
        return np.clip(scaled, 0.0, clip).astype(np.float32)


def flat_mlp(input_dim: int, output_dim: int, hidden: int = 64) -> nn.Sequential:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, output_dim),
        nn.Tanh(),
    )


def observation_signature(obs: np.ndarray) -> bytes:
    # TFP observations are mostly binary plus low-bandwidth direction cues. Quantizing
    # to 1/8 increments preserves meaningful local distinctions while staying robust
    # to tiny floating-point differences.
    q = np.clip(np.rint(np.asarray(obs, dtype=np.float32) * 8.0), -16, 16).astype(np.int8)
    return q.tobytes()


class EpisodicMemoryMixin:
    def __init__(self) -> None:
        self._episode_memory: dict[int, list[np.ndarray]] = defaultdict(list)

    def reset(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._episode_memory.pop(int(env_id), None)
