from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ._common import observation_signature

INTRINSIC_SPEC = {
    "key": "001_episodic_count",
    "display_name": "001 · Episodic Count",
    "default_coef": 0.04,
    "anneal_fraction": 0.75,
    "trainable": False,
    "description": "Observation-count novelty bonus 1/sqrt(N_episode(o)+1) with per-environment episode reset.",
}


class EpisodicCount001:
    def __init__(self, observation_shape, action_count, device) -> None:
        del observation_shape, action_count, device
        self._counts: dict[int, dict[bytes, int]] = defaultdict(dict)

    def reset(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._counts.pop(int(env_id), None)

    def compute(self, obs, next_obs, actions, env_ids, dones) -> np.ndarray:
        del obs, actions
        bonuses = []
        for nxt, env_id, done in zip(next_obs, env_ids, dones):
            key = int(env_id)
            sig = observation_signature(nxt)
            count = self._counts[key].get(sig, 0)
            bonuses.append(1.0 / np.sqrt(float(count) + 1.0))
            self._counts[key][sig] = count + 1
            if done:
                self._counts.pop(key, None)
        return np.asarray(bonuses, dtype=np.float32)

    def update(self, obs, next_obs, actions) -> float:
        del obs, next_obs, actions
        return 0.0

    def state_dict(self):
        return {}


def create_intrinsic(observation_shape, action_count, device):
    return EpisodicCount001(observation_shape, action_count, device)
