from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ._common import RunningScale, flat_mlp

INTRINSIC_SPEC = {
    "key": "001_gobi_compact",
    "display_name": "001 · GoBI Compact",
    "default_coef": 0.018,
    "anneal_fraction": 0.80,
    "trainable": True,
    "description": "Lightweight GoBI adaptation combining lifelong count novelty with imagined episodic reachability expansion from a tiny latent world model.",
}


class GoBICompact001(nn.Module):
    """Tiny latent-world-model GoBI adaptation for 5x5 local TFP observations."""

    def __init__(self, observation_shape, action_count, device) -> None:
        super().__init__()
        self.device = device
        self.action_count = int(action_count)
        input_dim = int(np.prod(observation_shape))
        self.encoder = flat_mlp(input_dim, 24, hidden=48).to(device)
        # Stable novelty signatures: encoder remains fixed; only dynamics learns.
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.dynamics = nn.Sequential(nn.Linear(24 + action_count, 64), nn.Tanh(), nn.Linear(64, 24), nn.Tanh()).to(device)
        self.optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=7e-4)
        self.global_counts: dict[bytes, int] = defaultdict(int)
        self.episode_signatures: dict[int, set[bytes]] = defaultdict(set)
        self.scale = RunningScale()
        self.rng = np.random.default_rng(2023)

    def reset(self, env_ids) -> None:
        for env_id in env_ids:
            self.episode_signatures.pop(int(env_id), None)

    def _one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_count).float()

    @staticmethod
    def _sig(z: np.ndarray) -> bytes:
        q = np.clip(np.rint(z * 5.0), -8, 8).astype(np.int8)
        return q.tobytes()

    @torch.no_grad()
    def compute(self, obs, next_obs, actions, env_ids, dones) -> np.ndarray:
        del obs, actions
        nx = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        nz = self.encoder(nx)
        raw = []
        for i, (env_id, done) in enumerate(zip(env_ids, dones)):
            key = int(env_id)
            z0 = nz[i : i + 1]
            observed_sig = self._sig(z0[0].cpu().numpy())
            lifetime = 1.0 / np.sqrt(float(self.global_counts[observed_sig]) + 1.0)
            self.global_counts[observed_sig] += 1
            before = len(self.episode_signatures[key])
            self.episode_signatures[key].add(observed_sig)

            frontier = z0
            # Very small imagination budget: 3 random branches x 2 steps.
            for _ in range(2):
                branches = []
                for _branch in range(3):
                    a_np = self.rng.integers(0, self.action_count, size=frontier.shape[0])
                    a = torch.as_tensor(a_np, dtype=torch.long, device=self.device)
                    pred = self.dynamics(torch.cat([frontier, self._one_hot(a)], dim=1))
                    branches.append(pred)
                    for row in pred.cpu().numpy():
                        self.episode_signatures[key].add(self._sig(row))
                frontier = torch.cat(branches, dim=0)[:6]
            expansion = max(0, len(self.episode_signatures[key]) - before)
            raw.append(lifetime * float(min(expansion, 7)))
            if done:
                self.episode_signatures.pop(key, None)
        return self.scale.update_and_scale(np.asarray(raw, dtype=np.float32), clip=4.0)

    def update(self, obs, next_obs, actions) -> float:
        x = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        nx = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        with torch.no_grad():
            z = self.encoder(x)
            nz = self.encoder(nx)
        pred = self.dynamics(torch.cat([z, self._one_hot(a)], dim=1))
        loss = F.mse_loss(pred, nz)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.dynamics.parameters(), 1.0)
        self.optimizer.step()
        return float(loss.detach().cpu())


def create_intrinsic(observation_shape, action_count, device):
    return GoBICompact001(observation_shape, action_count, device)
