from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn

from tfp.models._action_memory_core import LearnableActionMemory


def intercept_safety_prior_logits(x: torch.Tensor, action_count: int, view_size: int, route_scale: float = 2.2) -> torch.Tensor:
    """Residual action prior from observable route and hazard telemetry only.

    It steers toward the environment-provided interception/delivery waypoint, chooses
    WAIT when already positioned at a future interception cell, and softly biases away
    from a nearby wolf. PPO learns residual corrections on top of this prior.
    """
    batch = x.shape[0]
    bias = torch.zeros((batch, action_count), dtype=x.dtype, device=x.device)
    dy = x[:, 7].mean(dim=(-2, -1))
    dx = x[:, 8].mean(dim=(-2, -1))
    bias[:, 0] += torch.relu(-dy) * route_scale
    bias[:, 1] += torch.relu(dy) * route_scale
    bias[:, 2] += torch.relu(-dx) * route_scale
    bias[:, 3] += torch.relu(dx) * route_scale

    carrying = x[:, 6].mean(dim=(-2, -1)) > 0.5
    pickup = x[:, 14].mean(dim=(-2, -1)) > 0.5
    delivery = x[:, 15].mean(dim=(-2, -1)) > 0.5
    at_waypoint = (dy.abs() + dx.abs()) < (0.25 / max(1, view_size))
    if action_count > 6:
        should_wait = at_waypoint & (~carrying) & (~pickup) & (~delivery)
        bias[:, 6] += should_wait.to(x.dtype) * 2.6

    # Global low-bandwidth hazard bearing is used only as a near-field safety bias.
    wolf_dy = x[:, 11].mean(dim=(-2, -1))
    wolf_dx = x[:, 12].mean(dim=(-2, -1))
    proximity = x[:, 13].mean(dim=(-2, -1))
    risk = torch.clamp((proximity - 0.10) * 4.0, min=0.0, max=1.0)
    toward_up = (wolf_dy < -1e-5).to(x.dtype)
    toward_down = (wolf_dy > 1e-5).to(x.dtype)
    toward_left = (wolf_dx < -1e-5).to(x.dtype)
    toward_right = (wolf_dx > 1e-5).to(x.dtype)
    bias[:, 0] += risk * (-2.8 * toward_up + 1.0 * toward_down)
    bias[:, 1] += risk * (-2.8 * toward_down + 1.0 * toward_up)
    bias[:, 2] += risk * (-2.8 * toward_left + 1.0 * toward_right)
    bias[:, 3] += risk * (-2.8 * toward_right + 1.0 * toward_left)
    if action_count > 6:
        bias[:, 6] -= 1.2 * risk
    return bias


class DynamicContextEncoder(nn.Module):
    """Separate local spatial evidence from broadcast motion/task context."""

    def __init__(self, observation_shape: tuple[int, int, int], feature_dim: int = 144) -> None:
        super().__init__()
        channels, height, width = observation_shape
        if channels < 18:
            raise ValueError("Moving-cargo models require the 18-channel dynamic observation.")
        self.spatial_channels = 6
        self.context_start = 6
        self.context_dim = channels - self.context_start
        self.spatial = nn.Sequential(
            nn.Conv2d(self.spatial_channels, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
            nn.Flatten(),
            nn.Linear(48 * height * width, 144), nn.LayerNorm(144), nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(self.context_dim, 72), nn.LayerNorm(72), nn.GELU(),
            nn.Linear(72, 72), nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(216, feature_dim), nn.LayerNorm(feature_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        visual = self.spatial(x[:, : self.spatial_channels])
        context = self.context(x[:, self.context_start :].mean(dim=(-2, -1)))
        return self.fusion(torch.cat([visual, context], dim=1))


class InterceptSafetyActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.view_size = int(observation_shape[1])
        self.encoder = DynamicContextEncoder(observation_shape, 144)
        self.policy = nn.Sequential(nn.Linear(144, 112), nn.GELU(), nn.Linear(112, action_count))
        nn.init.normal_(self.policy[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy[-1].bias)
        self.value = nn.Sequential(nn.Linear(144, 112), nn.GELU(), nn.Linear(112, 1))

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        logits = self.policy(z) + intercept_safety_prior_logits(x, self.action_count, self.view_size)
        return logits, self.value(z).squeeze(-1)


class InterceptActionMemoryActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.view_size = int(observation_shape[1])
        self.history_length = 10
        self.pad_token = self.action_count
        self.encoder = DynamicContextEncoder(observation_shape, 144)
        self.action_memory = LearnableActionMemory(action_count, self.history_length, embed_dim=16, memory_dim=48)
        self.fusion = nn.Sequential(
            nn.Linear(144 + 48, 176), nn.LayerNorm(176), nn.GELU(),
            nn.Linear(176, 144), nn.GELU(),
        )
        self.policy = nn.Linear(144, action_count)
        nn.init.normal_(self.policy.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy.bias)
        self.value = nn.Linear(144, 1)
        self._action_history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.history_length))

    def _empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch, self.history_length), self.pad_token, dtype=torch.long, device=device)

    def _history_tensor(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw_id in env_ids:
            items = list(self._action_history[int(raw_id)])[-self.history_length:]
            rows.append([self.pad_token] * (self.history_length - len(items)) + items)
        return torch.tensor(rows, dtype=torch.long, device=device) if rows else self._empty_history(0, device)

    def _heads(self, x: torch.Tensor, history: torch.Tensor):
        z = self.encoder(x)
        mem = self.action_memory(history)
        fused = self.fusion(torch.cat([z, mem], dim=1))
        logits = self.policy(fused) + intercept_safety_prior_logits(x, self.action_count, self.view_size)
        return logits, self.value(fused).squeeze(-1)

    def forward(self, x: torch.Tensor):
        return self._heads(x, self._empty_history(x.shape[0], x.device))

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        del update
        history = self._history_tensor(env_ids, x.device)
        logits, value = self._heads(x, history)
        return logits, value, history.detach()

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        return self._heads(x, context.long())

    def observe_action_memory(self, env_ids, actions, rewards, dones) -> None:
        del rewards
        for env_id, action, done in zip(env_ids, actions, dones):
            key = int(env_id)
            self._action_history[key].append(int(action))
            if done:
                self._action_history.pop(key, None)

    def reset_action_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._action_history.pop(int(env_id), None)


class InterceptGRUActionMemoryActorCritic(nn.Module):
    """Motion-context encoder + explicit action history + temporal dynamics memory."""

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.view_size = int(observation_shape[1])
        self.history_length = 8
        self.pad_token = self.action_count
        self.hidden_dim = 112
        self.encoder = DynamicContextEncoder(observation_shape, 144)
        self.action_memory = LearnableActionMemory(action_count, self.history_length, embed_dim=12, memory_dim=36)
        self.recurrent = nn.GRUCell(144 + 36, self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(144 + 112 + 36, 224), nn.LayerNorm(224), nn.GELU(),
            nn.Linear(224, 144), nn.GELU(),
        )
        self.policy = nn.Linear(144, action_count)
        nn.init.normal_(self.policy.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy.bias)
        self.value = nn.Linear(144, 1)
        self._hidden: dict[int, torch.Tensor] = {}
        self._action_history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.history_length))

    def _empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch, self.history_length), self.pad_token, dtype=torch.long, device=device)

    def _history_for_envs(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw_id in env_ids:
            items = list(self._action_history[int(raw_id)])[-self.history_length:]
            rows.append([self.pad_token] * (self.history_length - len(items)) + items)
        return torch.tensor(rows, dtype=torch.long, device=device) if rows else self._empty_history(0, device)

    def _hidden_for_envs(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = [self._hidden.get(int(raw_id), torch.zeros(self.hidden_dim, device=device)).to(device) for raw_id in env_ids]
        return torch.stack(rows, dim=0) if rows else torch.zeros((0, self.hidden_dim), device=device)

    def _pack(self, hidden: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden, history.float()], dim=1).detach()

    def _unpack(self, context: torch.Tensor):
        hidden = context[:, :self.hidden_dim].float()
        history = context[:, self.hidden_dim:self.hidden_dim+self.history_length].round().long()
        return hidden, history

    def _heads(self, x: torch.Tensor, hidden: torch.Tensor, history: torch.Tensor):
        z = self.encoder(x)
        mem = self.action_memory(history)
        hidden_new = self.recurrent(torch.cat([z, mem], dim=1), hidden)
        fused = self.fusion(torch.cat([z, hidden_new, mem], dim=1))
        logits = self.policy(fused) + intercept_safety_prior_logits(x, self.action_count, self.view_size)
        return logits, self.value(fused).squeeze(-1), hidden_new

    def forward(self, x: torch.Tensor):
        hidden = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)
        history = self._empty_history(x.shape[0], x.device)
        logits, value, _ = self._heads(x, hidden, history)
        return logits, value

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        hidden = self._hidden_for_envs(env_ids, x.device)
        history = self._history_for_envs(env_ids, x.device)
        context = self._pack(hidden, history)
        logits, value, hidden_new = self._heads(x, hidden, history)
        if update:
            for i, raw_id in enumerate(env_ids):
                self._hidden[int(raw_id)] = hidden_new[i].detach()
        return logits, value, context

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        hidden, history = self._unpack(context)
        logits, value, _ = self._heads(x, hidden, history)
        return logits, value

    def training_sequence_forward(self, x_seq, initial_context, actions_seq, dones_seq):
        hidden, history = self._unpack(initial_context)
        logits_out, values_out = [], []
        for t in range(x_seq.shape[0]):
            logits, value, hidden_new = self._heads(x_seq[t], hidden, history)
            logits_out.append(logits); values_out.append(value)
            history = torch.cat([history[:, 1:], actions_seq[t].long().unsqueeze(1)], dim=1)
            done = dones_seq[t].bool().unsqueeze(1)
            hidden = torch.where(done, torch.zeros_like(hidden_new), hidden_new)
            history = torch.where(done.expand_as(history), torch.full_like(history, self.pad_token), history)
        return torch.stack(logits_out, dim=0), torch.stack(values_out, dim=0)

    def observe_action_memory(self, env_ids, actions, rewards, dones) -> None:
        del rewards
        for env_id, action, done in zip(env_ids, actions, dones):
            key = int(env_id)
            self._action_history[key].append(int(action))
            if done:
                self._action_history.pop(key, None)
                self._hidden.pop(key, None)

    def reset_action_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._action_history.pop(int(env_id), None)
            self._hidden.pop(int(env_id), None)

    def reset_sequence_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._hidden.pop(int(env_id), None)
