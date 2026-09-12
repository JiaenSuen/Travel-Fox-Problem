from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn

from tfp.models._action_memory_core import LearnableActionMemory


def route_prior_logits(x: torch.Tensor, action_count: int, view_size: int, scale: float = 2.0) -> torch.Tensor:
    """Fixed residual navigation prior derived only from observable route channels.

    Channels 7/8 contain the normalized offset to the next geodesic turning waypoint.
    The prior favors the corresponding cardinal move. When that waypoint is one cell
    away and a closed door is adjacent, it favors TOGGLE_DOOR instead. PPO learns a
    residual on top of this prior and can override it where experience warrants.
    """
    batch = x.shape[0]
    bias = torch.zeros((batch, action_count), dtype=x.dtype, device=x.device)
    dy = x[:, 7].mean(dim=(-2, -1))
    dx = x[:, 8].mean(dim=(-2, -1))
    bias[:, 0] = torch.relu(-dy)
    bias[:, 1] = torch.relu(dy)
    bias[:, 2] = torch.relu(-dx)
    bias[:, 3] = torch.relu(dx)

    if action_count > 6:
        adjacent_closed = x[:, 11].mean(dim=(-2, -1)) > 0.5
        # Adjacent waypoint has magnitude exactly 1/view_size on one axis. Allow a
        # small numerical margin but do not encourage toggling unrelated side doors.
        near_waypoint = (dy.abs() + dx.abs()) <= (1.25 / max(1, view_size))
        toggle = (adjacent_closed & near_waypoint).to(x.dtype)
        bias[:, 6] = toggle * 1.25
    return bias * float(scale)


class RouteContextEncoder(nn.Module):
    """Encode local geometry separately from low-bandwidth task/navigation context.

    ROOM-DOOR-TRANSPORT channels 0..5 are spatial (walls, doors, object, goal, agent). Channels
    6..13 are broadcast scalar state (carrying, hierarchical route waypoint, interaction
    affordances, nearby-door state, open-door fraction). Treating the latter as images
    wastes capacity and weakens tiny direction signals on large maps; this encoder keeps
    them explicit.
    """

    def __init__(self, observation_shape: tuple[int, int, int], feature_dim: int = 128) -> None:
        super().__init__()
        channels, height, width = observation_shape
        if channels < 14:
            raise ValueError("ROOM-DOOR-TRANSPORT route models require the 14-channel room observation.")
        self.spatial_channels = 6
        self.context_start = 6
        self.context_dim = channels - self.context_start

        self.spatial = nn.Sequential(
            nn.Conv2d(self.spatial_channels, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 48, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(48 * height * width, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(self.context_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        visual = self.spatial(x[:, : self.spatial_channels])
        # Broadcast task channels are constant over the crop by construction, so mean
        # pooling is exact and remains robust if a future channel contains tiny noise.
        context = x[:, self.context_start :].mean(dim=(-2, -1))
        context = self.context(context)
        return self.fusion(torch.cat([visual, context], dim=1))


class RouteFusionActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.view_size = int(observation_shape[1])
        self.encoder = RouteContextEncoder(observation_shape, 128)
        self.policy = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Linear(96, action_count))
        nn.init.normal_(self.policy[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy[-1].bias)
        self.value = nn.Sequential(nn.Linear(128, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        logits = self.policy(z) + route_prior_logits(x, self.action_count, self.view_size)
        return logits, self.value(z).squeeze(-1)


class RouteActionMemoryActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.view_size = int(observation_shape[1])
        self.history_length = 10
        self.pad_token = self.action_count
        self.encoder = RouteContextEncoder(observation_shape, 128)
        self.action_memory = LearnableActionMemory(
            action_count=self.action_count,
            history_length=self.history_length,
            embed_dim=16,
            memory_dim=48,
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 48, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Linear(160, 128),
            nn.GELU(),
        )
        self.policy = nn.Linear(128, action_count)
        nn.init.normal_(self.policy.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy.bias)
        self.value = nn.Linear(128, 1)
        self._action_history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.history_length))

    def _empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch, self.history_length), self.pad_token, dtype=torch.long, device=device)

    def _history_tensor(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw_id in env_ids:
            items = list(self._action_history[int(raw_id)])[-self.history_length :]
            rows.append([self.pad_token] * (self.history_length - len(items)) + items)
        return torch.tensor(rows, dtype=torch.long, device=device) if rows else self._empty_history(0, device)

    def _heads(self, x: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        route = self.encoder(x)
        memory = self.action_memory(history)
        z = self.fusion(torch.cat([route, memory], dim=1))
        logits = self.policy(z) + route_prior_logits(x, self.action_count, self.view_size)
        return logits, self.value(z).squeeze(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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


class RouteGRUActionMemoryActorCritic(nn.Module):
    """Structured route encoder + explicit action history + recurrent room memory."""

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.view_size = int(observation_shape[1])
        self.history_length = 8
        self.pad_token = self.action_count
        self.hidden_dim = 96
        self.encoder = RouteContextEncoder(observation_shape, 128)
        self.action_memory = LearnableActionMemory(
            action_count=self.action_count,
            history_length=self.history_length,
            embed_dim=12,
            memory_dim=32,
        )
        self.recurrent = nn.GRUCell(128 + 32, self.hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(128 + self.hidden_dim + 32, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Linear(192, 128),
            nn.GELU(),
        )
        self.policy = nn.Linear(128, action_count)
        nn.init.normal_(self.policy.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.policy.bias)
        self.value = nn.Linear(128, 1)
        self._hidden: dict[int, torch.Tensor] = {}
        self._action_history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.history_length))

    @property
    def context_dim(self) -> int:
        return self.hidden_dim + self.history_length

    def _empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch, self.history_length), self.pad_token, dtype=torch.long, device=device)

    def _history_for_envs(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw_id in env_ids:
            items = list(self._action_history[int(raw_id)])[-self.history_length :]
            rows.append([self.pad_token] * (self.history_length - len(items)) + items)
        return torch.tensor(rows, dtype=torch.long, device=device) if rows else self._empty_history(0, device)

    def _hidden_for_envs(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = [self._hidden.get(int(raw_id), torch.zeros(self.hidden_dim, device=device)).to(device) for raw_id in env_ids]
        return torch.stack(rows, dim=0) if rows else torch.zeros((0, self.hidden_dim), device=device)

    def _pack_context(self, hidden: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return torch.cat([hidden, history.float()], dim=1).detach()

    def _unpack_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = context[:, : self.hidden_dim].float()
        history = context[:, self.hidden_dim : self.hidden_dim + self.history_length].round().long()
        return hidden, history

    def _heads(self, x: torch.Tensor, hidden: torch.Tensor, history: torch.Tensor):
        route = self.encoder(x)
        action_mem = self.action_memory(history)
        hidden_new = self.recurrent(torch.cat([route, action_mem], dim=1), hidden)
        z = self.fusion(torch.cat([route, hidden_new, action_mem], dim=1))
        logits = self.policy(z) + route_prior_logits(x, self.action_count, self.view_size)
        return logits, self.value(z).squeeze(-1), hidden_new

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)
        history = self._empty_history(x.shape[0], x.device)
        logits, value, _ = self._heads(x, hidden, history)
        return logits, value

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        hidden = self._hidden_for_envs(env_ids, x.device)
        history = self._history_for_envs(env_ids, x.device)
        context = self._pack_context(hidden, history)
        logits, value, hidden_new = self._heads(x, hidden, history)
        if update:
            for i, raw_id in enumerate(env_ids):
                self._hidden[int(raw_id)] = hidden_new[i].detach()
        return logits, value, context

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        hidden, history = self._unpack_context(context)
        logits, value, _ = self._heads(x, hidden, history)
        return logits, value

    def training_sequence_forward(self, x_seq, initial_context, actions_seq, dones_seq):
        hidden, history = self._unpack_context(initial_context)
        logits_out = []
        values_out = []
        for t in range(x_seq.shape[0]):
            logits, value, hidden_new = self._heads(x_seq[t], hidden, history)
            logits_out.append(logits)
            values_out.append(value)

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

    def reset_sequence_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._hidden.pop(int(env_id), None)
