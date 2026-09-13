from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn

from tfp.models._action_memory_core import LearnableActionMemory

SPATIAL_CHANNELS = 13
CONTEXT_START = 13
CONTEXT_DIM = 30
OBS_CHANNELS = 43


def _context(x: torch.Tensor) -> torch.Tensor:
    return x[:, CONTEXT_START:].mean(dim=(-2, -1))


def dependency_safety_prior_logits(x: torch.Tensor, action_count: int, scale: float = 2.6) -> torch.Tensor:
    """Low-bandwidth hierarchical route prior plus a soft one-step safety shield.

    The prior uses only channels already present in the policy observation. It is not a
    planner action oracle: the network still chooses how to route around local obstacles,
    when to wait for hazards, and how strongly to override the prior.
    """
    b = x.shape[0]
    bias = torch.zeros((b, action_count), dtype=x.dtype, device=x.device)
    dy = x[:, 19].mean(dim=(-2, -1))
    dx = x[:, 20].mean(dim=(-2, -1))
    denom = torch.maximum(dy.abs(), dx.abs()).clamp_min(1e-4)
    ndy, ndx = dy / denom, dx / denom
    bias[:, 0] += torch.relu(-ndy)
    bias[:, 1] += torch.relu(ndy)
    bias[:, 2] += torch.relu(-ndx)
    bias[:, 3] += torch.relu(ndx)

    pickup = x[:, 24].mean(dim=(-2, -1)) > 0.5
    delivery = x[:, 25].mean(dim=(-2, -1)) > 0.5
    adjacent_closed = x[:, 26].mean(dim=(-2, -1)) > 0.5
    adjacent_unlockable = x[:, 27].mean(dim=(-2, -1)) > 0.5
    adjacent_locked = x[:, 28].mean(dim=(-2, -1)) > 0.5
    near_waypoint = (dy.abs() + dx.abs()) < 0.17
    if action_count > 4:
        bias[:, 4] += pickup.to(x.dtype) * 2.4
    if action_count > 5:
        bias[:, 5] += delivery.to(x.dtype) * 3.0
    if action_count > 6:
        toggle = (near_waypoint & (adjacent_closed | adjacent_unlockable) & ~adjacent_locked).to(x.dtype)
        bias[:, 6] += toggle * 2.1
        bias[:, 6] -= adjacent_locked.to(x.dtype) * 0.8

    # Motion-aware soft shield. Current wolf bearing is combined with observed
    # one-step velocity, so an approaching predator suppresses the cell it is moving
    # toward rather than only the cell it currently occupies. No future RNG target or
    # global hazard path is exposed to the policy.
    hazard_specs = ((29, 37, 38, 41), (32, 39, 40, 42))
    for base, vy_idx, vx_idx, closing_idx in hazard_specs:
        wy = x[:, base].mean(dim=(-2, -1))
        wx = x[:, base + 1].mean(dim=(-2, -1))
        dist = x[:, base + 2].mean(dim=(-2, -1))
        vy = x[:, vy_idx].mean(dim=(-2, -1))
        vx = x[:, vx_idx].mean(dim=(-2, -1))
        closing = torch.relu(x[:, closing_idx].mean(dim=(-2, -1)))
        pred_y = wy + 0.125 * vy
        pred_x = wx + 0.125 * vx
        closeness = torch.relu(0.62 - dist) / 0.62
        danger = closeness * (1.0 + 0.65 * closing)
        bias[:, 0] -= danger * torch.relu(-pred_y) * 2.8
        bias[:, 1] -= danger * torch.relu(pred_y) * 2.8
        bias[:, 2] -= danger * torch.relu(-pred_x) * 2.8
        bias[:, 3] -= danger * torch.relu(pred_x) * 2.8
        if action_count > 7:
            # Waiting is useful only as a timing action. An approaching predator makes
            # staying still less attractive; a nearby lateral hazard can make it safer.
            wait_support = torch.relu(0.34 - dist) / 0.34 * (1.0 - 0.8 * closing)
            bias[:, 7] += wait_support * 0.9

    # Explicitly suppress stepping onto a visible adjacent wolf.
    radius = x.shape[-1] // 2
    wolf = x[:, 11]
    if radius >= 1:
        bias[:, 0] -= wolf[:, radius - 1, radius] * 5.0
        bias[:, 1] -= wolf[:, radius + 1, radius] * 5.0
        bias[:, 2] -= wolf[:, radius, radius - 1] * 5.0
        bias[:, 3] -= wolf[:, radius, radius + 1] * 5.0
    return bias * float(scale)


class DependencyFiLMEncoder(nn.Module):
    """Local visual encoder modulated by symbolic access/progress/hazard context."""

    def __init__(self, observation_shape: tuple[int, int, int], feature_dim: int = 192) -> None:
        super().__init__()
        channels, height, width = observation_shape
        if channels != OBS_CHANNELS:
            raise ValueError(f"KEYED-HAZARD-LOGISTICS models require {OBS_CHANNELS} channels, got {channels}.")
        self.conv1 = nn.Sequential(nn.Conv2d(SPATIAL_CHANNELS, 40, 3, padding=1), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(40, 64, 3, padding=1), nn.GELU())
        self.film = nn.Sequential(nn.Linear(CONTEXT_DIM, 128), nn.GELU(), nn.Linear(128, 128))
        self.visual_head = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(), nn.Flatten(),
            nn.Linear(64 * height * width, 192), nn.LayerNorm(192), nn.GELU(),
        )
        self.context_head = nn.Sequential(nn.Linear(CONTEXT_DIM, 96), nn.LayerNorm(96), nn.GELU(), nn.Linear(96, 80), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(192 + 80, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = _context(x)
        h = self.conv2(self.conv1(x[:, :SPATIAL_CHANNELS]))
        gamma, beta = self.film(context).chunk(2, dim=1)
        h = h * (1.0 + 0.35 * torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1)) + 0.35 * beta.unsqueeze(-1).unsqueeze(-1)
        visual = self.visual_head(h)
        return self.fusion(torch.cat([visual, self.context_head(context)], dim=1))


class DependencyFiLMShieldActorCritic(nn.Module):
    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.encoder = DependencyFiLMEncoder(observation_shape, 192)
        self.policy = nn.Sequential(nn.Linear(192, 144), nn.GELU(), nn.Linear(144, action_count))
        self.value = nn.Sequential(nn.Linear(192, 144), nn.GELU(), nn.Linear(144, 1))
        nn.init.normal_(self.policy[-1].weight, mean=0.0, std=0.006)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        residual = 0.55 * torch.tanh(self.policy(z))
        return dependency_safety_prior_logits(x, self.action_count) + residual, self.value(z).squeeze(-1)


class EventMemoryTransformerActorCritic(nn.Module):
    """Sparse compact-state history rather than recurrent visual-frame memory."""

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.history_length = 10
        self.telemetry_dim = CONTEXT_DIM
        self.encoder = DependencyFiLMEncoder(observation_shape, 176)
        self.token = nn.Linear(self.telemetry_dim + 1, 72)
        self.positional = nn.Parameter(torch.zeros(1, self.history_length, 72))
        layer = nn.TransformerEncoderLayer(
            d_model=72, nhead=4, dim_feedforward=160, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.fusion = nn.Sequential(nn.Linear(176 + 72, 216), nn.LayerNorm(216), nn.GELU(), nn.Linear(216, 160), nn.GELU())
        self.policy = nn.Linear(160, action_count)
        self.value = nn.Linear(160, 1)
        nn.init.normal_(self.policy.weight, mean=0.0, std=0.006); nn.init.zeros_(self.policy.bias)
        self._history: dict[int, deque[torch.Tensor]] = defaultdict(lambda: deque(maxlen=self.history_length))

    def _empty(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch, self.history_length, self.telemetry_dim + 1), dtype=torch.float32, device=device)

    def _history_tensor(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        out = self._empty(len(env_ids), device)
        for i, raw in enumerate(env_ids):
            vals = list(self._history[int(raw)])[-self.history_length:]
            if vals:
                start = self.history_length - len(vals)
                out[i, start:, :self.telemetry_dim] = torch.stack([v.to(device) for v in vals], dim=0)
                out[i, start:, self.telemetry_dim] = 1.0
        return out

    def _heads(self, x: torch.Tensor, history: torch.Tensor):
        current = self.encoder(x)
        present = history[..., self.telemetry_dim]
        encoded = self.temporal(self.token(history) + self.positional)
        denom = present.sum(dim=1, keepdim=True).clamp_min(1.0)
        memory = (encoded * present.unsqueeze(-1)).sum(dim=1) / denom
        z = self.fusion(torch.cat([current, memory], dim=1))
        logits = dependency_safety_prior_logits(x, self.action_count) + 0.55 * torch.tanh(self.policy(z))
        return logits, self.value(z).squeeze(-1)

    def forward(self, x: torch.Tensor):
        return self._heads(x, self._empty(x.shape[0], x.device))

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        history = self._history_tensor(env_ids, x.device)
        logits, value = self._heads(x, history)
        if update:
            telemetry = _context(x).detach().cpu()
            for i, raw in enumerate(env_ids):
                self._history[int(raw)].append(telemetry[i])
        return logits, value, history.detach()

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        return self._heads(x, context.float())

    def observe_action_memory(self, env_ids, actions, rewards, dones) -> None:
        del actions, rewards
        for raw, done in zip(env_ids, dones):
            if done:
                self._history.pop(int(raw), None)

    def reset_sequence_memory(self, env_ids: Sequence[int]) -> None:
        for raw in env_ids:
            self._history.pop(int(raw), None)


class DualTimescaleGRUActorCritic(nn.Module):
    """Long task-state memory + short hazard dynamics + explicit action history."""

    TASK_CONTEXT = tuple(range(13, 29)) + (35, 36)
    HAZARD_CONTEXT = tuple(range(29, 35)) + tuple(range(37, 43))

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.history_length = 10
        self.pad_token = self.action_count
        self.task_hidden = 112
        self.hazard_hidden = 64
        self.encoder = DependencyFiLMEncoder(observation_shape, 168)
        self.action_memory = LearnableActionMemory(action_count, self.history_length, embed_dim=12, memory_dim=36)
        self.task_gru = nn.GRUCell(len(self.TASK_CONTEXT) + 36, self.task_hidden)
        self.hazard_gru = nn.GRUCell(len(self.HAZARD_CONTEXT) + 36, self.hazard_hidden)
        self.fusion = nn.Sequential(
            nn.Linear(168 + self.task_hidden + self.hazard_hidden + 36, 288), nn.LayerNorm(288), nn.GELU(),
            nn.Linear(288, 184), nn.GELU(),
        )
        self.policy = nn.Linear(184, action_count)
        self.value = nn.Linear(184, 1)
        nn.init.normal_(self.policy.weight, mean=0.0, std=0.006); nn.init.zeros_(self.policy.bias)
        self._task: dict[int, torch.Tensor] = {}
        self._hazard: dict[int, torch.Tensor] = {}
        self._actions: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.history_length))

    def _empty_actions(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full((batch, self.history_length), self.pad_token, dtype=torch.long, device=device)

    def _actions_for(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw in env_ids:
            vals = list(self._actions[int(raw)])[-self.history_length:]
            rows.append([self.pad_token] * (self.history_length - len(vals)) + vals)
        return torch.tensor(rows, dtype=torch.long, device=device) if rows else self._empty_actions(0, device)

    def _hidden_for(self, store: dict[int, torch.Tensor], dim: int, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = [store.get(int(raw), torch.zeros(dim, device=device)).to(device) for raw in env_ids]
        return torch.stack(rows, 0) if rows else torch.zeros((0, dim), device=device)

    def _pack(self, task: torch.Tensor, hazard: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.cat([task, hazard, actions.float()], dim=1).detach()

    def _unpack(self, context: torch.Tensor):
        a = self.task_hidden; b = a + self.hazard_hidden
        return context[:, :a].float(), context[:, a:b].float(), context[:, b:b+self.history_length].round().long()

    @staticmethod
    def _select_context(x: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
        return torch.stack([x[:, i].mean(dim=(-2, -1)) for i in indices], dim=1)

    def _heads(self, x: torch.Tensor, task: torch.Tensor, hazard: torch.Tensor, actions: torch.Tensor):
        visual = self.encoder(x)
        amem = self.action_memory(actions)
        task_new = self.task_gru(torch.cat([self._select_context(x, self.TASK_CONTEXT), amem], 1), task)
        hazard_new = self.hazard_gru(torch.cat([self._select_context(x, self.HAZARD_CONTEXT), amem], 1), hazard)
        z = self.fusion(torch.cat([visual, task_new, hazard_new, amem], 1))
        logits = dependency_safety_prior_logits(x, self.action_count) + 0.55 * torch.tanh(self.policy(z))
        return logits, self.value(z).squeeze(-1), task_new, hazard_new

    def forward(self, x: torch.Tensor):
        task = torch.zeros((x.shape[0], self.task_hidden), device=x.device)
        hazard = torch.zeros((x.shape[0], self.hazard_hidden), device=x.device)
        actions = self._empty_actions(x.shape[0], x.device)
        logits, value, _, _ = self._heads(x, task, hazard, actions)
        return logits, value

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        task = self._hidden_for(self._task, self.task_hidden, env_ids, x.device)
        hazard = self._hidden_for(self._hazard, self.hazard_hidden, env_ids, x.device)
        actions = self._actions_for(env_ids, x.device)
        context = self._pack(task, hazard, actions)
        logits, value, task_new, hazard_new = self._heads(x, task, hazard, actions)
        if update:
            for i, raw in enumerate(env_ids):
                self._task[int(raw)] = task_new[i].detach()
                self._hazard[int(raw)] = hazard_new[i].detach()
        return logits, value, context

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        task, hazard, actions = self._unpack(context)
        logits, value, _, _ = self._heads(x, task, hazard, actions)
        return logits, value

    def training_sequence_forward(self, x_seq, initial_context, actions_seq, dones_seq):
        task, hazard, actions = self._unpack(initial_context)
        logits_out, values_out = [], []
        for t in range(x_seq.shape[0]):
            logits, value, task_new, hazard_new = self._heads(x_seq[t], task, hazard, actions)
            logits_out.append(logits); values_out.append(value)
            actions = torch.cat([actions[:, 1:], actions_seq[t].long().unsqueeze(1)], dim=1)
            done = dones_seq[t].bool().unsqueeze(1)
            task = torch.where(done, torch.zeros_like(task_new), task_new)
            hazard = torch.where(done, torch.zeros_like(hazard_new), hazard_new)
            actions = torch.where(done.expand_as(actions), torch.full_like(actions, self.pad_token), actions)
        return torch.stack(logits_out, 0), torch.stack(values_out, 0)

    def observe_action_memory(self, env_ids, actions, rewards, dones) -> None:
        del rewards
        for raw, action, done in zip(env_ids, actions, dones):
            key = int(raw)
            self._actions[key].append(int(action))
            if done:
                self._actions.pop(key, None); self._task.pop(key, None); self._hazard.pop(key, None)

    def reset_action_memory(self, env_ids: Sequence[int]) -> None:
        for raw in env_ids:
            self._actions.pop(int(raw), None)

    def reset_sequence_memory(self, env_ids: Sequence[int]) -> None:
        for raw in env_ids:
            self._task.pop(int(raw), None); self._hazard.pop(int(raw), None)
