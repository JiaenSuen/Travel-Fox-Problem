from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn


class CompactVisualEncoder(nn.Module):
    """Visual stack intentionally aligned with 001_simple_cnn for weight transfer."""

    def __init__(self, observation_shape: tuple[int, int, int], feature_dim: int = 128) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.net = nn.Sequential(
            nn.Conv2d(channels, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * height * width, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactActionHistory(nn.Module):
    def __init__(self, action_count: int, history_length: int = 6, output_dim: int = 24) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.history_length = int(history_length)
        self.pad_token = int(action_count)
        self.embedding = nn.Embedding(action_count + 1, 8, padding_idx=self.pad_token)
        self.project = nn.Sequential(
            nn.Linear(history_length * 8, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, output_dim),
            nn.Tanh(),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(history.long().clamp(0, self.pad_token))
        return self.project(emb.flatten(1))


class CompactRecurrentActorCritic(nn.Module):
    """Small recurrent PPO actor-critic with exact rollout-context replay.

    The old in-run staged-switch experiment is intentionally not used. A bootstrap model can
    start from the fully trained 001 Simple CNN weights. The recurrent contribution is
    residual and smoothly ramped, so adding memory cannot abruptly destroy the mature
    visual policy at the beginning of fine-tuning.
    """

    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        action_count: int,
        *,
        cell_type: str,
        staged: bool = False,
        action_memory: bool = False,
        warmup_fraction: float = 0.30,
        bootstrap_mode: bool = False,
    ) -> None:
        super().__init__()
        if cell_type not in {"gru", "lstm"}:
            raise ValueError("cell_type must be 'gru' or 'lstm'.")
        self.cell_type = cell_type
        self.action_count = int(action_count)
        self.bootstrap_mode = bool(bootstrap_mode)
        self.feature_dim = 128 if self.bootstrap_mode else 96
        self.hidden_dim = 64
        self.staged = bool(staged)  # retained only for legacy checkpoint/config parsing
        self.action_memory_enabled = bool(action_memory)
        self.warmup_fraction = float(warmup_fraction)
        self.recurrent_enabled = True
        self.history_length = 6
        self.pad_token = int(action_count)
        self.memory_scale = 0.35
        self._training_progress = 0.0

        self.encoder = CompactVisualEncoder(observation_shape, self.feature_dim)
        self.action_memory = CompactActionHistory(action_count, self.history_length, 24) if action_memory else None
        recurrent_input = self.feature_dim + (24 if action_memory else 0)
        if cell_type == "gru":
            self.recurrent = nn.GRUCell(recurrent_input, self.hidden_dim)
        else:
            self.recurrent = nn.LSTMCell(recurrent_input, self.hidden_dim)
        self.memory_adapter = nn.Linear(self.hidden_dim, self.feature_dim)
        nn.init.zeros_(self.memory_adapter.weight)
        nn.init.zeros_(self.memory_adapter.bias)
        self.policy = nn.Linear(self.feature_dim, action_count)
        self.value = nn.Linear(self.feature_dim, 1)

        self._hidden: dict[int, torch.Tensor] = {}
        self._cell: dict[int, torch.Tensor] = {}
        self._action_history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=self.history_length))

    @property
    def context_dim(self) -> int:
        base = self.hidden_dim * (2 if self.cell_type == "lstm" else 1)
        return base + (self.history_length if self.action_memory_enabled else 0)

    def initialize_from_feedforward_state(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Transfer 001_simple_cnn encoder and actor/value heads exactly.

        The baseline uses ``encoder.N`` while this class wraps the same layers under
        ``encoder.net.N``. Shapes are checked by ``load_state_dict`` rather than silently
        truncating incompatible tensors.
        """
        mapped: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                mapped[f"encoder.net.{key[len('encoder.'):]}"] = value
            elif key.startswith("policy.") or key.startswith("value."):
                mapped[key] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        allowed_missing = {
            key for key in self.state_dict()
            if key.startswith("recurrent.") or key.startswith("memory_adapter.") or key.startswith("action_memory.")
        }
        real_missing = [key for key in missing if key not in allowed_missing]
        if real_missing or unexpected:
            raise ValueError(
                f"Bootstrap state mismatch: missing={real_missing}, unexpected={list(unexpected)}"
            )

    def set_training_progress(self, progress: float) -> None:
        progress = float(max(0.0, min(1.0, progress)))
        self._training_progress = progress

        # Legacy staged mode is intentionally no longer used by active plugins. Keeping
        # the branch avoids crashing a legacy config while making the behavior explicit.
        if self.staged and not self.bootstrap_mode:
            self.recurrent_enabled = progress >= self.warmup_fraction
        else:
            self.recurrent_enabled = True

        for p in self.recurrent.parameters():
            p.requires_grad_(self.recurrent_enabled)
        for p in self.memory_adapter.parameters():
            p.requires_grad_(self.recurrent_enabled)
        if self.action_memory is not None:
            for p in self.action_memory.parameters():
                p.requires_grad_(self.recurrent_enabled)

        # For bootstrap fine-tuning, protect the mature visual policy briefly while the
        # new recurrent residual learns a useful scale. Afterwards all components adapt.
        freeze_visual = self.bootstrap_mode and progress < 0.10
        for p in self.encoder.parameters():
            p.requires_grad_(not freeze_visual)
        for p in self.policy.parameters():
            p.requires_grad_(not freeze_visual)
        for p in self.value.parameters():
            p.requires_grad_(not freeze_visual)

    def _residual_scale(self) -> float:
        if not self.bootstrap_mode:
            return self.memory_scale
        # Smoothly introduce memory during the first 25% of recurrent fine-tuning.
        ramp = min(1.0, self._training_progress / 0.25)
        return self.memory_scale * ramp

    def _empty_state(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = torch.zeros((batch, self.hidden_dim), dtype=torch.float32, device=device)
        c = torch.zeros_like(h) if self.cell_type == "lstm" else None
        return h, c

    def _state_for_envs(self, env_ids: Sequence[int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
        hs = []
        cs = []
        for raw_id in env_ids:
            key = int(raw_id)
            hs.append(self._hidden.get(key, torch.zeros(self.hidden_dim, device=device)).to(device))
            if self.cell_type == "lstm":
                cs.append(self._cell.get(key, torch.zeros(self.hidden_dim, device=device)).to(device))
        if not hs:
            return self._empty_state(0, device)
        h = torch.stack(hs, dim=0)
        c = torch.stack(cs, dim=0) if self.cell_type == "lstm" else None
        return h, c

    def _history_for_envs(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw_id in env_ids:
            items = list(self._action_history[int(raw_id)])[-self.history_length:]
            rows.append([self.pad_token] * (self.history_length - len(items)) + items)
        if not rows:
            return torch.full((0, self.history_length), self.pad_token, dtype=torch.long, device=device)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def _pack_context(self, h: torch.Tensor, c: torch.Tensor | None, history: torch.Tensor | None) -> torch.Tensor:
        parts = [h]
        if c is not None:
            parts.append(c)
        if history is not None:
            parts.append(history.float())
        return torch.cat(parts, dim=1).detach()

    def _unpack_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        cursor = 0
        h = context[:, cursor: cursor + self.hidden_dim].float()
        cursor += self.hidden_dim
        c = None
        if self.cell_type == "lstm":
            c = context[:, cursor: cursor + self.hidden_dim].float()
            cursor += self.hidden_dim
        history = None
        if self.action_memory_enabled:
            history = context[:, cursor: cursor + self.history_length].round().long()
        return h, c, history

    def _heads_from_context(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor | None,
        history: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        visual = self.encoder(x)
        if not self.recurrent_enabled:
            return self.policy(visual), self.value(visual).squeeze(-1), h_prev, c_prev

        recurrent_input = visual
        if self.action_memory is not None:
            if history is None:
                history = torch.full(
                    (x.shape[0], self.history_length), self.pad_token, dtype=torch.long, device=x.device
                )
            recurrent_input = torch.cat([visual, self.action_memory(history)], dim=1)

        if self.cell_type == "gru":
            h_new = self.recurrent(recurrent_input, h_prev)
            c_new = None
        else:
            if c_prev is None:
                c_prev = torch.zeros_like(h_prev)
            h_new, c_new = self.recurrent(recurrent_input, (h_prev, c_prev))

        memory_delta = torch.tanh(self.memory_adapter(h_new))
        fused = visual + self._residual_scale() * memory_delta
        return self.policy(fused), self.value(fused).squeeze(-1), h_new, c_new

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = self._empty_state(x.shape[0], x.device)
        history = None
        if self.action_memory_enabled:
            history = torch.full((x.shape[0], self.history_length), self.pad_token, dtype=torch.long, device=x.device)
        logits, value, _, _ = self._heads_from_context(x, h, c, history)
        return logits, value

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        h_prev, c_prev = self._state_for_envs(env_ids, x.device)
        history = self._history_for_envs(env_ids, x.device) if self.action_memory_enabled else None
        context = self._pack_context(h_prev, c_prev, history)
        logits, value, h_new, c_new = self._heads_from_context(x, h_prev, c_prev, history)
        if update and self.recurrent_enabled:
            for i, raw_id in enumerate(env_ids):
                key = int(raw_id)
                self._hidden[key] = h_new[i].detach()
                if c_new is not None:
                    self._cell[key] = c_new[i].detach()
        return logits, value, context

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        h_prev, c_prev, history = self._unpack_context(context)
        logits, value, _, _ = self._heads_from_context(x, h_prev, c_prev, history)
        return logits, value

    def training_sequence_forward(
        self,
        x_seq: torch.Tensor,
        initial_context: torch.Tensor,
        actions_seq: torch.Tensor,
        dones_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c, history = self._unpack_context(initial_context)
        logits_out = []
        values_out = []
        for t in range(x_seq.shape[0]):
            logits, value, h_new, c_new = self._heads_from_context(x_seq[t], h, c, history)
            logits_out.append(logits)
            values_out.append(value)

            if self.action_memory_enabled:
                assert history is not None
                history = torch.cat([history[:, 1:], actions_seq[t].long().unsqueeze(1)], dim=1)

            done = dones_seq[t].bool().unsqueeze(1)
            h = torch.where(done, torch.zeros_like(h_new), h_new)
            if self.cell_type == "lstm":
                assert c_new is not None
                c = torch.where(done, torch.zeros_like(c_new), c_new)
            if self.action_memory_enabled:
                pad = torch.full_like(history, self.pad_token)
                history = torch.where(done.expand_as(history), pad, history)

        return torch.stack(logits_out, dim=0), torch.stack(values_out, dim=0)

    def observe_action_memory(self, env_ids, actions, rewards, dones) -> None:
        del rewards
        for env_id, action, done in zip(env_ids, actions, dones):
            key = int(env_id)
            if self.action_memory_enabled:
                self._action_history[key].append(int(action))
            if done:
                self._action_history.pop(key, None)
                self._hidden.pop(key, None)
                self._cell.pop(key, None)

    def reset_action_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._action_history.pop(int(env_id), None)

    def reset_sequence_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            key = int(env_id)
            self._hidden.pop(key, None)
            self._cell.pop(key, None)
