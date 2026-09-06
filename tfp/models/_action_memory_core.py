from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn

class LearnableActionMemory(nn.Module):
    """Encode a short action sequence with embeddings and a tiny temporal convolution.

    The memory state is discrete action IDs, not stored images or large recurrent
    hidden states. Padding uses a dedicated start token, making episode resets explicit.
    """

    def __init__(self, action_count: int, history_length: int = 8, embed_dim: int = 16, memory_dim: int = 48) -> None:
        super().__init__()
        self.action_count = action_count
        self.history_length = history_length
        self.pad_token = action_count
        self.embedding = nn.Embedding(action_count + 1, embed_dim, padding_idx=self.pad_token)
        self.position = nn.Parameter(torch.zeros(1, history_length, embed_dim))
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        self.temporal = nn.Sequential(
            nn.Conv1d(embed_dim, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1, groups=4),
            nn.GELU(),
        )
        self.attention_score = nn.Linear(32, 1)
        self.output = nn.Sequential(
            nn.Linear(32, memory_dim),
            nn.LayerNorm(memory_dim),
            nn.GELU(),
        )

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        history = action_history.long().clamp(0, self.pad_token)
        tokens = self.embedding(history) + self.position[:, : history.shape[1]]
        encoded = self.temporal(tokens.transpose(1, 2)).transpose(1, 2)

        # Learned pooling emphasizes behavior motifs while ignoring padded positions.
        scores = self.attention_score(encoded).squeeze(-1)
        valid = history.ne(self.pad_token)
        scores = scores.masked_fill(~valid, -1e9)
        all_pad = ~valid.any(dim=1)
        weights = torch.softmax(scores, dim=1)
        if bool(all_pad.any()):
            weights = weights.clone()
            weights[all_pad] = 0.0
        pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.output(pooled)


class ActionMemoryCNN(nn.Module):
    """CNN policy/value network conditioned on a learnable history of past actions.

    The environment observation interface is unchanged. For each environment instance,
    the model stores only the last eight executed action IDs. Those IDs are passed
    through a learnable embedding + tiny temporal convolution, then fused into the CNN
    visual representation through a gated residual path. PPO stores the exact action
    history used at decision time, so shuffled minibatches reconstruct the same policy
    distribution during optimization without an LSTM or image-frame buffer.
    """

    def __init__(self, observation_shape: tuple[int, int, int], action_count: int) -> None:
        super().__init__()
        channels, height, width = observation_shape
        self.action_count = action_count
        self.history_length = 8
        self.pad_token = action_count
        self.visual_dim = 128
        self.memory_dim = 48

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 24, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * height * width, self.visual_dim),
            nn.ReLU(inplace=True),
        )
        self.action_memory = LearnableActionMemory(
            action_count=action_count,
            history_length=self.history_length,
            embed_dim=16,
            memory_dim=self.memory_dim,
        )
        # Residual memory adapter: the visual path is architecturally identical to
        # Simple CNN 001. The adapter is zero-initialized so the experiment starts
        # from a pure visual policy and learns only as action history proves useful.
        self.memory_adapter = nn.Linear(self.memory_dim, self.visual_dim)
        nn.init.zeros_(self.memory_adapter.weight)
        nn.init.zeros_(self.memory_adapter.bias)
        self.policy = nn.Linear(self.visual_dim, action_count)
        self.value = nn.Linear(self.visual_dim, 1)

        self._action_history: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=self.history_length)
        )

    def _empty_history(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.full(
            (batch, self.history_length),
            self.pad_token,
            dtype=torch.long,
            device=device,
        )

    def _history_tensor(self, env_ids: Sequence[int], device: torch.device) -> torch.Tensor:
        rows = []
        for raw_env_id in env_ids:
            items = list(self._action_history[int(raw_env_id)])[-self.history_length :]
            padded = [self.pad_token] * (self.history_length - len(items)) + items
            rows.append(padded)
        if not rows:
            return self._empty_history(0, device)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def _heads(self, x: torch.Tensor, action_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.encoder(x)
        memory = self.action_memory(action_history)
        memory_delta = torch.tanh(self.memory_adapter(memory))
        # A small residual scale makes action memory an augmentation rather than a
        # replacement for visual navigation. This keeps the model compact and makes
        # ablation against Simple CNN 001 easier to interpret.
        fused = visual + 0.30 * memory_delta
        return self.policy(fused), self.value(fused).squeeze(-1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        history = self._empty_history(x.shape[0], x.device)
        return self._heads(x, history)

    def act_forward(self, x: torch.Tensor, env_ids: Sequence[int], update: bool = True):
        # ``update`` is intentionally ignored: the history changes only after the
        # action is actually executed through ``observe_action_memory``.
        history = self._history_tensor(env_ids, x.device)
        logits, value = self._heads(x, history)
        return logits, value, history.detach()

    def training_forward(self, x: torch.Tensor, context: torch.Tensor):
        return self._heads(x, context.long())

    def observe_action_memory(
        self,
        env_ids: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        del rewards  # Action memory intentionally stores behavior, not reward history.
        for env_id, action, done in zip(env_ids, actions, dones):
            key = int(env_id)
            self._action_history[key].append(int(action))
            if done:
                self._action_history.pop(key, None)

    def reset_action_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            self._action_history.pop(int(env_id), None)


