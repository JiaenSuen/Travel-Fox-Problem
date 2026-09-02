from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

import torch
from torch import nn

MODEL_SPEC = {
    "key": "simple_cnn_tabu_002",
    "display_name": "Simple CNN + Tabu 002",
    "algorithm": "ppo",
    "recurrent": False,
    "memory_type": "inference_only_behavior_cycle",
    "description": (
        "Simple CNN 001 trained as a normal PPO network. During greedy inference only, "
        "a model-owned short action-history filter detects repeated A-B, pickup-drop, "
        "single-action, and short A-B-C cycles and rewrites cycle-continuing decisions."
    ),
}


class SimpleCNNTabu002(nn.Module):
    """Feed-forward CNN with an inference-only short behavior-cycle filter.

    The visual input, trainable CNN, policy logits, and value output are identical in
    shape to ``simple_cnn_001``. PPO rollout sampling is intentionally untouched. The
    only addition is a non-neural post-decision controller used by greedy inference;
    it remembers executed actions and detects periodic short motifs
    without reading the hidden map, robot position, or oracle distance.

    When a repeated motif is found, actions that would keep the robot inside the same
    motif are temporarily discouraged. If a legal action outside the motif exists,
    the whole motif is tabu for the current escape decision. If the robot is in a
    corridor where every legal action belongs to the motif, only the *next expected*
    cycle action is tabu, which encourages commitment in one direction instead of
    left-right oscillation. If there is literally no legal alternative, the rule is
    relaxed so the heuristic can never deadlock the agent.
    """

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
        self.action_count = action_count

        # A short window is enough to identify local oscillation while keeping the
        # mechanism intentionally small and interpretable.
        self.history_window = 12
        self.max_period = 3
        self.escape_horizon = 4
        self.recency_penalty = 0.35

        self._history: dict[int, deque[tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=self.history_window)
        )
        self._escape_motif: dict[int, tuple[int, ...]] = {}
        self._escape_steps: dict[int, int] = defaultdict(int)
        self._cycle_events: dict[int, int] = defaultdict(int)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.policy(features), self.value(features).squeeze(-1)

    def reset_behavior_memory(self, env_ids: Sequence[int]) -> None:
        for env_id in env_ids:
            env_id = int(env_id)
            self._history.pop(env_id, None)
            self._escape_motif.pop(env_id, None)
            self._escape_steps.pop(env_id, None)
            self._cycle_events.pop(env_id, None)

    def observe_behavior(
        self,
        env_ids: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        """Remember executed actions; reward is retained only for inspection.

        Cycle detection is deliberately *not* cancelled by a positive reward. In Fox
        Transport, PICKUP has a positive reward, so using ``reward > 0`` as a proxy for
        task progress would incorrectly approve PICKUP→DROP→PICKUP→DROP loops.
        """
        for env_id, action, reward, done in zip(env_ids, actions, rewards, dones):
            env_id = int(env_id)
            self._history[env_id].append((int(action), float(reward)))
            if done:
                self.reset_behavior_memory((env_id,))

    @staticmethod
    def _periodic_suffix(actions: Sequence[int], max_period: int = 3) -> tuple[int, ...] | None:
        """Return the shortest repeated suffix motif, e.g. AB in ABAB or ABC in ABCABC."""
        actions = tuple(int(a) for a in actions)
        for period in range(1, max_period + 1):
            # Single-action repetition needs four decisions; longer motifs need two
            # complete repetitions. This catches A-B-A-B before it becomes persistent.
            repeats = 4 if period == 1 else 2
            need = period * repeats
            if len(actions) < need:
                continue
            suffix = actions[-need:]
            motif = suffix[-period:]
            if all(suffix[i] == motif[i % period] for i in range(need)):
                return motif
        return None

    def _cycle_state(self, env_id: int) -> tuple[tuple[int, ...] | None, int | None]:
        history_actions = [a for a, _ in self._history[env_id]]
        motif = self._periodic_suffix(history_actions, self.max_period)
        if motif is None:
            return None, None
        # The suffix ends on a complete motif repetition, so continuing the same cycle
        # would start at motif[0] again.
        return motif, int(motif[0])

    def cycle_event_count(self, env_id: int) -> int:
        """Small inspection hook useful in tests and research diagnostics."""
        return int(self._cycle_events.get(int(env_id), 0))

    def rewrite_actions(
        self,
        proposed: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
        env_ids: Sequence[int],
    ) -> torch.Tensor:
        """Rewrite a cycle-continuing decision while preserving the CNN I/O contract.

        The rule is action-agnostic: 2↔3 (LEFT/RIGHT), 0↔1 (UP/DOWN), 4↔5
        (PICKUP/DROP), repeated single actions, and short three-action motifs are all
        handled by the same periodicity detector.
        """
        final = proposed.clone()
        detached_logits = logits.detach()

        for row, raw_env_id in enumerate(env_ids):
            env_id = int(raw_env_id)
            legal = mask[row].clone()
            if not bool(legal.any()):
                continue

            detected_motif, expected_next = self._cycle_state(env_id)
            if detected_motif is not None:
                previous = self._escape_motif.get(env_id)
                if previous != detected_motif or self._escape_steps.get(env_id, 0) <= 0:
                    self._cycle_events[env_id] += 1
                self._escape_motif[env_id] = detected_motif
                self._escape_steps[env_id] = self.escape_horizon

            motif = self._escape_motif.get(env_id)
            escape_steps = int(self._escape_steps.get(env_id, 0))
            candidate = int(proposed[row].item())
            if motif is None or escape_steps <= 0:
                continue

            motif_set = {a for a in motif if 0 <= a < legal.numel()}
            scores = detached_logits[row].clone()

            # Prefer an action outside the repeated motif whenever the environment
            # offers one. This is what stops PICKUP↔DROP and small-area action loops.
            outside = legal.clone()
            for action in motif_set:
                outside[action] = False

            if bool(outside.any()):
                allowed = outside
            else:
                # In a narrow corridor there may be no third action. Blocking all A/B
                # would deadlock the robot, so block only the action that continues the
                # detected periodic phase. This turns L-R-L-R into directional
                # commitment (for example R-R-...) until a new option becomes visible.
                allowed = legal.clone()
                if expected_next is not None and 0 <= expected_next < allowed.numel():
                    allowed[expected_next] = False
                if not bool(allowed.any()):
                    allowed = legal

            recent = [a for a, _ in list(self._history[env_id])[-6:]]
            for action in range(scores.numel()):
                scores[action] -= self.recency_penalty * recent.count(action)
            scores = scores.masked_fill(~allowed, -1e9)
            best = int(scores.argmax().item())

            # Rewrite whenever the original proposal would continue/re-enter the
            # repeated motif and an alternative exists. If the proposal is already an
            # escape action, preserve the CNN decision.
            if candidate in motif_set and bool(allowed[best]):
                final[row] = best

            self._escape_steps[env_id] = max(0, escape_steps - 1)
            if int(final[row].item()) not in motif_set:
                # A genuine escape action ends the hard phase immediately; history
                # remains available so a new repeated basin can still be detected.
                self._escape_steps[env_id] = 0
                self._escape_motif.pop(env_id, None)

        return final


def create_model(observation_shape: tuple[int, int, int], action_count: int) -> nn.Module:
    return SimpleCNNTabu002(observation_shape, action_count)
