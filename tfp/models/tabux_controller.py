from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Sequence

import torch

# Local Transport action semantics. TabuX intentionally uses only executed actions
# and the legal-action mask; it never reads the hidden map or oracle distance.
_MOVE_DELTA = {
    0: (0, -1),  # UP
    1: (0, 1),   # DOWN
    2: (-1, 0),  # LEFT
    3: (1, 0),   # RIGHT
}
_PICKUP = 4
_DROP = 5

BehaviorState = tuple[int, int, int]  # relative x, relative y, estimated carrying flag
Transition = tuple[BehaviorState, int, BehaviorState]


@dataclass
class _EnvTabuState:
    x: int = 0
    y: int = 0
    carrying: bool = False
    recent_states: deque[BehaviorState] = field(default_factory=lambda: deque([(0, 0, 0)], maxlen=18))
    transitions: deque[Transition] = field(default_factory=lambda: deque(maxlen=18))
    recent_actions: deque[int] = field(default_factory=lambda: deque(maxlen=18))
    active_basin: set[BehaviorState] = field(default_factory=set)
    basin_tenure: int = 0
    tabu_ttl: dict[tuple[BehaviorState, int], int] = field(default_factory=dict)
    cycle_events: int = 0

    @property
    def state(self) -> BehaviorState:
        return (self.x, self.y, int(self.carrying))


class TabuXController:
    """Inference-only short-term Tabu Search controller for local behavior loops.

    TabuX goes beyond action-pattern suppression. It integrates executed grid actions
    into a tiny *behavior coordinate* and remembers a short window of recently visited
    behavior states and transitions. When the agent repeatedly revisits a 1-4 state
    local basin, transitions inside that basin become tabu for several decisions. The
    basin remains active after the first escape step, so a move that immediately
    returns to the problematic region is also suppressed. A soft revisit penalty makes
    unexplored/less-recent alternatives preferable before a hard cycle forms.

    This controller is deliberately inference-only and model-owned. It does not use
    the hidden map, agent coordinates from the environment, oracle distance, or extra
    observation channels. If every legal action is tabu, aspiration relaxes the rule
    and selects the least-revisited legal alternative, preventing controller deadlock.
    """

    def __init__(
        self,
        action_count: int,
        history_window: int = 18,
        max_basin_states: int = 4,
        basin_tenure: int = 9,
        transition_tenure: int = 7,
        revisit_penalty: float = 0.65,
        transition_penalty: float = 0.30,
        action_recency_penalty: float = 0.10,
    ) -> None:
        self.action_count = int(action_count)
        self.history_window = int(history_window)
        self.max_basin_states = int(max_basin_states)
        self.default_basin_tenure = int(basin_tenure)
        self.transition_tenure = int(transition_tenure)
        self.revisit_penalty = float(revisit_penalty)
        self.transition_penalty = float(transition_penalty)
        self.action_recency_penalty = float(action_recency_penalty)
        self._env: dict[int, _EnvTabuState] = defaultdict(self._new_state)

    def _new_state(self) -> _EnvTabuState:
        state = _EnvTabuState()
        state.recent_states = deque([(0, 0, 0)], maxlen=self.history_window)
        state.transitions = deque(maxlen=self.history_window)
        state.recent_actions = deque(maxlen=self.history_window)
        return state

    @staticmethod
    def _predict(state: BehaviorState, action: int) -> BehaviorState:
        x, y, carrying = state
        if action in _MOVE_DELTA:
            dx, dy = _MOVE_DELTA[action]
            return (x + dx, y + dy, carrying)
        if action == _PICKUP:
            return (x, y, 1)
        if action == _DROP:
            return (x, y, 0)
        return state

    def reset(self, env_ids: Sequence[int]) -> None:
        for raw_env_id in env_ids:
            self._env.pop(int(raw_env_id), None)

    def cycle_event_count(self, env_id: int) -> int:
        return int(self._env[int(env_id)].cycle_events)

    def _decay(self, state: _EnvTabuState) -> None:
        expired = []
        for key, ttl in state.tabu_ttl.items():
            ttl -= 1
            if ttl <= 0:
                expired.append(key)
            else:
                state.tabu_ttl[key] = ttl
        for key in expired:
            state.tabu_ttl.pop(key, None)
        if state.basin_tenure > 0:
            state.basin_tenure -= 1
            if state.basin_tenure <= 0:
                state.active_basin.clear()

    def _detect_local_basin(self, state: _EnvTabuState) -> set[BehaviorState]:
        """Find repeatedly visited behavior states in the recent short-term window."""
        counts = Counter(state.recent_states)
        repeated = {s for s, count in counts.items() if count >= 2}
        if not repeated or len(repeated) > self.max_basin_states:
            return set()

        repeated_visits = sum(counts[s] for s in repeated)
        # Require enough evidence to avoid treating one ordinary backtrack as a cycle.
        if repeated_visits < 5:
            return set()

        # The current state should belong to or sit directly next to the repeated
        # region. This keeps an old remote loop from becoming tabu later in the run.
        current = state.state
        if current not in repeated:
            cx, cy, _ = current
            if not any(abs(cx - x) + abs(cy - y) <= 1 for x, y, _ in repeated):
                return set()
        return repeated

    def _activate_basin(self, state: _EnvTabuState, basin: set[BehaviorState]) -> None:
        if not basin:
            return
        changed = basin != state.active_basin or state.basin_tenure <= 0
        if changed:
            state.cycle_events += 1
        state.active_basin = set(basin)
        state.basin_tenure = self.default_basin_tenure

        # Tabu the exact transitions that have recently kept the agent inside the
        # basin. This is the classic short-term move-memory component of Tabu Search.
        for source, action, target in state.transitions:
            if source in basin and target in basin:
                state.tabu_ttl[(source, action)] = max(
                    state.tabu_ttl.get((source, action), 0), self.transition_tenure
                )

    def observe(
        self,
        env_ids: Sequence[int],
        actions: Sequence[int],
        rewards: Sequence[float],
        dones: Sequence[bool],
    ) -> None:
        # Reward is intentionally ignored: positive PICKUP reward must not legitimize
        # PICKUP->DROP->PICKUP->DROP cycles.
        del rewards
        for raw_env_id, raw_action, done in zip(env_ids, actions, dones):
            env_id = int(raw_env_id)
            action = int(raw_action)
            state = self._env[env_id]
            self._decay(state)

            source = state.state
            target = self._predict(source, action)
            state.transitions.append((source, action, target))
            state.recent_actions.append(action)
            state.x, state.y, carrying = target
            state.carrying = bool(carrying)
            state.recent_states.append(target)

            basin = self._detect_local_basin(state)
            if basin:
                self._activate_basin(state, basin)

            if done:
                self.reset((env_id,))

    def rewrite_actions(
        self,
        proposed: torch.Tensor,
        logits: torch.Tensor,
        mask: torch.Tensor,
        env_ids: Sequence[int],
    ) -> torch.Tensor:
        final = proposed.clone()
        detached = logits.detach()

        for row, raw_env_id in enumerate(env_ids):
            env_id = int(raw_env_id)
            memory = self._env[env_id]
            legal = mask[row].clone()
            if not bool(legal.any()):
                continue

            current = memory.state
            recent_counts = Counter(memory.recent_states)
            scores = detached[row].clone()
            hard_tabu = torch.zeros_like(legal)

            for action in range(min(self.action_count, legal.numel())):
                if not bool(legal[action]):
                    continue
                target = self._predict(current, action)
                # Soft Tabu pressure: prefer actions leading to less-revisited states
                # even before a hard local basin has fully formed.
                scores[action] -= self.revisit_penalty * recent_counts.get(target, 0)
                transition_repeats = sum(
                    1 for source, past_action, past_target in memory.transitions
                    if source == current and past_action == action and past_target == target
                )
                scores[action] -= self.transition_penalty * transition_repeats
                scores[action] -= self.action_recency_penalty * memory.recent_actions.count(action)

                if memory.tabu_ttl.get((current, action), 0) > 0:
                    hard_tabu[action] = True
                # Crucial TabuX rule: the basin persists after escape. From an outside
                # cell, an action that would re-enter the recent problematic region is
                # still tabu until the tenure expires.
                if memory.basin_tenure > 0 and target in memory.active_basin:
                    hard_tabu[action] = True

            allowed = legal & ~hard_tabu
            if bool(allowed.any()):
                scores = scores.masked_fill(~allowed, -1e9)
                final[row] = int(scores.argmax().item())
                continue

            # Aspiration / relaxation: never deadlock the robot. If every action is
            # tabu, choose the legal action with the lowest revisit cost, using the
            # original CNN logit as a tie-breaker.
            aspiration_scores = detached[row].clone()
            for action in range(min(self.action_count, legal.numel())):
                if not bool(legal[action]):
                    continue
                target = self._predict(current, action)
                aspiration_scores[action] -= 1.25 * recent_counts.get(target, 0)
                transition_repeats = sum(
                    1 for source, past_action, past_target in memory.transitions
                    if source == current and past_action == action and past_target == target
                )
                aspiration_scores[action] -= 0.40 * transition_repeats
            aspiration_scores = aspiration_scores.masked_fill(~legal, -1e9)
            final[row] = int(aspiration_scores.argmax().item())

        return final
