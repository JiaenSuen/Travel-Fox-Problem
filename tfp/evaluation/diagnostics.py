from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence


def periodic_suffix(actions: Sequence[int], max_period: int = 3) -> tuple[int, ...] | None:
    """Return a short repeated suffix motif used only for behavior diagnostics."""
    actions = tuple(int(a) for a in actions)
    for period in range(1, max_period + 1):
        repeats = 4 if period == 1 else 2
        need = period * repeats
        if len(actions) < need:
            continue
        suffix = actions[-need:]
        motif = suffix[-period:]
        if all(suffix[i] == motif[i % period] for i in range(need)):
            return motif
    return None


@dataclass
class BehaviorDiagnostics:
    """Task-light diagnostics for identifying locally trapped embodied policies."""

    steps: int = 0
    cycle_events: int = 0
    interaction_cycle_events: int = 0
    revisited_states: int = 0

    def __post_init__(self) -> None:
        self._actions: list[int] = []
        self._active_motif: tuple[int, ...] | None = None
        self._visited: set[Hashable] = set()

    def observe(self, action: int, state_key: Hashable) -> None:
        self.steps += 1
        self._actions.append(int(action))
        motif = periodic_suffix(self._actions[-12:])
        if motif is not None and motif != self._active_motif:
            self.cycle_events += 1
            if 4 in motif or 5 in motif:
                self.interaction_cycle_events += 1
            self._active_motif = motif
        elif motif is None:
            self._active_motif = None

        if state_key in self._visited:
            self.revisited_states += 1
        self._visited.add(state_key)

    @property
    def state_revisit_rate(self) -> float:
        return float(self.revisited_states / self.steps) if self.steps else 0.0
