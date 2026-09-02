from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, object]


def load_ascii_map(path: str | Path) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """Load a rectangular ASCII map.

    Symbols:
        #: wall
        .: free floor
        A: agent start
        G: delivery goal

    The task object is sampled from reachable free cells at reset time.
    """
    path = Path(path)
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Map is empty: {path}")

    width = len(lines[0])
    if any(len(line) != width for line in lines):
        raise ValueError(f"Map must be rectangular: {path}")

    grid = np.zeros((len(lines), width), dtype=np.uint8)
    start = None
    goal = None

    for r, line in enumerate(lines):
        for c, char in enumerate(line):
            if char == "#":
                grid[r, c] = 1
            elif char in ".AG":
                grid[r, c] = 0
                if char == "A":
                    if start is not None:
                        raise ValueError(f"Map has multiple agent starts: {path}")
                    start = (r, c)
                elif char == "G":
                    if goal is not None:
                        raise ValueError(f"Map has multiple goals: {path}")
                    goal = (r, c)
            else:
                raise ValueError(f"Unknown map symbol {char!r} in {path}")

    if start is None or goal is None:
        raise ValueError(f"Map needs exactly one A and one G: {path}")

    return grid, start, goal


def reachable_cells(grid: np.ndarray, start: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return all floor cells reachable from start with four-neighbor motion."""
    h, w = grid.shape
    queue = [start]
    visited = {start}
    output: List[Tuple[int, int]] = []

    while queue:
        r, c = queue.pop(0)
        output.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return output
