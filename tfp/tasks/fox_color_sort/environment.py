from __future__ import annotations

from collections import deque
from itertools import permutations
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tfp.rewards import load_reward_plugin
from tfp.rewards.reward_api import normalize_reward_name


ColorPos = Dict[int, Tuple[int, int]]


class ColorSortEnv:
    """Fox Color Sort · Local.

    Each episode samples 2–5 color-matched object/goal pairs on a low-complexity
    layout. Capacity is one object, so the policy must repeatedly navigate to an
    object, PICKUP, navigate to its same-color destination, and DROP.

    Observation channels (26 x V x V):
      0 wall; 1..5 colored objects; 6..10 colored goals; 11 agent;
      12..16 carried-color one-hot; 17..21 current-target-color one-hot;
      22/23 target row/column offset; 24 pickup available; 25 delivery available.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    TASK_ID = "TFP-FoxColorSort-Local"
    TASK_CODE = "FOX-CS-L2"
    TASK_NAME = "Fox Color Sort · Local"
    COLOR_NAMES = ("red", "green", "blue", "yellow", "purple")

    def __init__(
        self,
        map_paths: Sequence[str | Path],
        observation_mode: str = "local",
        view_size: int = 5,
        max_steps: Optional[int] = None,
        seed: int = 0,
        reward_module: str = "001_dense_color_sort",
        min_items: int = 2,
        max_items: int = 5,
    ) -> None:
        if observation_mode != "local":
            raise ValueError("TFP supports local observation only.")
        if view_size not in {5, 7}:
            raise ValueError("Color Sort supports view_size 5 or 7.")
        if not map_paths:
            raise ValueError("At least one map path is required.")
        if not (2 <= min_items <= max_items <= 5):
            raise ValueError("Color Sort item range must satisfy 2 <= min_items <= max_items <= 5.")

        self.map_paths = [Path(p) for p in map_paths]
        self.observation_mode = observation_mode
        self.view_size = int(view_size)
        self._configured_max_steps = max_steps
        self.max_steps = max_steps or 180
        self.rng = np.random.default_rng(seed)
        self.min_items = int(min_items)
        self.max_items = int(max_items)
        self.reward_module = normalize_reward_name(reward_module)
        self.reward_spec, reward_factory = load_reward_plugin(self.reward_module, task_id=self.TASK_ID)
        self.reward_function = reward_factory()

        self.action_space_n = 6
        self.observation_shape = (26, self.view_size, self.view_size)
        self.capacity = 1

        self.grid: np.ndarray
        self.start_pos: Tuple[int, int]
        self.agent_pos: Tuple[int, int]
        self.current_map: Optional[Path] = None
        self.objects: ColorPos = {}
        self.goals: ColorPos = {}
        self.initial_objects: ColorPos = {}
        self.active_colors: tuple[int, ...] = ()
        self.delivered_colors: set[int] = set()
        self.carrying_color: Optional[int] = None
        self.steps = 0
        self.collisions = 0
        self.invalid_actions = 0
        self.pickups = 0
        self.first_pickup_step: Optional[int] = None
        self.oracle_steps = 0
        self.last_reward = 0.0
        self.last_distance_delta = 0
        self.last_event = "reset"
        self._distance_cache: dict[Tuple[int, int], np.ndarray] = {}

    @property
    def carrying(self) -> bool:
        return self.carrying_color is not None

    def _load_layout(self, path: Path) -> tuple[np.ndarray, Tuple[int, int]]:
        lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Map is empty: {path}")
        width = len(lines[0])
        if any(len(line) != width for line in lines):
            raise ValueError(f"Map must be rectangular: {path}")
        grid = np.zeros((len(lines), width), dtype=np.uint8)
        start = None
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == "#":
                    grid[r, c] = 1
                elif char in ".A":
                    if char == "A":
                        if start is not None:
                            raise ValueError(f"Map has multiple starts: {path}")
                        start = (r, c)
                else:
                    raise ValueError(f"Unknown map symbol {char!r} in {path}")
        if start is None:
            raise ValueError(f"Map needs exactly one A: {path}")
        return grid, start

    def _reachable_cells(self, start: Tuple[int, int]) -> list[Tuple[int, int]]:
        queue = deque([start])
        visited = {start}
        out = []
        while queue:
            pos = queue.popleft()
            out.append(pos)
            for dr, dc in self.ACTIONS.values():
                nxt = (pos[0] + dr, pos[1] + dc)
                if self._is_free(nxt) and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return out

    def reset(self, seed: Optional[int] = None, map_path: Optional[str | Path] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_map = Path(map_path) if map_path is not None else self.map_paths[int(self.rng.integers(0, len(self.map_paths)))]
        self.grid, self.start_pos = self._load_layout(self.current_map)
        self.agent_pos = self.start_pos
        self.steps = 0
        self.collisions = 0
        self.invalid_actions = 0
        self.pickups = 0
        self.first_pickup_step = None
        self.last_reward = 0.0
        self.last_distance_delta = 0
        self.last_event = "reset"
        self.carrying_color = None
        self.delivered_colors = set()
        self._distance_cache = {}

        reachable = [p for p in self._reachable_cells(self.start_pos) if p != self.start_pos]
        size = max(self.grid.shape)
        curriculum_max = 3 if size <= 8 else 4 if size <= 10 else 5
        episode_max_items = min(self.max_items, curriculum_max)
        item_count = int(self.rng.integers(self.min_items, episode_max_items + 1))
        if len(reachable) < item_count * 2:
            raise ValueError(f"Map {self.current_map} does not have enough free cells for {item_count} pairs.")

        chosen_idx = self.rng.choice(len(reachable), size=item_count * 2, replace=False)
        chosen = [reachable[int(i)] for i in chosen_idx]
        self.active_colors = tuple(range(item_count))
        self.objects = {color: chosen[color] for color in self.active_colors}
        self.goals = {color: chosen[item_count + color] for color in self.active_colors}
        self.initial_objects = dict(self.objects)
        self.oracle_steps = self._compute_oracle_steps()
        self.max_steps = self._configured_max_steps or max(120, int(self.oracle_steps * 3.0), 8 * item_count * max(self.grid.shape))
        return self._observation(), self._info(success=False)

    def _current_target(self) -> tuple[Optional[int], Optional[Tuple[int, int]]]:
        if self.carrying_color is not None:
            color = self.carrying_color
            return color, self.goals[color]
        if not self.objects:
            return None, None
        ranked = []
        for color, pos in self.objects.items():
            ranked.append((int(self._distance_field(pos)[self.agent_pos]), int(color), pos))
        _, color, pos = min(ranked)
        return color, pos

    def _target_distance(self, pos: Tuple[int, int]) -> int:
        _, target = self._current_target()
        if target is None:
            return 0
        return int(self._distance_field(target)[pos])

    def step(self, action: int):
        if not 0 <= int(action) < self.action_space_n:
            raise ValueError(f"Invalid action {action}")
        self.steps += 1
        terminated = False
        truncated = False
        success = False
        invalid = False
        collision = False
        moved = False
        pickup = False
        item_delivered = False
        task_complete = False

        distance_before = self._target_distance(self.agent_pos)
        pickup_color_before = self._object_color_at(self.agent_pos) if self.carrying_color is None else None
        delivery_available_before = bool(
            self.carrying_color is not None and self.goals.get(self.carrying_color) == self.agent_pos
        )
        items_delivered_before = len(self.delivered_colors)

        if action in self.ACTIONS:
            dr, dc = self.ACTIONS[action]
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self._is_free(nxt):
                self.agent_pos = nxt
                moved = True
            else:
                self.collisions += 1
                self.invalid_actions += 1
                invalid = True
                collision = True
                self.last_event = "collision"
        elif action == 4:
            color = self._object_color_at(self.agent_pos)
            if self.carrying_color is None and color is not None:
                self.carrying_color = color
                self.objects.pop(color, None)
                self.pickups += 1
                if self.first_pickup_step is None:
                    self.first_pickup_step = self.steps
                pickup = True
                self.last_event = f"pickup_{self.COLOR_NAMES[color]}"
            else:
                invalid = True
                self.invalid_actions += 1
                self.last_event = "invalid_pickup"
        elif action == 5:
            if self.carrying_color is not None and self.goals[self.carrying_color] == self.agent_pos:
                color = self.carrying_color
                self.carrying_color = None
                self.delivered_colors.add(color)
                item_delivered = True
                self.last_event = f"deliver_{self.COLOR_NAMES[color]}"
                if len(self.delivered_colors) == len(self.active_colors):
                    task_complete = True
                    success = True
                    terminated = True
                    self.last_event = "complete"
            else:
                invalid = True
                self.invalid_actions += 1
                self.last_event = "invalid_drop"

        distance_after = distance_before if (pickup or item_delivered or terminated) else self._target_distance(self.agent_pos)
        distance_delta = int(distance_before - distance_after)
        self.last_distance_delta = distance_delta
        if moved:
            self.last_event = "progress" if distance_delta > 0 else "move"

        transition = {
            "action": int(action),
            "moved": moved,
            "invalid": invalid,
            "collision": collision,
            "pickup": pickup,
            "item_delivered": item_delivered,
            "delivered": item_delivered,
            "task_complete": task_complete,
            "distance_before": int(distance_before),
            "distance_after": int(distance_after),
            "distance_delta": int(distance_delta),
            "carrying": self.carrying,
            "pickup_available_before": pickup_color_before is not None,
            "delivery_available_before": delivery_available_before,
            "missed_pickup": bool(pickup_color_before is not None and int(action) != 4),
            "missed_delivery": bool(delivery_available_before and int(action) != 5),
            "items_total": len(self.active_colors),
            "items_delivered_before": items_delivered_before,
            "items_delivered": len(self.delivered_colors),
            "step": int(self.steps),
        }
        reward = float(self.reward_function.compute(transition))
        self.last_reward = reward

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            self.last_event = "timeout"
        return self._observation(), reward, terminated, truncated, self._info(success=success)

    def valid_action_mask(self, mode: str = "task") -> np.ndarray:
        if mode not in {"task", "valid"}:
            raise ValueError("mask mode must be 'task' or 'valid'.")
        mask = np.zeros(self.action_space_n, dtype=bool)
        pickup_available = self.carrying_color is None and self._object_color_at(self.agent_pos) is not None
        delivery_available = self.carrying_color is not None and self.goals[self.carrying_color] == self.agent_pos
        if mode == "task" and pickup_available:
            mask[4] = True
            return mask
        if mode == "task" and delivery_available:
            mask[5] = True
            return mask
        for action, (dr, dc) in self.ACTIONS.items():
            if self._is_free((self.agent_pos[0] + dr, self.agent_pos[1] + dc)):
                mask[action] = True
        if pickup_available:
            mask[4] = True
        if delivery_available:
            mask[5] = True
        return mask

    def _observation(self) -> np.ndarray:
        v = self.view_size
        radius = v // 2
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        ar, ac = self.agent_pos
        for vr in range(v):
            for vc in range(v):
                gr, gc = ar + vr - radius, ac + vc - radius
                if not (0 <= gr < self.grid.shape[0] and 0 <= gc < self.grid.shape[1]):
                    obs[0, vr, vc] = 1.0
                    continue
                if self.grid[gr, gc] == 1:
                    obs[0, vr, vc] = 1.0
                for color, pos in self.objects.items():
                    if pos == (gr, gc):
                        obs[1 + color, vr, vc] = 1.0
                for color, pos in self.goals.items():
                    if pos == (gr, gc):
                        obs[6 + color, vr, vc] = 1.0
        obs[11, radius, radius] = 1.0
        if self.carrying_color is not None:
            obs[12 + self.carrying_color, :, :] = 1.0
        target_color, target = self._current_target()
        if target_color is not None:
            obs[17 + target_color, :, :] = 1.0
        if target is not None:
            scale = float(max(1, self.view_size // 2))
            obs[22, :, :] = np.clip((target[0] - ar) / scale, -1.0, 1.0)
            obs[23, :, :] = np.clip((target[1] - ac) / scale, -1.0, 1.0)
        obs[24, :, :] = 1.0 if (self.carrying_color is None and self._object_color_at(self.agent_pos) is not None) else 0.0
        obs[25, :, :] = 1.0 if (self.carrying_color is not None and self.goals[self.carrying_color] == self.agent_pos) else 0.0
        return obs

    def _object_color_at(self, pos: Tuple[int, int]) -> Optional[int]:
        for color, object_pos in self.objects.items():
            if object_pos == pos:
                return color
        return None

    def _is_free(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and self.grid[r, c] == 0

    def _distance_field(self, goal: Tuple[int, int]) -> np.ndarray:
        if goal in self._distance_cache:
            return self._distance_cache[goal]
        field = np.full(self.grid.shape, self.grid.size, dtype=np.int32)
        field[goal] = 0
        queue = deque([goal])
        while queue:
            r, c = queue.popleft()
            nxt_d = int(field[r, c]) + 1
            for dr, dc in self.ACTIONS.values():
                nr, nc = r + dr, c + dc
                if self._is_free((nr, nc)) and nxt_d < field[nr, nc]:
                    field[nr, nc] = nxt_d
                    queue.append((nr, nc))
        self._distance_cache[goal] = field
        return field

    def _compute_oracle_steps(self) -> int:
        best = None
        for order in permutations(self.active_colors):
            pos = self.start_pos
            total = 0
            for color in order:
                obj = self.initial_objects[color]
                goal = self.goals[color]
                total += int(self._distance_field(obj)[pos]) + 1
                total += int(self._distance_field(goal)[obj]) + 1
                pos = goal
            best = total if best is None else min(best, total)
        return int(best or 0)

    def _info(self, success: bool) -> dict[str, object]:
        delivered = len(self.delivered_colors)
        total = len(self.active_colors)
        completion = float(delivered / total) if total else 0.0
        efficiency = float(self.oracle_steps / self.steps) if success and self.steps > 0 else 0.0
        return {
            "task_id": self.TASK_ID,
            "task_code": self.TASK_CODE,
            "success": bool(success),
            "carrying": self.carrying,
            "carrying_color": self.carrying_color,
            "capacity": self.capacity,
            "steps": self.steps,
            "map": self.current_map.name if self.current_map else None,
            "map_shape": tuple(int(v) for v in self.grid.shape),
            "agent_pos": self.agent_pos,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "reward_module": self.reward_module,
            "collisions": self.collisions,
            "invalid_actions": self.invalid_actions,
            "pickup_step": self.first_pickup_step,
            "pickups": self.pickups,
            "items_total": total,
            "items_delivered": delivered,
            "completion_rate": completion,
            "oracle_steps": self.oracle_steps,
            "path_efficiency": efficiency,
            "last_reward": self.last_reward,
            "last_distance_delta": self.last_distance_delta,
            "behavior_state": (
                self.agent_pos,
                self.carrying_color,
                tuple(sorted(self.objects.items())),
                tuple(sorted(self.delivered_colors)),
            ),
        }

    def robot_status(self) -> dict[str, object]:
        carry = "Empty" if self.carrying_color is None else self.COLOR_NAMES[self.carrying_color]
        target_color, _ = self._current_target()
        target = "Complete" if target_color is None else self.COLOR_NAMES[target_color]
        return {
            "Robot": "Fox-02",
            "Position": f"{self.agent_pos[0]}, {self.agent_pos[1]}",
            "Carried items": 1 if self.carrying else 0,
            "Capacity": 1,
            "Cargo color": carry,
            "Current target": target,
            "Delivered": f"{len(self.delivered_colors)}/{len(self.active_colors)}",
            "Last event": self.last_event,
        }

    def render_entities(self) -> dict[str, list[tuple[int, Tuple[int, int]]]]:
        return {
            "objects": sorted(self.objects.items()),
            "goals": sorted(self.goals.items()),
        }

    def render_ascii(self) -> str:
        chars = np.full(self.grid.shape, ".", dtype="<U1")
        chars[self.grid == 1] = "#"
        for color, pos in self.goals.items():
            chars[pos] = self.COLOR_NAMES[color][0].upper()
        for color, pos in self.objects.items():
            chars[pos] = self.COLOR_NAMES[color][0]
        chars[self.agent_pos] = "f" if self.carrying else "F"
        return "\n".join("".join(row) for row in chars)
