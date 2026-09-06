from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tfp.rewards import load_reward_plugin
from tfp.rewards.reward_api import normalize_reward_name
from .base import load_ascii_map, reachable_cells


class TransportEnv:
    """Fox Transport · Local environment.

    The default observation is a 5x5 local crop. The full map remains hidden from
    the learning model, while low-bandwidth target-direction cues keep this first
    TFP benchmark solvable by a feed-forward CNN.

    Action space:
        0: move up
        1: move down
        2: move left
        3: move right
        4: pick up
        5: drop
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    TASK_ID = "TFP-FoxTransport-Local"
    TASK_CODE = "FOX-TR-L1"
    TASK_NAME = "Fox Transport · Local"

    def __init__(
        self,
        map_paths: Sequence[str | Path],
        observation_mode: str = "local",
        view_size: int = 5,
        max_steps: Optional[int] = None,
        seed: int = 0,
        reward_module: str = "001_dense_transport",
    ) -> None:
        if observation_mode != "local":
            raise ValueError("TFP supports local observation only.")
        if view_size % 2 == 0 or view_size < 3:
            raise ValueError("view_size must be an odd integer >= 3.")
        if not map_paths:
            raise ValueError("At least one map path is required.")

        self.map_paths = [Path(p) for p in map_paths]
        self.observation_mode = observation_mode
        self.view_size = view_size
        self._configured_max_steps = max_steps
        self.max_steps = max_steps or 120
        self.rng = np.random.default_rng(seed)
        self.reward_module = normalize_reward_name(reward_module)
        self.reward_spec, reward_factory = load_reward_plugin(self.reward_module)
        self.reward_function = reward_factory()

        self.action_space_n = 6
        self.observation_shape = (10, view_size, view_size)

        self.grid: np.ndarray
        self.agent_pos: Tuple[int, int]
        self.start_pos: Tuple[int, int]
        self.goal_pos: Tuple[int, int]
        self.object_pos: Optional[Tuple[int, int]]
        self.initial_object_pos: Optional[Tuple[int, int]] = None
        self.carrying = False
        self.capacity = 1
        self.steps = 0
        self.current_map: Optional[Path] = None
        self.collisions = 0
        self.invalid_actions = 0
        self.pickup_step: Optional[int] = None
        self.oracle_steps = 0
        self.last_reward = 0.0
        self.last_distance_delta = 0
        self.last_event = "reset"
        self._goal_distance_field: Optional[np.ndarray] = None
        self._object_distance_field: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None, map_path: Optional[str | Path] = None) -> Tuple[np.ndarray, Dict[str, object]]:
        """Reset an episode with deterministic map-by-seed behavior when requested."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if map_path is None:
            self.current_map = self.map_paths[int(self.rng.integers(0, len(self.map_paths)))]
        else:
            self.current_map = Path(map_path)

        self.grid, self.start_pos, self.goal_pos = load_ascii_map(self.current_map)
        self.agent_pos = self.start_pos
        self.carrying = False
        self.steps = 0
        self.collisions = 0
        self.invalid_actions = 0
        self.pickup_step = None
        self.last_reward = 0.0
        self.last_distance_delta = 0
        self.last_event = "reset"
        self.max_steps = self._configured_max_steps or max(80, 6 * (self.grid.shape[0] + self.grid.shape[1]))

        reachable = reachable_cells(self.grid, self.start_pos)
        start_distance_field = self._distance_field(self.start_pos)
        self._goal_distance_field = self._distance_field(self.goal_pos)
        size_scale = max(self.grid.shape)
        min_start_distance = 2 if size_scale <= 10 else 4 if size_scale <= 15 else 6
        min_goal_distance = 2 if size_scale <= 10 else 3 if size_scale <= 15 else 4
        candidates = [
            p
            for p in reachable
            if p not in {self.start_pos, self.goal_pos}
            and start_distance_field[p] >= min_start_distance
            and self._goal_distance_field[p] >= min_goal_distance
        ]
        if not candidates:
            candidates = [p for p in reachable if p not in {self.start_pos, self.goal_pos}]
        if not candidates:
            raise ValueError(f"No valid object spawn cells in map {self.current_map}")

        self.object_pos = candidates[int(self.rng.integers(0, len(candidates)))]
        self.initial_object_pos = self.object_pos
        self._object_distance_field = self._distance_field(self.object_pos)
        self.oracle_steps = int(start_distance_field[self.object_pos] + 1 + self._goal_distance_field[self.object_pos] + 1)
        return self._observation(), self._info(success=False)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, object]]:
        if not 0 <= int(action) < self.action_space_n:
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        terminated = False
        truncated = False
        success = False
        invalid = False
        collision = False
        pickup = False
        delivered = False
        moved = False
        distance_before = self._target_distance(self.agent_pos)
        pickup_available_before = bool((not self.carrying) and self.object_pos == self.agent_pos)
        delivery_available_before = bool(self.carrying and self.agent_pos == self.goal_pos)

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
            if not self.carrying and self.object_pos == self.agent_pos:
                self.carrying = True
                self.object_pos = None
                self.pickup_step = self.steps
                pickup = True
                self.last_event = "pickup"
            else:
                self.invalid_actions += 1
                invalid = True
                self.last_event = "invalid_pickup"
        elif action == 5:
            if self.carrying and self.agent_pos == self.goal_pos:
                self.carrying = False
                delivered = True
                terminated = True
                success = True
                self.last_event = "delivery"
            elif self.carrying:
                self.carrying = False
                self.object_pos = self.agent_pos
                self._object_distance_field = self._distance_field(self.object_pos)
                self.invalid_actions += 1
                invalid = True
                self.last_event = "early_drop"
            else:
                self.invalid_actions += 1
                invalid = True
                self.last_event = "invalid_drop"

        distance_after = self._target_distance(self.agent_pos) if not terminated else distance_before
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
            "delivered": delivered,
            "distance_before": int(distance_before),
            "distance_after": int(distance_after),
            "distance_delta": distance_delta,
            "carrying": bool(self.carrying),
            "pickup_available_before": pickup_available_before,
            "delivery_available_before": delivery_available_before,
            "missed_pickup": bool(pickup_available_before and int(action) != 4),
            "missed_delivery": bool(delivery_available_before and int(action) != 5),
            "step": int(self.steps),
        }
        reward = float(self.reward_function.compute(transition))
        self.last_reward = reward

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            self.last_event = "timeout"

        return self._observation(), reward, terminated, truncated, self._info(success)

    def robot_status(self) -> dict[str, object]:
        """Task-facing status panel contract used by the visualizer.

        Future tasks can expose different fields without changing the renderer API.
        """
        return {
            "Robot": "Fox-01",
            "Position": f"{self.agent_pos[0]}, {self.agent_pos[1]}",
            "Carried items": 1 if self.carrying else 0,
            "Capacity": self.capacity,
            "Cargo": "Transport object" if self.carrying else "Empty",
            "Current target": "Delivery goal" if self.carrying else "Cargo object",
            "Last event": self.last_event,
            "Last reward": f"{self.last_reward:+.3f}",
            "Remaining steps": max(0, self.max_steps - self.steps),
        }

    def _observation(self) -> np.ndarray:
        return self._local_observation()

    def _write_common_channels(self, obs: np.ndarray) -> None:
        ar, ac = self.agent_pos
        target = self.goal_pos if self.carrying else self.object_pos
        obs[4, :, :] = 1.0 if self.carrying else 0.0
        if target is not None:
            row_scale = col_scale = float(max(1, self.view_size // 2))
            obs[5, :, :] = np.clip((target[0] - ar) / row_scale, -1.0, 1.0)
            obs[6, :, :] = np.clip((target[1] - ac) / col_scale, -1.0, 1.0)
        obs[7, :, :] = 1.0 if (not self.carrying and self.object_pos == self.agent_pos) else 0.0
        obs[8, :, :] = 1.0 if (self.carrying and self.agent_pos == self.goal_pos) else 0.0

    def _local_observation(self) -> np.ndarray:
        v = self.view_size
        radius = v // 2
        obs = np.zeros((10, v, v), dtype=np.float32)
        ar, ac = self.agent_pos
        for vr in range(v):
            for vc in range(v):
                gr = ar + (vr - radius)
                gc = ac + (vc - radius)
                if not (0 <= gr < self.grid.shape[0] and 0 <= gc < self.grid.shape[1]):
                    obs[9, vr, vc] = 1.0
                    continue
                if self.grid[gr, gc] == 1:
                    obs[0, vr, vc] = 1.0
                if self.object_pos == (gr, gc):
                    obs[1, vr, vc] = 1.0
                if self.goal_pos == (gr, gc):
                    obs[2, vr, vc] = 1.0
        obs[3, radius, radius] = 1.0
        self._write_common_channels(obs)
        return obs

    def valid_action_mask(self, mode: str = "task") -> np.ndarray:
        """Return the action mask used by a model-facing controller."""
        if mode not in {"task", "valid"}:
            raise ValueError("mask mode must be 'task' or 'valid'; unmasked action selection is not supported.")

        mask = np.zeros(self.action_space_n, dtype=bool)
        if mode == "task" and (not self.carrying and self.object_pos == self.agent_pos):
            mask[4] = True
            return mask
        if mode == "task" and (self.carrying and self.agent_pos == self.goal_pos):
            mask[5] = True
            return mask

        for action, (dr, dc) in self.ACTIONS.items():
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self._is_free(nxt):
                mask[action] = True
        if not self.carrying and self.object_pos == self.agent_pos:
            mask[4] = True
        # "valid" means semantically valid task actions, not merely executable
        # actions.  Early DROP outside the goal is classified as invalid by step(),
        # so exposing it here creates a pickup/drop reward exploit.
        if self.carrying and self.agent_pos == self.goal_pos:
            mask[5] = True
        return mask

    def render_ascii(self) -> str:
        chars = np.full(self.grid.shape, ".", dtype="<U1")
        chars[self.grid == 1] = "#"
        chars[self.goal_pos] = "G"
        if self.object_pos is not None:
            chars[self.object_pos] = "O"
        chars[self.agent_pos] = "f" if self.carrying else "F"
        return "\n".join("".join(row) for row in chars)

    def _is_free(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and self.grid[r, c] == 0

    def _distance_field(self, goal: Tuple[int, int]) -> np.ndarray:
        field = np.full(self.grid.shape, self.grid.size, dtype=np.int32)
        field[goal] = 0
        queue = deque([goal])
        while queue:
            r, c = queue.popleft()
            next_distance = int(field[r, c]) + 1
            for dr, dc in self.ACTIONS.values():
                nr, nc = r + dr, c + dc
                if self._is_free((nr, nc)) and next_distance < field[nr, nc]:
                    field[nr, nc] = next_distance
                    queue.append((nr, nc))
        return field

    def _target_distance(self, pos: Tuple[int, int]) -> int:
        field = self._goal_distance_field if self.carrying else self._object_distance_field
        if field is None:
            return 0
        return int(field[pos])

    def _info(self, success: bool) -> Dict[str, object]:
        efficiency = (self.oracle_steps / self.steps) if success and self.steps > 0 else 0.0
        return {
            "task_id": self.TASK_ID,
            "task_code": self.TASK_CODE,
            "success": success,
            "carrying": self.carrying,
            "capacity": self.capacity,
            "steps": self.steps,
            "map": self.current_map.name if self.current_map else None,
            "map_shape": tuple(int(v) for v in self.grid.shape),
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "object_pos": self.object_pos,
            "initial_object_pos": self.initial_object_pos,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "reward_module": self.reward_module,
            "collisions": self.collisions,
            "invalid_actions": self.invalid_actions,
            "pickup_step": self.pickup_step,
            "oracle_steps": self.oracle_steps,
            "path_efficiency": float(efficiency),
            "last_reward": self.last_reward,
            "last_distance_delta": self.last_distance_delta,
        }
