from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tfp.rewards import load_reward_plugin
from tfp.rewards.reward_api import normalize_reward_name


Pos = Tuple[int, int]


class RoomTransportEnv:
    """Fox Room Transport · Doors.

    A long-horizon local-POMDP transport task over multi-room layouts. One object and
    one destination are sampled from different rooms at reset time. Door cells connect
    rooms and have episode-seeded open/closed state. Closed doors block motion and are
    operated with a seventh action, TOGGLE_DOOR.

    Observation channels (14 x V x V):
      0 wall/outside; 1 closed door; 2 open door; 3 object; 4 goal; 5 agent;
      6 carrying; 7/8 next-doorway route row/column offset; 9 pickup available;
      10 delivery available; 11 adjacent closed door; 12 adjacent open door;
      13 open-door fraction.

    V5 note: channels 7/8 no longer point straight through walls to the final target.
    They point to the next required doorway on an executable shortest path,
    falling back to the active object/goal when no doorway remains. This remains a
    low-bandwidth 2-D navigation cue rather than exposing the global map.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP", "TOGGLE_DOOR")
    TASK_ID = "TFP-FoxRoomTransport-Local"
    TASK_CODE = "FOX-RM-L3"
    TASK_NAME = "Fox Room Transport · Doors"

    def __init__(
        self,
        map_paths: Sequence[str | Path],
        observation_mode: str = "local",
        view_size: int = 7,
        max_steps: Optional[int] = None,
        seed: int = 0,
        reward_module: str = "001_actionable_geodesic",
    ) -> None:
        if observation_mode != "local":
            raise ValueError("TFP supports local observation only.")
        if view_size not in {5, 7}:
            raise ValueError("Fox Room Transport supports view_size 5 or 7.")
        if not map_paths:
            raise ValueError("At least one map path is required.")

        self.map_paths = [Path(p) for p in map_paths]
        self.observation_mode = observation_mode
        self.view_size = int(view_size)
        self._configured_max_steps = max_steps
        self.max_steps = max_steps or 300
        self.rng = np.random.default_rng(seed)
        self.reward_module = normalize_reward_name(reward_module)
        self.reward_spec, reward_factory = load_reward_plugin(self.reward_module, task_id=self.TASK_ID)
        self.reward_function = reward_factory()

        self.action_space_n = 7
        self.observation_shape = (14, self.view_size, self.view_size)
        self.capacity = 1

        self.grid: np.ndarray
        self.start_pos: Pos
        self.agent_pos: Pos
        self.object_pos: Optional[Pos] = None
        self.initial_object_pos: Optional[Pos] = None
        self.goal_pos: Pos
        self.current_map: Optional[Path] = None
        self.door_positions: set[Pos] = set()
        self.open_doors: set[Pos] = set()
        self.room_cells: list[list[Pos]] = []
        self.room_index: dict[Pos, int] = {}
        self.room_graph: dict[int, set[int]] = {}
        self.object_room = -1
        self.goal_room = -1
        self.start_room = -1
        self.carrying = False
        self.steps = 0
        self.collisions = 0
        self.invalid_actions = 0
        self.pickup_step: Optional[int] = None
        self.door_opens = 0
        self.door_closes = 0
        self.oracle_steps = 0
        self.last_reward = 0.0
        self.last_distance_delta = 0
        self.last_event = "reset"
        self._object_distance_field: Optional[np.ndarray] = None
        self._goal_distance_field: Optional[np.ndarray] = None
        self._action_object_distance_field: Optional[np.ndarray] = None
        self._action_goal_distance_field: Optional[np.ndarray] = None
        self._state_visits: dict[tuple[object, ...], int] = {}
        self._visited_rooms: set[int] = set()

    def _load_layout(self, path: Path) -> tuple[np.ndarray, Pos, set[Pos]]:
        lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Map is empty: {path}")
        width = len(lines[0])
        if any(len(line) != width for line in lines):
            raise ValueError(f"Map must be rectangular: {path}")
        grid = np.zeros((len(lines), width), dtype=np.uint8)
        start = None
        doors: set[Pos] = set()
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == "#":
                    grid[r, c] = 1
                elif char in ".A":
                    if char == "A":
                        if start is not None:
                            raise ValueError(f"Map has multiple starts: {path}")
                        start = (r, c)
                elif char == "D":
                    doors.add((r, c))
                else:
                    raise ValueError(f"Unknown map symbol {char!r} in {path}")
        if start is None:
            raise ValueError(f"Map needs exactly one A: {path}")
        if not doors:
            raise ValueError(f"Room map needs at least one D door: {path}")
        return grid, start, doors

    def _build_rooms(self) -> None:
        """Connected components when doors are treated as separators."""
        self.room_cells = []
        self.room_index = {}
        h, w = self.grid.shape
        visited: set[Pos] = set()
        for r in range(h):
            for c in range(w):
                start = (r, c)
                if start in visited or start in self.door_positions or self.grid[start] == 1:
                    continue
                queue = deque([start])
                visited.add(start)
                cells: list[Pos] = []
                room_id = len(self.room_cells)
                while queue:
                    pos = queue.popleft()
                    cells.append(pos)
                    self.room_index[pos] = room_id
                    for dr, dc in self.ACTIONS.values():
                        nxt = (pos[0] + dr, pos[1] + dc)
                        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                            continue
                        if nxt in visited or nxt in self.door_positions or self.grid[nxt] == 1:
                            continue
                        visited.add(nxt)
                        queue.append(nxt)
                self.room_cells.append(cells)

        self.room_graph = {i: set() for i in range(len(self.room_cells))}
        for door in self.door_positions:
            adjacent_rooms = set()
            for dr, dc in self.ACTIONS.values():
                nxt = (door[0] + dr, door[1] + dc)
                if nxt in self.room_index:
                    adjacent_rooms.add(self.room_index[nxt])
            for a in adjacent_rooms:
                for b in adjacent_rooms:
                    if a != b:
                        self.room_graph[a].add(b)

    def _room_distances(self, source: int) -> dict[int, int]:
        dist = {source: 0}
        queue = deque([source])
        while queue:
            room = queue.popleft()
            for nxt in self.room_graph.get(room, ()):
                if nxt not in dist:
                    dist[nxt] = dist[room] + 1
                    queue.append(nxt)
        return dist

    def _deep_room_cells(self, room_id: int) -> list[Pos]:
        cells = self.room_cells[room_id]
        door_adjacent = set()
        for door in self.door_positions:
            for dr, dc in self.ACTIONS.values():
                door_adjacent.add((door[0] + dr, door[1] + dc))
        deep = [p for p in cells if p not in door_adjacent and p != self.start_pos]
        return deep or [p for p in cells if p != self.start_pos] or list(cells)

    def _sample_object_goal(self) -> tuple[Pos, Pos, int, int]:
        self.start_room = self.room_index[self.start_pos]
        from_start = self._room_distances(self.start_room)
        if len(from_start) < 2:
            raise ValueError(f"Room graph is disconnected in {self.current_map}")

        max_start = max(from_start.values())
        object_rooms = [r for r, d in from_start.items() if r != self.start_room and d >= max(1, max_start - 1)]
        object_room = int(self.rng.choice(object_rooms))

        from_object = self._room_distances(object_room)
        max_obj = max(from_object.values())
        goal_rooms = [r for r, d in from_object.items() if r != object_room and d >= max(1, max_obj - 1)]
        if len(goal_rooms) > 1 and self.start_room in goal_rooms:
            # Usually avoid collapsing the second leg back to the spawn room.
            goal_rooms = [r for r in goal_rooms if r != self.start_room]
        goal_room = int(self.rng.choice(goal_rooms))

        object_candidates = self._deep_room_cells(object_room)
        goal_candidates = self._deep_room_cells(goal_room)
        object_pos = object_candidates[int(self.rng.integers(0, len(object_candidates)))]
        goal_pos = goal_candidates[int(self.rng.integers(0, len(goal_candidates)))]
        return object_pos, goal_pos, object_room, goal_room

    def reset(self, seed: Optional[int] = None, map_path: Optional[str | Path] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_map = Path(map_path) if map_path is not None else self.map_paths[int(self.rng.integers(0, len(self.map_paths)))]
        self.grid, self.start_pos, self.door_positions = self._load_layout(self.current_map)
        self._build_rooms()
        self.agent_pos = self.start_pos
        self.carrying = False
        self.steps = 0
        self.collisions = 0
        self.invalid_actions = 0
        self.pickup_step = None
        self.door_opens = 0
        self.door_closes = 0
        self.last_reward = 0.0
        self.last_distance_delta = 0
        self.last_event = "reset"
        self.max_steps = self._configured_max_steps or max(220, 14 * (self.grid.shape[0] + self.grid.shape[1]))

        # Seeded door states: most doors start closed so door interaction is a genuine
        # part of the task, while some open doors prevent every episode being identical.
        self.open_doors = {door for door in self.door_positions if float(self.rng.random()) < 0.35}
        if len(self.open_doors) == len(self.door_positions):
            self.open_doors.remove(sorted(self.open_doors)[0])

        self.object_pos, self.goal_pos, self.object_room, self.goal_room = self._sample_object_goal()
        self.initial_object_pos = self.object_pos
        self._object_distance_field = self._distance_field(self.object_pos, structural=True)
        self._goal_distance_field = self._distance_field(self.goal_pos, structural=True)
        self._refresh_action_distance_fields()
        self.oracle_steps = self._estimate_oracle_steps()
        self._state_visits = {self._behavior_state(): 1}
        self._visited_rooms = {self.start_room}
        return self._observation(), self._info(success=False)

    def _adjacent_door(self, prefer_closed: bool = True) -> Optional[Pos]:
        ordered = []
        for action in range(4):
            dr, dc = self.ACTIONS[action]
            pos = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if pos in self.door_positions:
                ordered.append(pos)
        if not ordered:
            return None
        if prefer_closed:
            closed = [p for p in ordered if p not in self.open_doors]
            if closed:
                return closed[0]
        return ordered[0]

    def _action_distance_field(self, goal: Pos) -> np.ndarray:
        """Exact shortest action cost to ``goal`` under the current door state.

        Entering an open/floor cell costs one MOVE. Entering a closed door costs two
        actions (TOGGLE + MOVE). Computing the field from the goal lets every normal
        step query O(1); it is refreshed only when a door state or dropped-object
        position changes.
        """
        inf = self.grid.size * 20
        field = np.full(self.grid.shape, inf, dtype=np.int32)
        field[goal] = 0
        heap: list[tuple[int, Pos]] = [(0, goal)]
        while heap:
            cost, pos = heappop(heap)
            if cost != int(field[pos]):
                continue
            # Reverse edge predecessor -> pos. The forward cost depends on whether
            # the destination cell ``pos`` is a currently closed door.
            enter_cost = 1 + int(pos in self.door_positions and pos not in self.open_doors)
            for dr, dc in self.ACTIONS.values():
                prev = (pos[0] + dr, pos[1] + dc)
                if not self._is_floor_or_door(prev):
                    continue
                new = cost + enter_cost
                if new < int(field[prev]):
                    field[prev] = new
                    heappush(heap, (new, prev))
        return field

    def _refresh_action_distance_fields(self) -> None:
        self._action_object_distance_field = (
            self._action_distance_field(self.object_pos) if self.object_pos is not None else None
        )
        self._action_goal_distance_field = self._action_distance_field(self.goal_pos)

    def _actionable_target_distance(self, pos: Pos, carrying: Optional[bool] = None) -> int:
        use_goal = self.carrying if carrying is None else bool(carrying)
        field = self._action_goal_distance_field if use_goal else self._action_object_distance_field
        if field is None:
            return 0
        return int(field[pos])

    def _room_for_agent(self) -> int:
        if self.agent_pos in self.room_index:
            return self.room_index[self.agent_pos]
        # Door cells sit between rooms. Use an adjacent room, preferring one already
        # reached in this episode so the waypoint remains stable while crossing.
        adjacent = []
        for dr, dc in self.ACTIONS.values():
            rid = self.room_index.get((self.agent_pos[0] + dr, self.agent_pos[1] + dc))
            if rid is not None:
                adjacent.append(rid)
        if not adjacent:
            return self.start_room
        for rid in adjacent:
            if rid in self._visited_rooms:
                return rid
        return adjacent[0]

    def _route_waypoint(self) -> Pos:
        """Return the next required doorway on an actionable shortest path.

        V5 deliberately exposes a *high-level* route cue rather than the next movement
        action. The environment traces its executable shortest path internally only to
        identify the first doorway; the policy receives just the 2-D offset to that door
        (or to the active object/goal if no door remains). Local obstacle avoidance and
        door interaction still have to be learned from the crop and reward.
        """
        target = self.goal_pos if self.carrying else self.object_pos
        if target is None:
            return self.goal_pos
        field = self._action_goal_distance_field if self.carrying else self._action_object_distance_field
        if field is None:
            return target

        current = self.agent_pos
        visited = {current}
        max_trace = self.grid.size
        for _ in range(max_trace):
            if current == target:
                return target
            current_cost = int(field[current])
            candidates: list[tuple[int, int, Pos]] = []
            for action, (dr, dc) in self.ACTIONS.items():
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in visited or not self._is_floor_or_door(nxt):
                    continue
                enter_cost = 1 + int(nxt in self.door_positions and nxt not in self.open_doors)
                if current_cost == int(field[nxt]) + enter_cost:
                    tie = abs(target[0] - nxt[0]) + abs(target[1] - nxt[1])
                    candidates.append((tie, action, nxt))
            if not candidates:
                return target
            _, _, nxt = min(candidates)
            if nxt in self.door_positions and nxt != self.agent_pos:
                return nxt
            visited.add(nxt)
            current = nxt
        return target

    def step(self, action: int):
        if not 0 <= int(action) < self.action_space_n:
            raise ValueError(f"Invalid action {action}")
        action = int(action)
        self.steps += 1
        terminated = False
        truncated = False
        success = False
        invalid = False
        collision = False
        pickup = False
        delivered = False
        moved = False
        door_opened = False
        door_closed = False
        phase_before = bool(self.carrying)
        distance_before = self._target_distance(self.agent_pos)
        actionable_distance_before = self._actionable_target_distance(self.agent_pos, carrying=phase_before)
        room_before = self._room_for_agent()
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
                collision = invalid = True
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
                success = True
                terminated = True
                self.last_event = "delivery"
            elif self.carrying:
                self.carrying = False
                self.object_pos = self.agent_pos
                self._object_distance_field = self._distance_field(self.object_pos, structural=True)
                self._refresh_action_distance_fields()
                self.invalid_actions += 1
                invalid = True
                self.last_event = "early_drop"
            else:
                self.invalid_actions += 1
                invalid = True
                self.last_event = "invalid_drop"
        elif action == 6:
            door = self._adjacent_door(prefer_closed=True)
            if door is None:
                self.invalid_actions += 1
                invalid = True
                self.last_event = "invalid_toggle"
            elif door in self.open_doors:
                self.open_doors.remove(door)
                self._refresh_action_distance_fields()
                self.door_closes += 1
                door_closed = True
                self.last_event = "door_close"
            else:
                self.open_doors.add(door)
                self._refresh_action_distance_fields()
                self.door_opens += 1
                door_opened = True
                self.last_event = "door_open"

        distance_after = self._target_distance(self.agent_pos) if not terminated else distance_before
        distance_delta = int(distance_before - distance_after)
        self.last_distance_delta = distance_delta

        # Potential shaping must compare the same task phase. Pickup/delivery switch
        # the active target and are rewarded as milestones instead of a bogus distance
        # jump. Early drop is explicitly invalid and also gets no potential credit.
        if pickup or delivered or self.last_event == "early_drop":
            actionable_distance_after = actionable_distance_before
            actionable_delta = 0
        else:
            actionable_distance_after = self._actionable_target_distance(self.agent_pos, carrying=phase_before)
            actionable_delta = int(actionable_distance_before - actionable_distance_after)

        room_after = self._room_for_agent()
        entered_new_room = bool(moved and room_after != room_before and room_after not in self._visited_rooms)
        if moved and room_after in self.room_graph:
            self._visited_rooms.add(room_after)
        if moved:
            self.last_event = "progress" if actionable_delta > 0 else "move"

        state = self._behavior_state()
        visit_count = self._state_visits.get(state, 0) + 1
        self._state_visits[state] = visit_count
        repeat_visit = visit_count >= 3

        transition = {
            "action": action,
            "moved": moved,
            "invalid": invalid,
            "collision": collision,
            "pickup": pickup,
            "delivered": delivered,
            "task_complete": success,
            "door_opened": door_opened,
            "door_closed": door_closed,
            "distance_before": int(distance_before),
            "distance_after": int(distance_after),
            "distance_delta": distance_delta,
            "actionable_distance_before": int(actionable_distance_before),
            "actionable_distance_after": int(actionable_distance_after),
            "actionable_delta": int(actionable_delta),
            "entered_new_room": entered_new_room,
            "room_before": int(room_before),
            "room_after": int(room_after),
            "carrying": bool(self.carrying),
            "pickup_available_before": pickup_available_before,
            "delivery_available_before": delivery_available_before,
            "missed_pickup": bool(pickup_available_before and action != 4),
            "missed_delivery": bool(delivery_available_before and action != 5),
            "early_drop": bool(self.last_event == "early_drop"),
            "repeat_visit": repeat_visit,
            "visit_count": visit_count,
            "step": int(self.steps),
        }
        reward = float(self.reward_function.compute(transition))
        self.last_reward = reward

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            self.last_event = "timeout"
        return self._observation(), reward, terminated, truncated, self._info(success=success)

    def _behavior_state(self) -> tuple[object, ...]:
        return (self.agent_pos, bool(self.carrying), self.object_pos, tuple(sorted(self.open_doors)))

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
                pos = (gr, gc)
                if self.grid[pos] == 1:
                    obs[0, vr, vc] = 1.0
                if pos in self.door_positions:
                    obs[2 if pos in self.open_doors else 1, vr, vc] = 1.0
                if self.object_pos == pos:
                    obs[3, vr, vc] = 1.0
                if self.goal_pos == pos:
                    obs[4, vr, vc] = 1.0
        obs[5, radius, radius] = 1.0
        obs[6, :, :] = 1.0 if self.carrying else 0.0
        waypoint = self._route_waypoint()
        row_scale = float(max(1, self.view_size))
        col_scale = float(max(1, self.view_size))
        obs[7, :, :] = np.clip((waypoint[0] - ar) / row_scale, -1.0, 1.0)
        obs[8, :, :] = np.clip((waypoint[1] - ac) / col_scale, -1.0, 1.0)
        obs[9, :, :] = 1.0 if (not self.carrying and self.object_pos == self.agent_pos) else 0.0
        obs[10, :, :] = 1.0 if (self.carrying and self.agent_pos == self.goal_pos) else 0.0
        adjacent = [
            (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            for dr, dc in self.ACTIONS.values()
        ]
        obs[11, :, :] = 1.0 if any(p in self.door_positions and p not in self.open_doors for p in adjacent) else 0.0
        obs[12, :, :] = 1.0 if any(p in self.open_doors for p in adjacent) else 0.0
        obs[13, :, :] = len(self.open_doors) / max(1, len(self.door_positions))
        return obs

    def valid_action_mask(self, mode: str = "task") -> np.ndarray:
        if mode not in {"task", "valid"}:
            raise ValueError("mask mode must be 'task' or 'valid'.")
        mask = np.zeros(self.action_space_n, dtype=bool)
        pickup_available = (not self.carrying) and self.object_pos == self.agent_pos
        delivery_available = self.carrying and self.agent_pos == self.goal_pos
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
        if self._adjacent_door(prefer_closed=False) is not None:
            mask[6] = True
        return mask

    def _is_floor_or_door(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and (self.grid[pos] == 0 or pos in self.door_positions)

    def _is_free(self, pos: Pos) -> bool:
        if not self._is_floor_or_door(pos):
            return False
        return pos not in self.door_positions or pos in self.open_doors

    def _distance_field(self, goal: Pos, structural: bool = True) -> np.ndarray:
        field = np.full(self.grid.shape, self.grid.size * 4, dtype=np.int32)
        field[goal] = 0
        queue = deque([goal])
        while queue:
            r, c = queue.popleft()
            next_distance = int(field[r, c]) + 1
            for dr, dc in self.ACTIONS.values():
                nxt = (r + dr, c + dc)
                allowed = self._is_floor_or_door(nxt) if structural else self._is_free(nxt)
                if allowed and next_distance < field[nxt]:
                    field[nxt] = next_distance
                    queue.append(nxt)
        return field

    def _target_distance(self, pos: Pos) -> int:
        field = self._goal_distance_field if self.carrying else self._object_distance_field
        return int(field[pos]) if field is not None else 0

    def _weighted_distance(self, start: Pos, goal: Pos) -> int:
        """Shortest estimated action count with closed-door traversal costing toggle+move."""
        inf = self.grid.size * 20
        dist: dict[Pos, int] = {start: 0}
        heap: list[tuple[int, Pos]] = [(0, start)]
        while heap:
            cost, pos = heappop(heap)
            if pos == goal:
                return cost
            if cost != dist.get(pos):
                continue
            for dr, dc in self.ACTIONS.values():
                nxt = (pos[0] + dr, pos[1] + dc)
                if not self._is_floor_or_door(nxt):
                    continue
                step_cost = 1 + int(nxt in self.door_positions and nxt not in self.open_doors)
                new = cost + step_cost
                if new < dist.get(nxt, inf):
                    dist[nxt] = new
                    heappush(heap, (new, nxt))
        return inf

    def _estimate_oracle_steps(self) -> int:
        assert self.object_pos is not None
        return int(self._weighted_distance(self.start_pos, self.object_pos) + 1 + self._weighted_distance(self.object_pos, self.goal_pos) + 1)

    def _size_tier(self) -> str:
        name = self.current_map.name if self.current_map else ""
        for tier in ("large_plusplus", "large_plus", "large", "medium", "small"):
            if f"_{tier}_" in name:
                return tier.replace("_plusplus", "++").replace("_plus", "+")
        return "unknown"

    def _info(self, success: bool) -> Dict[str, object]:
        efficiency = min(1.0, self.oracle_steps / self.steps) if success and self.steps > 0 else 0.0
        return {
            "task_id": self.TASK_ID,
            "task_code": self.TASK_CODE,
            "success": success,
            "carrying": self.carrying,
            "capacity": self.capacity,
            "steps": self.steps,
            "map": self.current_map.name if self.current_map else None,
            "map_shape": tuple(int(v) for v in self.grid.shape),
            "size_tier": self._size_tier(),
            "rooms": len(self.room_cells),
            "start_room": self.start_room,
            "object_room": self.object_room,
            "goal_room": self.goal_room,
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
            "pickups": 1 if self.pickup_step is not None else 0,
            "items_total": 1,
            "items_delivered": 1 if success else 0,
            "completion_rate": 1.0 if success else 0.0,
            "doors_total": len(self.door_positions),
            "doors_open": len(self.open_doors),
            "door_opens": self.door_opens,
            "door_closes": self.door_closes,
            "oracle_steps": self.oracle_steps,
            "path_efficiency": float(efficiency),
            "last_reward": self.last_reward,
            "last_distance_delta": self.last_distance_delta,
            "actionable_target_distance": self._actionable_target_distance(self.agent_pos),
            "route_waypoint": self._route_waypoint(),
            "visited_rooms": len(self._visited_rooms),
            "behavior_state": self._behavior_state(),
        }

    def robot_status(self) -> dict[str, object]:
        return {
            "Robot": "Fox-03",
            "Position": f"{self.agent_pos[0]}, {self.agent_pos[1]}",
            "Room": self.room_index.get(self.agent_pos, "door"),
            "Carried items": 1 if self.carrying else 0,
            "Current target": "Delivery goal" if self.carrying else "Cargo object",
            "Doors open": f"{len(self.open_doors)}/{len(self.door_positions)}",
            "Door toggles": self.door_opens + self.door_closes,
            "Last event": self.last_event,
        }

    def render_entities(self) -> dict[str, list[tuple[int, Pos]]]:
        return {
            "objects": [] if self.object_pos is None else [(0, self.object_pos)],
            "goals": [(0, self.goal_pos)],
        }

    def render_doors(self) -> list[tuple[Pos, bool]]:
        return [(door, door in self.open_doors) for door in sorted(self.door_positions)]

    def render_ascii(self) -> str:
        chars = np.full(self.grid.shape, ".", dtype="<U1")
        chars[self.grid == 1] = "#"
        for door in self.door_positions:
            chars[door] = "d" if door in self.open_doors else "D"
        chars[self.goal_pos] = "G"
        if self.object_pos is not None:
            chars[self.object_pos] = "O"
        chars[self.agent_pos] = "f" if self.carrying else "F"
        return "\n".join("".join(row) for row in chars)
