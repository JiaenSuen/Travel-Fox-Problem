from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tfp.rewards import load_reward_plugin
from tfp.rewards.reward_api import normalize_reward_name

Pos = Tuple[int, int]
INF = 10**9


class KeyedHazardLogisticsEnv:
    """Long-horizon multi-room logistics with access dependencies and dynamic hazards.

    Research focus
    --------------
    The task combines four sources of difficulty that are usually studied separately:
    partial observability, symbolic access dependencies (colored keys/doors), multi-goal
    transport ordering, and two stochastic moving hazards. Every episode builds a
    dependency chain from the room graph: one to three colored access frontiers must be
    opened in order before the deepest cargo rooms become reachable. Keys are persistent
    keycards; cargo capacity remains one.

    Observation channels (37 x V x V)
    ---------------------------------
      Spatial 0..12:
        wall, closed door, open door, locked-R/G/B, cargo, goal, key-R/G/B,
        wolf, agent.
      Broadcast context 13..36:
        carrying, key inventory R/G/B, remaining/delivered cargo, route waypoint dy/dx,
        phase one-hot (key/cargo/goal), pickup/delivery affordances, adjacent-door state,
        two wolf bearings/distances, unlocked-access fraction, episode-time fraction.

    The route cue is intentionally low bandwidth: only the next required doorway (or the
    active key/cargo/goal when no doorway remains) is exposed. The global map, full route,
    room graph and future wolf paths are never part of the policy observation.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP", "TOGGLE_DOOR", "WAIT")
    TASK_ID = "TFP-KeyedHazardLogistics"
    TASK_CODE = "KEYED-HAZARD-LOGISTICS"
    TASK_NAME = "Keyed Multi-Cargo Logistics"
    COLOR_NAMES = ("red", "green", "blue")

    def __init__(
        self,
        map_paths: Sequence[str | Path],
        observation_mode: str = "local",
        view_size: int = 7,
        max_steps: Optional[int] = None,
        seed: int = 0,
        reward_module: str = "001_dependency_risk_potential",
        wolves_enabled: bool = True,
    ) -> None:
        if observation_mode != "local":
            raise ValueError("TFP supports local observation only.")
        if view_size not in {5, 7}:
            raise ValueError("Keyed Multi-Cargo Logistics supports view_size 5 or 7.")
        if not map_paths:
            raise ValueError("At least one map path is required.")

        self.map_paths = [Path(p) for p in map_paths]
        self.observation_mode = observation_mode
        self.view_size = int(view_size)
        self._configured_max_steps = max_steps
        self.max_steps = max_steps or 900
        self.rng = np.random.default_rng(seed)
        self.reward_module = normalize_reward_name(reward_module)
        self.reward_spec, reward_factory = load_reward_plugin(self.reward_module, task_id=self.TASK_ID)
        self.reward_function = reward_factory()
        self.wolves_enabled = bool(wolves_enabled)

        self.action_space_n = 8
        self.observation_shape = (37, self.view_size, self.view_size)
        self.capacity = 1

        self.grid: np.ndarray
        self.current_map: Optional[Path] = None
        self.start_pos: Pos = (0, 0)
        self.agent_pos: Pos = (0, 0)
        self.goal_pos: Pos = (0, 0)
        self.start_room = -1
        self.goal_room = -1
        self.room_cells: list[list[Pos]] = []
        self.room_index: dict[Pos, int] = {}
        self.room_graph: dict[int, set[int]] = {}
        self.door_positions: set[Pos] = set()
        self.door_rooms: dict[Pos, tuple[int, int]] = {}
        self.open_doors: set[Pos] = set()
        self.door_colors: dict[Pos, int] = {}
        self.lock_order_colors: list[int] = []
        self.lock_thresholds: list[int] = []
        self.key_positions: dict[int, Pos] = {}
        self.key_rooms: dict[int, int] = {}
        self.keys_owned: set[int] = set()
        self.cargo_positions: list[Pos] = []
        self.cargo_rooms: list[int] = []
        self.total_cargo = 1
        self.delivered_count = 0
        self.carrying = False
        self.wolf_positions: list[Pos] = []
        self.wolf_targets: list[Optional[Pos]] = [None, None]

        self.steps = 0
        self.collisions = 0
        self.hazard_collisions = 0
        self.invalid_actions = 0
        self.pickup_count = 0
        self.key_pickups = 0
        self.door_opens = 0
        self.door_closes = 0
        self.last_reward = 0.0
        self.last_event = "reset"
        self.failure_reason = ""
        self.oracle_steps = 1
        self._visited_rooms: set[int] = set()
        self._state_visits: dict[tuple[object, ...], int] = {}
        self._cached_target: tuple[str, int, Pos] | None = None
        self._cached_field: np.ndarray | None = None

    # ------------------------------------------------------------------ layout
    def _load_layout(self, path: Path) -> tuple[np.ndarray, Pos, set[Pos]]:
        lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"Map is empty: {path}")
        width = len(lines[0])
        if any(len(line) != width for line in lines):
            raise ValueError(f"Map must be rectangular: {path}")
        grid = np.zeros((len(lines), width), dtype=np.uint8)
        start: Optional[Pos] = None
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
        if start is None or not doors:
            raise ValueError(f"Keyed logistics map needs one A and at least one D: {path}")
        return grid, start, doors

    def _build_rooms(self) -> None:
        self.room_cells = []
        self.room_index = {}
        h, w = self.grid.shape
        visited: set[Pos] = set()
        for r in range(h):
            for c in range(w):
                start = (r, c)
                if start in visited or start in self.door_positions or self.grid[start] == 1:
                    continue
                q = deque([start])
                visited.add(start)
                room_id = len(self.room_cells)
                cells: list[Pos] = []
                while q:
                    pos = q.popleft()
                    cells.append(pos)
                    self.room_index[pos] = room_id
                    for dr, dc in self.ACTIONS.values():
                        nxt = (pos[0] + dr, pos[1] + dc)
                        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                            continue
                        if nxt in visited or nxt in self.door_positions or self.grid[nxt] == 1:
                            continue
                        visited.add(nxt)
                        q.append(nxt)
                self.room_cells.append(cells)

        self.room_graph = {i: set() for i in range(len(self.room_cells))}
        self.door_rooms = {}
        for door in self.door_positions:
            adjacent = []
            for dr, dc in self.ACTIONS.values():
                room = self.room_index.get((door[0] + dr, door[1] + dc))
                if room is not None and room not in adjacent:
                    adjacent.append(room)
            if len(adjacent) != 2:
                raise ValueError(f"Door {door} must connect exactly two rooms in {self.current_map}")
            a, b = adjacent
            self.door_rooms[door] = (a, b)
            self.room_graph[a].add(b)
            self.room_graph[b].add(a)
        self.start_room = self.room_index[self.start_pos]

    def _room_distances(self) -> tuple[dict[int, int], dict[int, int], dict[int, Pos]]:
        dist = {self.start_room: 0}
        parent = {self.start_room: self.start_room}
        parent_door: dict[int, Pos] = {}
        q = deque([self.start_room])
        while q:
            room = q.popleft()
            for nxt in sorted(self.room_graph[room]):
                if nxt in dist:
                    continue
                door = next(d for d, pair in self.door_rooms.items() if {pair[0], pair[1]} == {room, nxt})
                dist[nxt] = dist[room] + 1
                parent[nxt] = room
                parent_door[nxt] = door
                q.append(nxt)
        return dist, parent, parent_door

    def _deep_room_cells(self, room: int, occupied: set[Pos] | None = None) -> list[Pos]:
        occupied = occupied or set()
        doors = [d for d, pair in self.door_rooms.items() if room in pair]
        cells = [p for p in self.room_cells[room] if p not in occupied and p != self.start_pos]
        if not cells:
            cells = [p for p in self.room_cells[room] if p not in occupied]
        if not cells:
            raise RuntimeError(f"Room {room} has no free entity cell")
        if doors:
            scored = sorted(cells, key=lambda p: min(abs(p[0]-d[0]) + abs(p[1]-d[1]) for d in doors), reverse=True)
            keep = max(1, len(scored) // 3)
            return scored[:keep]
        return cells

    def _configure_access_dependencies(self) -> None:
        dist, _, _ = self._room_distances()
        max_depth = max(dist.values())
        room_count = len(self.room_cells)
        lock_count = 1 + int(room_count >= 9) + int(room_count >= 16)
        lock_count = min(3, lock_count, max_depth)
        # Access frontiers are BFS-depth cuts. Locking *all* doors crossing a frontier
        # prevents cyclic room graphs from accidentally bypassing the intended key.
        raw = [max(1, int(round((i + 1) * max_depth / (lock_count + 1)))) for i in range(lock_count)]
        thresholds: list[int] = []
        for value in raw:
            value = min(max_depth, value)
            if value not in thresholds:
                thresholds.append(value)
        while len(thresholds) < lock_count:
            for d in range(1, max_depth + 1):
                if d not in thresholds:
                    thresholds.append(d)
                    if len(thresholds) == lock_count:
                        break
        thresholds.sort()
        self.lock_thresholds = thresholds
        self.lock_order_colors = list(range(len(thresholds)))
        self.door_colors = {}
        for color, threshold in enumerate(thresholds):
            for door, (a, b) in self.door_rooms.items():
                da, db = dist[a], dist[b]
                if min(da, db) < threshold <= max(da, db):
                    # The first frontier owning this edge determines its color.
                    self.door_colors.setdefault(door, color)

        # Key i appears immediately before frontier i, producing an explicit but
        # learnable dependency chain: key -> colored door -> next access zone.
        occupied: set[Pos] = {self.start_pos}
        self.key_positions = {}
        self.key_rooms = {}
        previous_threshold = 0
        for color, threshold in enumerate(thresholds):
            band = [r for r, d in dist.items() if previous_threshold <= d < threshold]
            if not band:
                band = [r for r, d in dist.items() if d < threshold]
            max_band_depth = max(dist[r] for r in band)
            candidates = [r for r in band if dist[r] == max_band_depth]
            room = int(self.rng.choice(candidates))
            cells = self._deep_room_cells(room, occupied)
            pos = cells[int(self.rng.integers(0, len(cells)))]
            self.key_positions[color] = pos
            self.key_rooms[color] = room
            occupied.add(pos)
            previous_threshold = threshold

        # Colored doors always begin closed. Ordinary doors vary by seed.
        self.open_doors = {
            door for door in self.door_positions
            if door not in self.door_colors and float(self.rng.random()) < 0.42
        }

    def _spawn_mission_entities(self) -> None:
        dist, _, _ = self._room_distances()
        occupied = {self.start_pos, *self.key_positions.values()}
        max_depth = max(dist.values())
        deepest = [r for r, d in dist.items() if d == max_depth]
        deep_pool = [r for r, d in dist.items() if d >= max(1, max_depth - 2)]

        # Goal is deliberately on the shallow side so carrying cargo back through the
        # unlocked access chain remains part of the problem rather than a one-way trip.
        shallow = [r for r, d in dist.items() if d <= max(1, max_depth // 3)]
        self.goal_room = int(self.rng.choice(shallow))
        goal_cells = self._deep_room_cells(self.goal_room, occupied)
        self.goal_pos = goal_cells[int(self.rng.integers(0, len(goal_cells)))]
        occupied.add(self.goal_pos)

        self.total_cargo = int(self.rng.integers(1, 4))
        rooms: list[int] = [int(self.rng.choice(deepest))]
        available = [r for r in deep_pool if r not in rooms]
        self.rng.shuffle(available)
        rooms.extend(available[: self.total_cargo - 1])
        while len(rooms) < self.total_cargo:
            rooms.append(int(self.rng.choice(deep_pool)))
        self.cargo_rooms = rooms
        self.cargo_positions = []
        for room in rooms:
            cells = self._deep_room_cells(room, occupied)
            pos = cells[int(self.rng.integers(0, len(cells)))]
            self.cargo_positions.append(pos)
            occupied.add(pos)

        self.wolf_positions = []
        self.wolf_targets = [None, None]
        if self.wolves_enabled:
            candidates = [p for room in self.room_cells for p in room if p not in occupied and p != self.start_pos]
            # Keep initial hazards out of immediate spawn range, but otherwise distribute
            # them across the full room graph.
            candidates = [p for p in candidates if abs(p[0]-self.start_pos[0]) + abs(p[1]-self.start_pos[1]) >= 5]
            self.rng.shuffle(candidates)
            for p in candidates:
                if all(abs(p[0]-q[0]) + abs(p[1]-q[1]) >= 5 for q in self.wolf_positions):
                    self.wolf_positions.append(p)
                    if len(self.wolf_positions) == 2:
                        break
            if len(self.wolf_positions) < 2:
                self.wolf_positions = candidates[:2]

    # ------------------------------------------------------------ navigation/planner
    def _is_structural(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and (self.grid[pos] == 0 or pos in self.door_positions)

    def _is_free(self, pos: Pos) -> bool:
        if not self._is_structural(pos):
            return False
        if pos in self.door_positions:
            if pos in self.door_colors and self.door_colors[pos] not in self.keys_owned:
                return False
            return pos in self.open_doors
        return True

    def _enter_cost(self, pos: Pos, keys: set[int] | None = None, open_doors: set[Pos] | None = None) -> int:
        keys = self.keys_owned if keys is None else keys
        open_doors = self.open_doors if open_doors is None else open_doors
        if not self._is_structural(pos):
            return INF
        if pos not in self.door_positions:
            return 1
        color = self.door_colors.get(pos)
        if color is not None and color not in keys:
            return INF
        return 1 if pos in open_doors else 2  # TOGGLE + MOVE

    def _action_distance_field(self, target: Pos, keys: set[int] | None = None, open_doors: set[Pos] | None = None) -> np.ndarray:
        field = np.full(self.grid.shape, INF, dtype=np.int32)
        field[target] = 0
        heap: list[tuple[int, Pos]] = [(0, target)]
        while heap:
            cost, pos = heappop(heap)
            if cost != int(field[pos]):
                continue
            enter = self._enter_cost(pos, keys=keys, open_doors=open_doors)
            if enter >= INF:
                continue
            for dr, dc in self.ACTIONS.values():
                prev = (pos[0] + dr, pos[1] + dc)
                if not self._is_structural(prev):
                    continue
                new = cost + enter
                if new < int(field[prev]):
                    field[prev] = new
                    heappush(heap, (new, prev))
        return field

    def _distance_cost(self, start: Pos, target: Pos, keys: set[int] | None = None, open_doors: set[Pos] | None = None) -> int:
        field = self._action_distance_field(target, keys=keys, open_doors=open_doors)
        return int(field[start])

    def _active_target(self) -> tuple[str, int, Pos]:
        # Dependency-first symbolic bottleneck. Keys are collected in access-frontier
        # order; after access is resolved, capacity-one cargo is ordered by estimated
        # pickup+delivery action cost.
        for color in self.lock_order_colors:
            if color not in self.keys_owned and color in self.key_positions:
                return ("key", color, self.key_positions[color])
        if self.carrying:
            return ("goal", -1, self.goal_pos)
        if self.cargo_positions:
            best: tuple[int, int, Pos] | None = None
            for idx, cargo in enumerate(self.cargo_positions):
                to_cargo = self._distance_cost(self.agent_pos, cargo)
                to_goal = self._distance_cost(cargo, self.goal_pos)
                score = to_cargo + to_goal
                candidate = (score, idx, cargo)
                if best is None or candidate < best:
                    best = candidate
            assert best is not None
            return ("cargo", best[1], best[2])
        return ("goal", -1, self.goal_pos)

    def _target_field(self) -> tuple[tuple[str, int, Pos], np.ndarray]:
        target = self._active_target()
        if self._cached_target != target or self._cached_field is None:
            self._cached_target = target
            self._cached_field = self._action_distance_field(target[2])
        return target, self._cached_field

    def _invalidate_planner(self) -> None:
        self._cached_target = None
        self._cached_field = None

    def _target_distance(self, pos: Pos) -> int:
        _, field = self._target_field()
        return int(field[pos])

    def _route_waypoint(self) -> Pos:
        target, field = self._target_field()
        current = self.agent_pos
        visited = {current}
        for _ in range(self.grid.size):
            if current == target[2]:
                return target[2]
            current_cost = int(field[current])
            candidates: list[tuple[int, int, Pos]] = []
            for action, (dr, dc) in self.ACTIONS.items():
                nxt = (current[0] + dr, current[1] + dc)
                if nxt in visited or not self._is_structural(nxt):
                    continue
                enter = self._enter_cost(nxt)
                if enter >= INF:
                    continue
                if current_cost == int(field[nxt]) + enter:
                    tie = abs(target[2][0] - nxt[0]) + abs(target[2][1] - nxt[1])
                    candidates.append((tie, action, nxt))
            if not candidates:
                return target[2]
            _, _, nxt = min(candidates)
            if nxt in self.door_positions:
                return nxt
            visited.add(nxt)
            current = nxt
        return target[2]

    def _adjacent_door(self, route_preferred: bool = True) -> Optional[Pos]:
        adjacent: list[Pos] = []
        for action in range(4):
            dr, dc = self.ACTIONS[action]
            p = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if p in self.door_positions:
                adjacent.append(p)
        if not adjacent:
            return None
        if route_preferred:
            waypoint = self._route_waypoint()
            if waypoint in adjacent:
                return waypoint
        closed = [p for p in adjacent if p not in self.open_doors]
        return (closed or adjacent)[0]

    # --------------------------------------------------------------- hazards
    def _wolf_passable(self, pos: Pos) -> bool:
        if not self._is_structural(pos):
            return False
        color = self.door_colors.get(pos)
        # Predators can push ordinary/unlocked doors but cannot cross an access door
        # until the matching key has been acquired by the agent.
        return color is None or color in self.keys_owned

    def _wolf_component(self, start: Pos) -> list[Pos]:
        q = deque([start]); seen = {start}; out: list[Pos] = []
        while q:
            p = q.popleft(); out.append(p)
            for dr, dc in self.ACTIONS.values():
                n = (p[0] + dr, p[1] + dc)
                if n not in seen and self._wolf_passable(n):
                    seen.add(n); q.append(n)
        return out

    def _wolf_next_step(self, start: Pos, target: Pos) -> Pos:
        if start == target:
            return start
        q = deque([start]); parent: dict[Pos, Optional[Pos]] = {start: None}
        found = False
        while q and not found:
            p = q.popleft()
            neighbors = []
            for dr, dc in self.ACTIONS.values():
                n = (p[0] + dr, p[1] + dc)
                if n not in parent and self._wolf_passable(n):
                    neighbors.append(n)
            self.rng.shuffle(neighbors)
            for n in neighbors:
                parent[n] = p
                if n == target:
                    found = True
                    break
                q.append(n)
        if target not in parent:
            return start
        cur = target
        while parent[cur] is not None and parent[cur] != start:
            cur = parent[cur]  # type: ignore[index]
        return cur

    def _move_wolves(self) -> None:
        if not self.wolves_enabled:
            return
        new_positions: list[Pos] = []
        for i, wolf in enumerate(self.wolf_positions):
            component = self._wolf_component(wolf)
            target = self.wolf_targets[i]
            if target not in component or target == wolf or float(self.rng.random()) < 0.035:
                far = [p for p in component if abs(p[0]-wolf[0]) + abs(p[1]-wolf[1]) >= max(4, self.grid.shape[0] // 4)]
                pool = far or component
                target = pool[int(self.rng.integers(0, len(pool)))]
                self.wolf_targets[i] = target
            nxt = self._wolf_next_step(wolf, target)
            if nxt in new_positions:
                nxt = wolf
            new_positions.append(nxt)
        self.wolf_positions = new_positions

    def _risk_score(self, pos: Pos) -> float:
        score = 0.0
        for wolf in self.wolf_positions:
            d = abs(pos[0] - wolf[0]) + abs(pos[1] - wolf[1])
            score += float(np.exp(-0.55 * d))
        return score

    # ------------------------------------------------------------------- reset
    def reset(self, seed: Optional[int] = None, map_path: Optional[str | Path] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_map = Path(map_path) if map_path is not None else self.map_paths[int(self.rng.integers(0, len(self.map_paths)))]
        self.grid, self.start_pos, self.door_positions = self._load_layout(self.current_map)
        self._build_rooms()
        self.agent_pos = self.start_pos
        self.keys_owned = set()
        self.carrying = False
        self.delivered_count = 0
        self.steps = 0
        self.collisions = 0
        self.hazard_collisions = 0
        self.invalid_actions = 0
        self.pickup_count = 0
        self.key_pickups = 0
        self.door_opens = 0
        self.door_closes = 0
        self.last_reward = 0.0
        self.last_event = "reset"
        self.failure_reason = ""
        self.max_steps = self._configured_max_steps or max(700, 19 * (self.grid.shape[0] + self.grid.shape[1]) + 180 * 3)
        self._configure_access_dependencies()
        self._spawn_mission_entities()
        self._visited_rooms = {self.start_room}
        self._state_visits = {}
        self._invalidate_planner()
        self.oracle_steps = self._estimate_oracle_steps()
        self._state_visits[self._behavior_state()] = 1
        if not self.structurally_solvable():
            raise RuntimeError(f"Generated episode is structurally unsolvable: {self.current_map}")
        return self._observation(), self._info(success=False)

    # -------------------------------------------------------------------- step
    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.action_space_n:
            raise ValueError(f"Invalid action {action}")
        self.steps += 1
        terminated = False
        truncated = False
        success = False
        invalid = False
        collision = False
        moved = False
        key_pickup = False
        cargo_pickup = False
        delivered = False
        door_opened = False
        door_closed = False
        locked_without_key = False
        wait_action = action == 7

        target_before, _ = self._target_field()
        distance_before = self._target_distance(self.agent_pos)
        risk_before = self._risk_score(self.agent_pos)
        room_before = self._room_for_agent()
        pickup_available_before = self._pickup_available()
        delivery_available_before = self.carrying and self.agent_pos == self.goal_pos

        if action in self.ACTIONS:
            dr, dc = self.ACTIONS[action]
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self._is_free(nxt):
                self.agent_pos = nxt
                moved = True
                self.last_event = "move"
            else:
                self.collisions += 1
                self.invalid_actions += 1
                collision = invalid = True
                self.last_event = "collision"
        elif action == 4:
            key_color = next((color for color, pos in self.key_positions.items() if color not in self.keys_owned and pos == self.agent_pos), None)
            if key_color is not None:
                self.keys_owned.add(int(key_color))
                self.key_pickups += 1
                key_pickup = True
                self.last_event = f"key_{self.COLOR_NAMES[int(key_color)]}"
                self._invalidate_planner()
            elif (not self.carrying) and self.agent_pos in self.cargo_positions:
                self.cargo_positions.remove(self.agent_pos)
                self.carrying = True
                self.pickup_count += 1
                cargo_pickup = True
                self.last_event = "cargo_pickup"
                self._invalidate_planner()
            else:
                self.invalid_actions += 1; invalid = True; self.last_event = "invalid_pickup"
        elif action == 5:
            if self.carrying and self.agent_pos == self.goal_pos:
                self.carrying = False
                self.delivered_count += 1
                delivered = True
                self.last_event = "delivery"
                self._invalidate_planner()
                if self.delivered_count >= self.total_cargo:
                    success = True
                    terminated = True
                    self.last_event = "complete"
            elif self.carrying:
                self.carrying = False
                self.cargo_positions.append(self.agent_pos)
                self.invalid_actions += 1; invalid = True; self.last_event = "early_drop"
                self._invalidate_planner()
            else:
                self.invalid_actions += 1; invalid = True; self.last_event = "invalid_drop"
        elif action == 6:
            door = self._adjacent_door(route_preferred=True)
            if door is None:
                self.invalid_actions += 1; invalid = True; self.last_event = "invalid_toggle"
            else:
                color = self.door_colors.get(door)
                if color is not None and color not in self.keys_owned:
                    self.invalid_actions += 1; invalid = True; locked_without_key = True
                    self.last_event = "locked_door"
                elif door in self.open_doors:
                    self.open_doors.remove(door); self.door_closes += 1; door_closed = True
                    self.last_event = "door_close"; self._invalidate_planner()
                else:
                    self.open_doors.add(door); self.door_opens += 1; door_opened = True
                    self.last_event = "door_open"; self._invalidate_planner()
        elif action == 7:
            self.last_event = "wait"

        # Agent-caused risk/progress are measured before stochastic hazard motion.
        risk_after_agent = self._risk_score(self.agent_pos)
        risk_improvement = float(risk_before - risk_after_agent)
        milestone = key_pickup or cargo_pickup or delivered or self.last_event == "early_drop"
        if milestone:
            distance_after = distance_before
            progress_delta = 0
        else:
            # Compare the same symbolic target where possible. A door toggle can change
            # executable cost without changing the target; that is valid shaping.
            if self._active_target()[:2] == target_before[:2]:
                distance_after = self._distance_cost(self.agent_pos, target_before[2])
                progress_delta = int(distance_before - distance_after)
            else:
                distance_after = distance_before
                progress_delta = 0

        predator_collision = False
        if self.agent_pos in self.wolf_positions and not terminated:
            predator_collision = True
        if not terminated and not predator_collision:
            self._move_wolves()
            predator_collision = self.agent_pos in self.wolf_positions
        if predator_collision:
            terminated = True
            success = False
            self.hazard_collisions += 1
            self.collisions += 1
            self.failure_reason = "wolf_collision"
            self.last_event = "wolf_collision"

        room_after = self._room_for_agent()
        entered_new_room = bool(moved and room_after != room_before and room_after not in self._visited_rooms)
        self._visited_rooms.add(room_after)
        state = self._behavior_state()
        visits = self._state_visits.get(state, 0) + 1
        self._state_visits[state] = visits
        repeat_visit = visits >= 3

        transition = {
            "action": action,
            "moved": moved,
            "invalid": invalid,
            "collision": collision,
            "predator_collision": predator_collision,
            "key_pickup": key_pickup,
            "cargo_pickup": cargo_pickup,
            "pickup": key_pickup or cargo_pickup,
            "delivered": delivered,
            "task_complete": success,
            "door_opened": door_opened,
            "door_closed": door_closed,
            "locked_without_key": locked_without_key,
            "distance_before": int(distance_before if distance_before < INF else self.grid.size * 10),
            "distance_after": int(distance_after if distance_after < INF else self.grid.size * 10),
            "progress_delta": int(progress_delta),
            "risk_before": float(risk_before),
            "risk_after_agent": float(risk_after_agent),
            "risk_improvement": float(risk_improvement),
            "repeat_visit": repeat_visit,
            "visit_count": visits,
            "entered_new_room": entered_new_room,
            "wait_action": wait_action,
            "early_drop": self.last_event == "early_drop",
            "keys_owned": len(self.keys_owned),
            "keys_total": len(self.lock_order_colors),
            "delivered_count": self.delivered_count,
            "items_total": self.total_cargo,
            "remaining_count": self.total_cargo - self.delivered_count,
            "step": self.steps,
        }
        reward = float(self.reward_function.compute(transition))
        self.last_reward = reward

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            self.failure_reason = "timeout"
            self.last_event = "timeout"
        return self._observation(), reward, terminated, truncated, self._info(success=success)

    # ------------------------------------------------------------- observations
    def _pickup_available(self) -> bool:
        if any(pos == self.agent_pos and color not in self.keys_owned for color, pos in self.key_positions.items()):
            return True
        return (not self.carrying) and self.agent_pos in self.cargo_positions

    def _room_for_agent(self) -> int:
        if self.agent_pos in self.room_index:
            return self.room_index[self.agent_pos]
        adjacent = []
        for dr, dc in self.ACTIONS.values():
            room = self.room_index.get((self.agent_pos[0] + dr, self.agent_pos[1] + dc))
            if room is not None:
                adjacent.append(room)
        for room in adjacent:
            if room in self._visited_rooms:
                return room
        return adjacent[0] if adjacent else self.start_room

    def _wolf_context(self) -> list[tuple[float, float, float]]:
        h, w = self.grid.shape
        scale = float(max(h, w))
        wolves = sorted(
            self.wolf_positions,
            key=lambda p: abs(p[0]-self.agent_pos[0]) + abs(p[1]-self.agent_pos[1]),
        )
        output: list[tuple[float, float, float]] = []
        for wolf in wolves[:2]:
            dy = np.clip((wolf[0] - self.agent_pos[0]) / 8.0, -1.0, 1.0)
            dx = np.clip((wolf[1] - self.agent_pos[1]) / 8.0, -1.0, 1.0)
            dist = min(1.0, (abs(wolf[0]-self.agent_pos[0]) + abs(wolf[1]-self.agent_pos[1])) / max(1.0, scale * 0.5))
            output.append((float(dy), float(dx), float(dist)))
        while len(output) < 2:
            output.append((0.0, 0.0, 1.0))
        return output

    def _observation(self) -> np.ndarray:
        v = self.view_size; radius = v // 2
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        ar, ac = self.agent_pos
        for vr in range(v):
            for vc in range(v):
                gr, gc = ar + vr - radius, ac + vc - radius
                if not (0 <= gr < self.grid.shape[0] and 0 <= gc < self.grid.shape[1]):
                    obs[0, vr, vc] = 1.0; continue
                pos = (gr, gc)
                if self.grid[pos] == 1:
                    obs[0, vr, vc] = 1.0
                if pos in self.door_positions:
                    color = self.door_colors.get(pos)
                    if color is not None and color not in self.keys_owned:
                        obs[3 + int(color), vr, vc] = 1.0
                    else:
                        obs[2 if pos in self.open_doors else 1, vr, vc] = 1.0
                if pos in self.cargo_positions: obs[6, vr, vc] = 1.0
                if pos == self.goal_pos: obs[7, vr, vc] = 1.0
                for color, key_pos in self.key_positions.items():
                    if color not in self.keys_owned and pos == key_pos:
                        obs[8 + int(color), vr, vc] = 1.0
                if pos in self.wolf_positions: obs[11, vr, vc] = 1.0
        obs[12, radius, radius] = 1.0

        obs[13, :, :] = float(self.carrying)
        for color in range(3): obs[14 + color, :, :] = float(color in self.keys_owned)
        obs[17, :, :] = (self.total_cargo - self.delivered_count - int(self.carrying)) / 3.0
        obs[18, :, :] = self.delivered_count / max(1, self.total_cargo)
        waypoint = self._route_waypoint()
        obs[19, :, :] = np.clip((waypoint[0] - ar) / max(1.0, self.grid.shape[0] / 2), -1.0, 1.0)
        obs[20, :, :] = np.clip((waypoint[1] - ac) / max(1.0, self.grid.shape[1] / 2), -1.0, 1.0)
        phase = self._active_target()[0]
        obs[21, :, :] = float(phase == "key")
        obs[22, :, :] = float(phase == "cargo")
        obs[23, :, :] = float(phase == "goal")
        obs[24, :, :] = float(self._pickup_available())
        obs[25, :, :] = float(self.carrying and self.agent_pos == self.goal_pos)
        adjacent = [(ar + dr, ac + dc) for dr, dc in self.ACTIONS.values()]
        obs[26, :, :] = float(any(p in self.door_positions and p not in self.open_doors and (self.door_colors.get(p) is None or self.door_colors.get(p) in self.keys_owned) for p in adjacent))
        obs[27, :, :] = float(any(p in self.door_colors and self.door_colors[p] in self.keys_owned and p not in self.open_doors for p in adjacent))
        obs[28, :, :] = float(any(p in self.door_colors and self.door_colors[p] not in self.keys_owned for p in adjacent))
        wolves = self._wolf_context()
        obs[29, :, :], obs[30, :, :], obs[31, :, :] = wolves[0]
        obs[32, :, :], obs[33, :, :], obs[34, :, :] = wolves[1]
        obs[35, :, :] = len(self.keys_owned) / max(1, len(self.lock_order_colors))
        obs[36, :, :] = min(1.0, self.steps / max(1, self.max_steps))
        return obs

    def valid_action_mask(self, mode: str = "task") -> np.ndarray:
        if mode not in {"task", "valid"}:
            raise ValueError("mask mode must be 'task' or 'valid'.")
        mask = np.zeros(self.action_space_n, dtype=bool)
        pickup = self._pickup_available()
        delivery = self.carrying and self.agent_pos == self.goal_pos
        if mode == "task" and pickup:
            mask[4] = True; return mask
        if mode == "task" and delivery:
            mask[5] = True; return mask
        for action, (dr, dc) in self.ACTIONS.items():
            if self._is_free((self.agent_pos[0] + dr, self.agent_pos[1] + dc)):
                mask[action] = True
        if pickup: mask[4] = True
        if delivery: mask[5] = True
        if self._adjacent_door(route_preferred=False) is not None: mask[6] = True
        mask[7] = True  # WAIT is essential for dynamic-hazard timing.
        return mask

    # ---------------------------------------------------------- diagnostics/info
    def _behavior_state(self) -> tuple[object, ...]:
        # Wolves are excluded intentionally: this state diagnoses controller loops, not
        # stochastic hazard motion.
        return (
            self.agent_pos, bool(self.carrying), tuple(sorted(self.keys_owned)),
            tuple(sorted(self.cargo_positions)), self.delivered_count, tuple(sorted(self.open_doors)),
        )

    def structurally_solvable(self) -> bool:
        """Check key/door dependencies at room-graph level, ignoring stochastic wolves."""
        room_keys = {room: color for color, room in self.key_rooms.items()}
        owned: set[int] = set()
        reachable = {self.start_room}
        changed = True
        while changed:
            changed = False
            for room in list(reachable):
                color = room_keys.get(room)
                if color is not None and color not in owned:
                    owned.add(color); changed = True
            for door, (a, b) in self.door_rooms.items():
                color = self.door_colors.get(door)
                if color is not None and color not in owned:
                    continue
                if a in reachable and b not in reachable:
                    reachable.add(b); changed = True
                if b in reachable and a not in reachable:
                    reachable.add(a); changed = True
        return self.goal_room in reachable and all(room in reachable for room in self.cargo_rooms)

    def _estimate_oracle_steps(self) -> int:
        keys: set[int] = set()
        pos = self.start_pos
        total = 0
        for color in self.lock_order_colors:
            key_pos = self.key_positions[color]
            d = self._distance_cost(pos, key_pos, keys=keys, open_doors=self.open_doors)
            total += max(1, d) + 1
            pos = key_pos; keys.add(color)
        remaining = list(self.cargo_positions)
        while remaining:
            scored = []
            for cargo in remaining:
                scored.append((self._distance_cost(pos, cargo, keys=keys, open_doors=self.open_doors) + self._distance_cost(cargo, self.goal_pos, keys=keys, open_doors=self.open_doors), cargo))
            _, cargo = min(scored)
            total += self._distance_cost(pos, cargo, keys=keys, open_doors=self.open_doors) + 1
            total += self._distance_cost(cargo, self.goal_pos, keys=keys, open_doors=self.open_doors) + 1
            pos = self.goal_pos
            remaining.remove(cargo)
        return max(1, int(total))

    def _size_tier(self) -> str:
        size = int(self.grid.shape[0])
        return {19: "medium", 23: "large", 27: "large+", 31: "large++", 35: "large+++"}.get(size, f"{size}x{size}")

    def _info(self, success: bool) -> Dict[str, object]:
        efficiency = min(1.0, self.oracle_steps / self.steps) if success and self.steps > 0 else 0.0
        target = self._active_target()
        return {
            "task_id": self.TASK_ID,
            "task_code": self.TASK_CODE,
            "success": bool(success),
            "carrying": self.carrying,
            "capacity": self.capacity,
            "steps": self.steps,
            "map": self.current_map.name if self.current_map else None,
            "map_shape": tuple(int(v) for v in self.grid.shape),
            "size_tier": self._size_tier(),
            "rooms": len(self.room_cells),
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "cargo_positions": tuple(self.cargo_positions),
            "keys_owned": tuple(self.COLOR_NAMES[c] for c in sorted(self.keys_owned)),
            "keys_total": len(self.lock_order_colors),
            "wolves": tuple(self.wolf_positions),
            "failure_reason": self.failure_reason,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "reward_module": self.reward_module,
            "collisions": self.collisions,
            "hazard_collisions": self.hazard_collisions,
            "invalid_actions": self.invalid_actions,
            "pickups": self.pickup_count,
            "key_pickups": self.key_pickups,
            "items_total": self.total_cargo,
            "items_delivered": self.delivered_count,
            "completion_rate": self.delivered_count / max(1, self.total_cargo),
            "doors_total": len(self.door_positions),
            "colored_doors": len(self.door_colors),
            "doors_open": len(self.open_doors),
            "door_opens": self.door_opens,
            "door_closes": self.door_closes,
            "oracle_steps": self.oracle_steps,
            "path_efficiency": float(efficiency),
            "last_reward": self.last_reward,
            "active_subgoal": target[0],
            "route_waypoint": self._route_waypoint(),
            "visited_rooms": len(self._visited_rooms),
            "behavior_state": self._behavior_state(),
        }

    def robot_status(self) -> dict[str, object]:
        keys = ", ".join(self.COLOR_NAMES[c][0].upper() for c in sorted(self.keys_owned)) or "—"
        return {
            "Robot": "Courier",
            "Position": f"{self.agent_pos[0]}, {self.agent_pos[1]}",
            "Keys": keys,
            "Cargo": f"{self.delivered_count}/{self.total_cargo} delivered" + (" + carrying" if self.carrying else ""),
            "Subgoal": self._active_target()[0],
            "Wolves": " / ".join(f"{p[0]},{p[1]}" for p in self.wolf_positions) or "disabled",
            "Doors": f"{len(self.open_doors)}/{len(self.door_positions)} open",
            "Last event": self.last_event,
            "Failure": self.failure_reason or "—",
        }

    # --------------------------------------------------------------- rendering
    def render_entities(self) -> dict[str, list[tuple[int, Pos]]]:
        return {"objects": [(0, p) for p in self.cargo_positions], "goals": [(0, self.goal_pos)]}

    def render_doors(self) -> list[tuple[Pos, bool]]:
        return [(door, door in self.open_doors) for door in sorted(self.door_positions) if door not in self.door_colors]

    def render_colored_doors(self) -> list[tuple[Pos, bool, int, bool]]:
        return [
            (door, door in self.open_doors, int(color), color not in self.keys_owned)
            for door, color in sorted(self.door_colors.items())
        ]

    def render_keys(self) -> list[tuple[int, Pos]]:
        return [(int(color), pos) for color, pos in sorted(self.key_positions.items()) if color not in self.keys_owned]

    def render_hazards(self) -> list[tuple[str, Pos]]:
        return [("wolf", p) for p in self.wolf_positions]

    def render_ascii(self) -> str:
        chars = np.full(self.grid.shape, ".", dtype="<U1")
        chars[self.grid == 1] = "#"
        for door in self.door_positions:
            color = self.door_colors.get(door)
            if color is not None and color not in self.keys_owned:
                chars[door] = "RGB"[color]
            else:
                chars[door] = "d" if door in self.open_doors else "D"
        for color, pos in self.key_positions.items():
            if color not in self.keys_owned: chars[pos] = "rgb"[color]
        for p in self.cargo_positions: chars[p] = "O"
        chars[self.goal_pos] = "G"
        for p in self.wolf_positions: chars[p] = "W"
        chars[self.agent_pos] = "f" if self.carrying else "F"
        return "\n".join("".join(row) for row in chars)
