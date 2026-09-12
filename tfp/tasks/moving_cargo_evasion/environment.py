from __future__ import annotations

from collections import deque
from pathlib import Path
import json
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from tfp.rewards import load_reward_plugin
from tfp.rewards.reward_api import normalize_reward_name

Pos = Tuple[int, int]


class MovingCargoEvasionEnv:
    """Dynamic transport with a cyclic cargo carrier and a roaming collision hazard.

    The cargo starts on a carrier that moves around a fixed four-neighbor cyclic track.
    The agent must intercept the carrier, pick up the cargo, and deliver it to the goal.
    A white wolf follows a seeded random-waypoint roaming process over traversable cells.
    Contact with the wolf terminates the episode as failure.

    Observation channels (24 x V x V):
      0 wall/outside; 1 track network; 2 cargo carrier; 3 goal; 4 white wolf; 5 agent;
      6 carrying; 7/8 route waypoint row/column offset; 9/10 carrier velocity;
      11/12 wolf bearing; 13 wolf proximity; 14 pickup available; 15 delivery available;
      16 normalized intercept ETA; 17 carrier cadence phase; 18/19 predicted carrier offset;
      20/21 observed wolf velocity; 22 interception slack; 23 upcoming carrier-turn proximity.

    The route waypoint is a two-value navigation cue computed from the earliest feasible
    interception point before pickup and from the delivery goal afterwards. It exposes
    neither the global map nor a complete route. The wolf bearing/proximity channels act
    as compact hazard telemetry so avoidance remains learnable under local vision.
    """

    ACTIONS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "PICKUP", "DROP", "WAIT")
    TASK_ID = "TFP-MovingCargoEvasion"
    TASK_CODE = "MOVING-CARGO-EVASION"
    TASK_NAME = "Moving Cargo & Predator Avoidance"
    VEHICLE_CADENCE = 2

    def __init__(
        self,
        map_paths: Sequence[str | Path],
        observation_mode: str = "local",
        view_size: int = 7,
        max_steps: Optional[int] = None,
        seed: int = 0,
        reward_module: str = "001_counterfactual_intercept_risk",
    ) -> None:
        if observation_mode != "local":
            raise ValueError("TFP supports local observation only.")
        if view_size not in {5, 7}:
            raise ValueError("Moving Cargo & Predator Avoidance supports view_size 5 or 7.")
        if not map_paths:
            raise ValueError("At least one map path is required.")

        self.map_paths = [Path(p) for p in map_paths]
        self.observation_mode = observation_mode
        self.view_size = int(view_size)
        self._configured_max_steps = max_steps
        self.max_steps = max_steps or 240
        self.rng = np.random.default_rng(seed)
        self.reward_module = normalize_reward_name(reward_module)
        self.reward_spec, reward_factory = load_reward_plugin(self.reward_module, task_id=self.TASK_ID)
        self.reward_function = reward_factory()

        self.action_space_n = 7
        self.observation_shape = (24, self.view_size, self.view_size)
        self.capacity = 1

        self.grid: np.ndarray
        self.start_pos: Pos
        self.agent_pos: Pos
        self.goal_pos: Pos
        self.track_cells: list[Pos] = []
        self.track_set: set[Pos] = set()
        self.route_family = "legacy_cycle"
        self.structure_family = "unknown"
        self.route_metrics: dict[str, int] = {}
        self.vehicle_cadence = self.VEHICLE_CADENCE
        self._track_distance_fields: list[np.ndarray] = []
        self._goal_distance_field: np.ndarray | None = None
        self.vehicle_index = 0
        self.vehicle_direction = 1
        self.vehicle_clock = 0
        self.vehicle_moves = 0
        self.cargo_on_vehicle = True
        self.carrying = False
        self.wolf_pos: Pos = (0, 0)
        self.wolf_prev_pos: Pos = (0, 0)
        self.wolf_heading = 0
        self.wolf_target: Pos = (0, 0)
        self._wolf_target_field: np.ndarray | None = None
        self.wolf_target_ttl = 0
        self.steps = 0
        self.current_map: Path | None = None
        self.collisions = 0
        self.hazard_collisions = 0
        self.invalid_actions = 0
        self.pickup_step: int | None = None
        self.oracle_steps = 0
        self.last_reward = 0.0
        self.last_event = "reset"
        self.failure_reason = ""

    @property
    def vehicle_pos(self) -> Pos:
        return self.track_cells[self.vehicle_index]

    def _parse_map(self, path: Path) -> tuple[np.ndarray, Pos, Pos, set[Pos]]:
        lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines or any(len(line) != len(lines[0]) for line in lines):
            raise ValueError(f"Map must be non-empty and rectangular: {path}")
        grid = np.zeros((len(lines), len(lines[0])), dtype=np.uint8)
        start = goal = None
        track: set[Pos] = set()
        for r, line in enumerate(lines):
            for c, char in enumerate(line):
                if char == "#":
                    grid[r, c] = 1
                elif char in ".AGT":
                    if char == "A":
                        if start is not None: raise ValueError(f"Multiple A markers: {path}")
                        start = (r, c)
                    elif char == "G":
                        if goal is not None: raise ValueError(f"Multiple G markers: {path}")
                        goal = (r, c)
                    elif char == "T":
                        track.add((r, c))
                else:
                    raise ValueError(f"Unknown map symbol {char!r} in {path}")
        if start is None or goal is None or len(track) < 8:
            raise ValueError(f"Map requires A, G, and a cyclic T track: {path}")
        return grid, start, goal, track

    def _order_track(self, track: set[Pos]) -> list[Pos]:
        def neighbors(p: Pos) -> list[Pos]:
            r, c = p
            return sorted(q for q in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)) if q in track)

        for p in track:
            if len(neighbors(p)) != 2:
                raise ValueError(f"Track must be one branch-free four-neighbor cycle; {p} has degree {len(neighbors(p))}")
        start = min(track)
        first = neighbors(start)[0]
        ordered = [start]
        prev, cur = start, first
        while cur != start:
            if cur in ordered:
                raise ValueError("Track contains a sub-cycle before returning to start.")
            ordered.append(cur)
            nxts = [q for q in neighbors(cur) if q != prev]
            if len(nxts) != 1:
                raise ValueError("Track ordering failed.")
            prev, cur = cur, nxts[0]
        if len(ordered) != len(track):
            raise ValueError("Track contains disconnected components.")
        return ordered


    def _load_route_program(self, path: Path, track: set[Pos]) -> list[Pos]:
        """Load an explicit cyclic carrier route when a sidecar is packaged.

        V7 separates the visible rail *network* from the route program. This permits
        crossings, shared junctions, nested loops and repeated junction visits while
        keeping carrier motion deterministic and reproducible. Legacy maps without a
        sidecar retain the original branch-free-cycle parser.
        """
        sidecar = path.with_suffix(".route.json")
        if not sidecar.exists():
            self.route_family = "legacy_cycle"
            self.structure_family = "unknown"
            self.route_metrics = {}
            self.vehicle_cadence = self.VEHICLE_CADENCE
            return self._order_track(track)

        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        route = [tuple(map(int, p)) for p in payload.get("route", [])]
        if len(route) < 8:
            raise ValueError(f"Route sidecar requires at least 8 route states: {sidecar}")
        for i, cell in enumerate(route):
            if cell not in track:
                raise ValueError(f"Route cell {cell} is not marked T in {path}")
            nxt = route[(i + 1) % len(route)]
            if self._manhattan(cell, nxt) != 1:
                raise ValueError(f"Route must move one four-neighbor cell per carrier move: {cell} -> {nxt}")
        self.route_family = str(payload.get("route_family", "explicit"))
        self.structure_family = str(payload.get("structure_family", "unknown"))
        self.vehicle_cadence = max(1, int(payload.get("vehicle_cadence", self.VEHICLE_CADENCE)))
        self.route_metrics = {
            key: int(payload.get(key, 0))
            for key in ("route_length", "turns", "repeated_route_nodes", "junction_cells")
        }
        return route

    def reset(self, seed: Optional[int] = None, map_path: Optional[str | Path] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_map = Path(map_path) if map_path is not None else self.map_paths[int(self.rng.integers(0, len(self.map_paths)))]
        self.grid, self.start_pos, self.goal_pos, raw_track = self._parse_map(self.current_map)
        self.track_set = set(raw_track)
        self.track_cells = self._load_route_program(self.current_map, raw_track)
        self._track_distance_fields = [self._distance_field(p) for p in self.track_cells]
        self._goal_distance_field = self._distance_field(self.goal_pos)

        self.agent_pos = self.start_pos
        self.carrying = False
        self.cargo_on_vehicle = True
        self.steps = 0
        self.collisions = 0
        self.hazard_collisions = 0
        self.invalid_actions = 0
        self.pickup_step = None
        self.last_reward = 0.0
        self.last_event = "reset"
        self.failure_reason = ""
        self.max_steps = self._configured_max_steps or max(220, min(520, 6 * (self.grid.shape[0] + self.grid.shape[1]) + 2 * len(self.track_cells)))

        self.vehicle_index = int(self.rng.integers(0, len(self.track_cells)))
        self.vehicle_direction = 1 if int(self.rng.integers(0, 2)) == 0 else -1
        self.vehicle_clock = int(self.rng.integers(0, self.vehicle_cadence))
        self.vehicle_moves = 0

        reachable = [p for p in self._reachable_cells(self.start_pos) if p not in {self.start_pos, self.goal_pos, self.vehicle_pos}]
        far = [p for p in reachable if self._manhattan(p, self.start_pos) >= max(4, self.view_size)]
        wolf_candidates = far or reachable
        if not wolf_candidates:
            raise ValueError(f"No valid wolf spawn cells in {self.current_map}")
        self.wolf_pos = wolf_candidates[int(self.rng.integers(0, len(wolf_candidates)))]
        self.wolf_prev_pos = self.wolf_pos
        self.wolf_heading = int(self.rng.integers(0, 4))
        self._choose_wolf_target()

        eta, intercept_pos, _ = self._best_intercept(self.start_pos)
        goal_after = int(self._goal_distance_field[intercept_pos]) if self._goal_distance_field is not None else 0
        self.oracle_steps = max(2, int(eta + 1 + goal_after + 1))
        return self._observation(), self._info(success=False)

    def step(self, action: int):
        action = int(action)
        if not 0 <= action < self.action_space_n:
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        terminated = truncated = success = False
        invalid = collision = pickup = delivered = waited = moved = False
        wolf_collision = False
        pickup_available_before = self._pickup_available()
        delivery_available_before = bool(self.carrying and self.agent_pos == self.goal_pos)
        nav_before = self._navigation_cost(self.agent_pos)
        wolf_before = self._manhattan(self.agent_pos, self.wolf_pos)
        risk_before = self._risk_cost(self.agent_pos)
        # Counterfactual local safety baseline: how safe could this step have been
        # under the same exogenous wolf state? This measures avoidable risk rather
        # than rewarding passive wolf motion. WAIT/current cell is included.
        safe_candidates = [self.agent_pos]
        for dr, dc in self.ACTIONS.values():
            p = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self._is_free(p) and p != self.wolf_pos:
                safe_candidates.append(p)
        best_local_risk = min(self._risk_cost(p) for p in safe_candidates)

        if action in self.ACTIONS:
            dr, dc = self.ACTIONS[action]
            nxt = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if not self._is_free(nxt):
                self.invalid_actions += 1
                self.collisions += 1
                invalid = collision = True
                self.last_event = "collision"
            elif nxt == self.wolf_pos:
                self.agent_pos = nxt
                self.collisions += 1
                self.hazard_collisions += 1
                collision = wolf_collision = True
                terminated = True
                self.failure_reason = "wolf_collision"
                self.last_event = "wolf_collision"
            else:
                self.agent_pos = nxt
                moved = True
                self.last_event = "move"
        elif action == 4:
            if self._pickup_available():
                self.carrying = True
                self.cargo_on_vehicle = False
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
                delivered = success = terminated = True
                self.last_event = "delivery"
            else:
                self.invalid_actions += 1
                invalid = True
                self.last_event = "invalid_drop"
        elif action == 6:
            waited = True
            self.last_event = "wait"

        nav_after_action = self._navigation_cost(self.agent_pos)
        navigation_delta = 0 if pickup or delivered else int(nav_before - nav_after_action)
        wolf_after_action = self._manhattan(self.agent_pos, self.wolf_pos)
        risk_after_action = self._risk_cost(self.agent_pos)
        safety_delta = float(risk_before - risk_after_action) if not wolf_collision else 0.0
        safety_regret = 0.0 if (pickup or delivered or wolf_collision) else max(0.0, float(risk_after_action - best_local_risk))
        intercept_eta_before = int(nav_before) if (not self.carrying and self.cargo_on_vehicle) else 0
        well_timed_wait = bool(waited and (not self.carrying) and intercept_eta_before <= max(2, self.vehicle_cadence + 1))

        if not terminated:
            self._advance_vehicle()
            self._advance_wolf()
            if self.wolf_pos == self.agent_pos:
                terminated = True
                wolf_collision = True
                self.collisions += 1
                self.hazard_collisions += 1
                self.failure_reason = "wolf_collision"
                self.last_event = "wolf_collision"

        wolf_distance = self._manhattan(self.agent_pos, self.wolf_pos)
        transition = {
            "action": action,
            "moved": moved,
            "waited": waited,
            "invalid": invalid,
            "collision": collision,
            "wolf_collision": wolf_collision,
            "pickup": pickup,
            "delivered": delivered,
            "navigation_delta": navigation_delta,
            "safety_delta": safety_delta,
            "safety_regret": safety_regret,
            "wolf_distance": wolf_distance,
            "well_timed_wait": well_timed_wait,
            "pickup_available_before": pickup_available_before,
            "delivery_available_before": delivery_available_before,
            "missed_pickup": bool(pickup_available_before and action != 4),
            "missed_delivery": bool(delivery_available_before and action != 5),
            "step": self.steps,
        }
        reward = float(self.reward_function.compute(transition))
        self.last_reward = reward

        if self.steps >= self.max_steps and not terminated:
            truncated = True
            self.failure_reason = "timeout"
            self.last_event = "timeout"
        return self._observation(), reward, terminated, truncated, self._info(success=success)

    def _advance_vehicle(self) -> None:
        self.vehicle_clock += 1
        if self.vehicle_clock >= self.vehicle_cadence:
            self.vehicle_clock = 0
            self.vehicle_index = (self.vehicle_index + self.vehicle_direction) % len(self.track_cells)
            self.vehicle_moves += 1

    def _choose_wolf_target(self) -> None:
        reachable = self._reachable_cells(self.wolf_pos)
        far = [p for p in reachable if self._manhattan(p, self.wolf_pos) >= max(5, max(self.grid.shape) // 2)]
        choices = far or reachable
        self.wolf_target = choices[int(self.rng.integers(0, len(choices)))]
        self._wolf_target_field = self._distance_field(self.wolf_target)
        self.wolf_target_ttl = int(self.rng.integers(8, 18))

    def _advance_wolf(self) -> None:
        self.wolf_prev_pos = self.wolf_pos
        # Random-waypoint roaming covers the complete map more reliably than an
        # unconstrained random walk while remaining non-adversarial and seed-reproducible.
        if self.wolf_target_ttl <= 0 or self.wolf_pos == self.wolf_target or self._wolf_target_field is None:
            self._choose_wolf_target()
        candidates: list[tuple[int, Pos]] = []
        for a, (dr, dc) in self.ACTIONS.items():
            p = (self.wolf_pos[0] + dr, self.wolf_pos[1] + dc)
            if self._is_free(p):
                candidates.append((a, p))
        if not candidates:
            return
        self.wolf_target_ttl -= 1
        u = float(self.rng.random())
        if u < 0.10:
            return
        best_distance = min(int(self._wolf_target_field[p]) for _, p in candidates)
        toward = [(a,p) for a,p in candidates if int(self._wolf_target_field[p]) == best_distance]
        if toward and u < 0.82:
            action, nxt = toward[int(self.rng.integers(0, len(toward)))]
        else:
            action, nxt = candidates[int(self.rng.integers(0, len(candidates)))]
        self.wolf_heading = int(action)
        self.wolf_pos = nxt

    def _future_vehicle_index(self, t: int) -> int:
        moves = (self.vehicle_clock + max(0, int(t))) // self.vehicle_cadence
        return (self.vehicle_index + self.vehicle_direction * moves) % len(self.track_cells)

    def _best_intercept(self, pos: Pos) -> tuple[int, Pos, int]:
        horizon = min(self.max_steps, max(24, len(self.track_cells) * self.vehicle_cadence + 8))
        best: tuple[int, Pos, int] | None = None
        for t in range(horizon + 1):
            idx = self._future_vehicle_index(t)
            dist = int(self._track_distance_fields[idx][pos])
            if dist <= t:
                return t, self.track_cells[idx], idx
            candidate = (max(dist, t), self.track_cells[idx], idx)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        return best

    def _navigation_cost(self, pos: Pos) -> int:
        if self.carrying:
            assert self._goal_distance_field is not None
            return int(self._goal_distance_field[pos])
        if not self.cargo_on_vehicle:
            return 0
        eta, _, _ = self._best_intercept(pos)
        return int(eta)

    def _route_waypoint(self) -> Pos:
        if self.carrying:
            target = self.goal_pos
            field = self._goal_distance_field
        else:
            _, target, idx = self._best_intercept(self.agent_pos)
            field = self._track_distance_fields[idx]
        if field is None or self.agent_pos == target:
            return self.agent_pos
        current = int(field[self.agent_pos])
        options: list[Pos] = []
        for dr, dc in self.ACTIONS.values():
            p = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            if self._is_free(p) and int(field[p]) < current:
                options.append(p)
        if not options:
            return self.agent_pos
        # Prefer a locally safer equally-short next step when the wolf is nearby.
        best_distance = min(int(field[p]) for p in options)
        shortest = [p for p in options if int(field[p]) == best_distance]
        return max(shortest, key=lambda p: self._manhattan(p, self.wolf_pos))

    def _predicted_wolf_pos(self) -> Pos:
        dr, dc = self.ACTIONS.get(int(self.wolf_heading), (0, 0))
        nxt = (self.wolf_pos[0] + dr, self.wolf_pos[1] + dc)
        return nxt if self._is_free(nxt) else self.wolf_pos

    def _risk_cost(self, pos: Pos) -> float:
        """Low-bandwidth local hazard potential used for symmetric safety shaping."""
        current = self._manhattan(pos, self.wolf_pos)
        predicted = self._manhattan(pos, self._predicted_wolf_pos())
        return float(max(0, 4-current) + 0.65 * max(0, 3-predicted))

    def _intercept_slack(self, pos: Pos) -> int:
        if self.carrying or not self.cargo_on_vehicle:
            return 0
        eta, _, idx = self._best_intercept(pos)
        distance = int(self._track_distance_fields[idx][pos])
        return max(0, int(eta - distance))

    def _turn_proximity(self, horizon: int = 8) -> float:
        """Return inverse route-move distance to the carrier's next direction change."""
        if len(self.track_cells) < 3:
            return 0.0
        direction = self.vehicle_direction
        idx = self.vehicle_index
        def delta(i: int) -> tuple[int, int]:
            j = (i + direction) % len(self.track_cells)
            a, b = self.track_cells[i], self.track_cells[j]
            return b[0]-a[0], b[1]-a[1]
        base = delta(idx)
        for k in range(1, horizon + 1):
            probe = (idx + direction * k) % len(self.track_cells)
            if delta(probe) != base:
                return 1.0 / float(k)
        return 0.0

    def _pickup_available(self) -> bool:
        return bool((not self.carrying) and self.cargo_on_vehicle and self.agent_pos == self.vehicle_pos)

    def valid_action_mask(self, mode: str = "task") -> np.ndarray:
        if mode not in {"task", "valid"}:
            raise ValueError("mask mode must be 'task' or 'valid'.")
        mask = np.zeros(self.action_space_n, dtype=bool)
        pickup = self._pickup_available()
        delivery = bool(self.carrying and self.agent_pos == self.goal_pos)
        if mode == "task" and pickup:
            mask[4] = True
            return mask
        if mode == "task" and delivery:
            mask[5] = True
            return mask
        for a, (dr, dc) in self.ACTIONS.items():
            p = (self.agent_pos[0] + dr, self.agent_pos[1] + dc)
            # The wolf's currently occupied cell is physically unsafe and is masked.
            if self._is_free(p) and p != self.wolf_pos:
                mask[a] = True
        if pickup: mask[4] = True
        if delivery: mask[5] = True
        mask[6] = True  # WAIT is needed for moving-target interception timing.
        return mask

    def _distance_field(self, goal: Pos) -> np.ndarray:
        inf = self.grid.size * 4
        field = np.full(self.grid.shape, inf, dtype=np.int32)
        field[goal] = 0
        q = deque([goal])
        while q:
            r, c = q.popleft()
            nd = int(field[r,c]) + 1
            for dr, dc in self.ACTIONS.values():
                p = (r+dr, c+dc)
                if self._is_free(p) and nd < field[p]:
                    field[p] = nd
                    q.append(p)
        return field

    def _reachable_cells(self, start: Pos) -> list[Pos]:
        q = deque([start]); seen = {start}
        while q:
            r,c = q.popleft()
            for dr,dc in self.ACTIONS.values():
                p=(r+dr,c+dc)
                if self._is_free(p) and p not in seen:
                    seen.add(p); q.append(p)
        return list(seen)

    def _is_free(self, pos: Pos) -> bool:
        r,c = pos
        return 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1] and self.grid[pos] == 0

    @staticmethod
    def _manhattan(a: Pos, b: Pos) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _observation(self) -> np.ndarray:
        v = self.view_size
        radius = v // 2
        obs = np.zeros(self.observation_shape, dtype=np.float32)
        ar, ac = self.agent_pos
        for vr in range(v):
            for vc in range(v):
                gr, gc = ar + vr - radius, ac + vc - radius
                if not (0 <= gr < self.grid.shape[0] and 0 <= gc < self.grid.shape[1]):
                    obs[0,vr,vc] = 1.0
                    continue
                p=(gr,gc)
                if self.grid[p] == 1: obs[0,vr,vc] = 1.0
                if p in self.track_set: obs[1,vr,vc] = 1.0
                if self.cargo_on_vehicle and p == self.vehicle_pos: obs[2,vr,vc] = 1.0
                if p == self.goal_pos: obs[3,vr,vc] = 1.0
                if p == self.wolf_pos: obs[4,vr,vc] = 1.0
        obs[5,radius,radius] = 1.0
        obs[6,:,:] = 1.0 if self.carrying else 0.0
        waypoint = self._route_waypoint()
        scale = float(max(1, self.view_size))
        obs[7,:,:] = np.clip((waypoint[0]-ar)/scale, -1.0, 1.0)
        obs[8,:,:] = np.clip((waypoint[1]-ac)/scale, -1.0, 1.0)
        nxt_idx = (self.vehicle_index + self.vehicle_direction) % len(self.track_cells)
        vr, vc = self.vehicle_pos; nr, nc = self.track_cells[nxt_idx]
        obs[9,:,:] = float(nr-vr)
        obs[10,:,:] = float(nc-vc)
        row_scale = float(max(1, self.grid.shape[0]-1)); col_scale = float(max(1, self.grid.shape[1]-1))
        obs[11,:,:] = np.clip((self.wolf_pos[0]-ar)/row_scale, -1.0, 1.0)
        obs[12,:,:] = np.clip((self.wolf_pos[1]-ac)/col_scale, -1.0, 1.0)
        wolf_dist = self._manhattan(self.agent_pos, self.wolf_pos)
        obs[13,:,:] = 1.0 / (1.0 + float(wolf_dist))
        obs[14,:,:] = 1.0 if self._pickup_available() else 0.0
        obs[15,:,:] = 1.0 if (self.carrying and self.agent_pos == self.goal_pos) else 0.0
        eta = 0 if self.carrying else self._best_intercept(self.agent_pos)[0]
        obs[16,:,:] = min(1.0, float(eta) / 24.0)
        obs[17,:,:] = float(self.vehicle_clock) / float(max(1, self.vehicle_cadence-1))

        future_idx = self._future_vehicle_index(max(4, 2 * self.vehicle_cadence))
        fr, fc = self.track_cells[future_idx]
        obs[18,:,:] = np.clip((fr-ar)/row_scale, -1.0, 1.0)
        obs[19,:,:] = np.clip((fc-ac)/col_scale, -1.0, 1.0)
        obs[20,:,:] = float(self.wolf_pos[0] - self.wolf_prev_pos[0])
        obs[21,:,:] = float(self.wolf_pos[1] - self.wolf_prev_pos[1])
        obs[22,:,:] = min(1.0, float(self._intercept_slack(self.agent_pos)) / 8.0)
        obs[23,:,:] = self._turn_proximity()
        return obs

    def _info(self, success: bool) -> Dict[str, object]:
        efficiency = min(1.0, self.oracle_steps / self.steps) if success and self.steps > 0 else 0.0
        return {
            "task_id": self.TASK_ID,
            "task_code": self.TASK_CODE,
            "success": bool(success),
            "carrying": self.carrying,
            "steps": self.steps,
            "map": self.current_map.name if self.current_map else None,
            "map_shape": tuple(int(v) for v in self.grid.shape),
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "vehicle_pos": self.vehicle_pos,
            "vehicle_direction": self.vehicle_direction,
            "vehicle_clock": self.vehicle_clock,
            "vehicle_cadence": self.vehicle_cadence,
            "route_family": self.route_family,
            "structure_family": self.structure_family,
            "route_metrics": dict(self.route_metrics),
            "wolf_pos": self.wolf_pos,
            "failure_reason": self.failure_reason,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "reward_module": self.reward_module,
            "collisions": self.collisions,
            "hazard_collisions": self.hazard_collisions,
            "invalid_actions": self.invalid_actions,
            "pickup_step": self.pickup_step,
            "pickups": 1 if self.pickup_step is not None else 0,
            "items_total": 1,
            "items_delivered": 1 if success else 0,
            "completion_rate": 1.0 if success else 0.0,
            "oracle_steps": self.oracle_steps,
            "path_efficiency": float(efficiency),
            "last_reward": self.last_reward,
            "intercept_eta": 0 if self.carrying else self._best_intercept(self.agent_pos)[0],
            "route_waypoint": self._route_waypoint(),
            "behavior_state": (self.agent_pos, bool(self.carrying), int(self.vehicle_index), self.wolf_pos),
        }

    def robot_status(self) -> dict[str, object]:
        return {
            "Robot": "Courier",
            "Position": f"{self.agent_pos[0]}, {self.agent_pos[1]}",
            "Cargo": "Carrying" if self.carrying else ("On carrier" if self.cargo_on_vehicle else "Transferred"),
            "Carrier": f"{self.vehicle_pos[0]}, {self.vehicle_pos[1]}",
            "Intercept ETA": 0 if self.carrying else self._best_intercept(self.agent_pos)[0],
            "Track": self.route_family,
            "Wolf": f"{self.wolf_pos[0]}, {self.wolf_pos[1]}",
            "Last event": self.last_event,
            "Failure": self.failure_reason or "—",
        }

    def render_entities(self) -> dict[str, list[tuple[int, Pos]]]:
        return {
            "objects": [] if (self.carrying or not self.cargo_on_vehicle) else [(0, self.vehicle_pos)],
            "goals": [(0, self.goal_pos)],
        }

    def render_track(self) -> list[Pos]:
        return list(self.track_cells)

    def render_hazards(self) -> list[tuple[str, Pos]]:
        return [("wolf", self.wolf_pos)]

    def render_vehicle(self) -> Pos:
        return self.vehicle_pos

    def render_ascii(self) -> str:
        chars = np.full(self.grid.shape, ".", dtype="<U1")
        chars[self.grid == 1] = "#"
        for p in self.track_cells: chars[p] = "T"
        chars[self.goal_pos] = "G"
        if self.cargo_on_vehicle: chars[self.vehicle_pos] = "V"
        chars[self.wolf_pos] = "W"
        chars[self.agent_pos] = "f" if self.carrying else "F"
        return "\n".join("".join(row) for row in chars)
