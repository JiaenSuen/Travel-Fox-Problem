from __future__ import annotations

from collections import deque
from pathlib import Path
import json
import random

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tfp" / "tasks" / "moving_cargo_evasion"

# Four spatial scales. Each scale has three train variants per route family and one
# held-out test variant per family: 72 train + 24 test maps.
SPECS = {12: (18, 6), 16: (18, 6), 20: (18, 6), 24: (18, 6)}
ROUTE_FAMILIES = ("ring", "serpentine", "figure8", "clover", "switchyard", "nested")
STRUCTURE_FAMILIES = ("open", "slalom", "rooms", "blocks", "corridors", "islands")
Pos = tuple[int, int]


def _append_segment(route: list[Pos], target: Pos, horizontal_first: bool) -> None:
    if not route:
        route.append(target)
        return
    r, c = route[-1]
    tr, tc = target
    axes = ("h", "v") if horizontal_first else ("v", "h")
    for axis in axes:
        if axis == "h":
            while c != tc:
                c += 1 if tc > c else -1
                route.append((r, c))
        else:
            while r != tr:
                r += 1 if tr > r else -1
                route.append((r, c))


def _cycle_from_waypoints(waypoints: list[Pos], alternate: bool = True) -> list[Pos]:
    route = [waypoints[0]]
    for i, target in enumerate(waypoints[1:] + [waypoints[0]]):
        _append_segment(route, target, horizontal_first=(i % 2 == 0 or not alternate))
    # Last element repeats the first position. The runtime route wraps implicitly.
    if len(route) > 1 and route[-1] == route[0]:
        route.pop()
    return route


def build_route(size: int, family: str, variant: int) -> list[Pos]:
    lo, hi = 2, size - 3
    mid = size // 2
    q1 = max(lo + 1, size // 3)
    q3 = min(hi - 1, size - 1 - size // 3)
    # Small deterministic offsets make maps within one family geometrically distinct.
    d = (variant % 3) - 1

    if family == "ring":
        return _cycle_from_waypoints([(lo, lo), (lo, hi), (hi, hi), (hi, lo)], alternate=False)

    if family == "serpentine":
        rows = list(range(lo, hi + 1, 2))
        if rows[-1] != hi:
            rows.append(hi)
        points: list[Pos] = [(rows[0], lo)]
        right = True
        for r in rows:
            points.append((r, hi if right else lo + 1))
            right = not right
        points.append((hi, lo))
        return _cycle_from_waypoints(points, alternate=False)

    if family == "figure8":
        j = (mid, mid + d)
        points = [
            j, (lo, j[1]), (lo, lo), (mid, lo), j,
            (mid, hi), (hi, hi), (hi, j[1]),
        ]
        return _cycle_from_waypoints(points, alternate=True)

    if family == "clover":
        j = (mid, mid)
        # Four lobes share the central junction. Re-visiting the junction is an
        # intentional part of the periodic route rather than an ambiguity.
        points = [
            j, (q1, mid), (q1, q1), (mid, q1), j,
            (mid, q3), (q1, q3), (q1, mid), j,
            (q3, mid), (q3, q3), (mid, q3), j,
            (mid, q1), (q3, q1), (q3, mid),
        ]
        return _cycle_from_waypoints(points, alternate=True)

    if family == "switchyard":
        points = [
            (lo, lo), (lo, hi), (mid, mid + d), (hi, hi),
            (hi, lo), (mid, mid - d), (q1, q3), (q3, q1),
        ]
        return _cycle_from_waypoints(points, alternate=True)

    if family == "nested":
        inner_lo, inner_hi = q1, q3
        points = [
            (lo, lo), (lo, hi), (hi, hi), (hi, lo),
            (inner_lo, lo), (inner_lo, inner_lo), (inner_lo, inner_hi),
            (inner_hi, inner_hi), (inner_hi, inner_lo), (inner_lo, inner_lo),
        ]
        return _cycle_from_waypoints(points, alternate=True)

    raise ValueError(f"Unknown route family: {family}")


def reachable(grid: list[list[str]], start: Pos) -> set[Pos]:
    h, w = len(grid), len(grid[0])
    q = deque([start]); seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            p = (r+dr, c+dc)
            if 0 <= p[0] < h and 0 <= p[1] < w and grid[p[0]][p[1]] != "#" and p not in seen:
                seen.add(p); q.append(p)
    return seen


def add_obstacles(grid: list[list[str]], track: set[Pos], family: str, rng: random.Random) -> None:
    h, w = len(grid), len(grid[0])
    protected = set(track)
    # Keep one-cell access around most track cells. Complex rail networks are useful
    # only if the courier can approach them from more than one side.
    for r, c in list(track):
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            protected.add((r+dr, c+dc))

    def wall(r: int, c: int) -> None:
        if 1 <= r < h-1 and 1 <= c < w-1 and (r,c) not in protected:
            grid[r][c] = "#"

    if family == "open":
        for _ in range(max(2, h // 5)):
            wall(rng.randint(2, h-3), rng.randint(2, w-3))
    elif family == "slalom":
        for c in range(3, w-3, 4):
            gap = rng.randint(2, h-3)
            for r in range(2, h-2):
                if abs(r-gap) > 1:
                    wall(r,c)
    elif family == "rooms":
        mr, mc = h // 2, w // 2
        gaps_r = {rng.randint(2, h-3), rng.randint(2, h-3)}
        gaps_c = {rng.randint(2, w-3), rng.randint(2, w-3)}
        for r in range(2, h-2):
            if r not in gaps_r: wall(r, mc)
        for c in range(2, w-2):
            if c not in gaps_c: wall(mr, c)
    elif family == "blocks":
        for _ in range(max(3, h // 4)):
            r, c = rng.randint(2, h-4), rng.randint(2, w-4)
            for dr, dc in ((0,0),(0,1),(1,0),(1,1)):
                wall(r+dr,c+dc)
    elif family == "corridors":
        for r in range(3, h-3, 4):
            gap = rng.randint(2, w-3)
            for c in range(2, w-2):
                if abs(c-gap) > 1: wall(r,c)
    elif family == "islands":
        for _ in range(max(4, h // 3)):
            r, c = rng.randint(2, h-3), rng.randint(2, w-3)
            wall(r,c)
            if rng.random() < 0.55: wall(r, min(w-3,c+1))
            if rng.random() < 0.35: wall(min(h-3,r+1), c)


def choose_markers(grid: list[list[str]], track: set[Pos], rng: random.Random) -> tuple[Pos, Pos]:
    h, w = len(grid), len(grid[0])
    candidates = [(r,c) for r in range(1,h-1) for c in range(1,w-1) if grid[r][c] == "." and (r,c) not in track]
    rng.shuffle(candidates)
    if not candidates:
        raise RuntimeError("No marker candidates")
    for start in candidates[: min(40, len(candidates))]:
        seen = reachable(grid, start)
        goals = [p for p in candidates if p in seen and p != start]
        if not goals: continue
        goals.sort(key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]), reverse=True)
        return start, goals[rng.randrange(min(8, len(goals)))]
    raise RuntimeError("No reachable marker pair")


def route_metrics(route: list[Pos]) -> dict[str, int]:
    turns = 0
    repeated = len(route) - len(set(route))
    for i in range(len(route)):
        a, b, c = route[i-1], route[i], route[(i+1) % len(route)]
        d1 = (b[0]-a[0], b[1]-a[1]); d2 = (c[0]-b[0], c[1]-b[1])
        turns += int(d1 != d2)
    network = set(route)
    junctions = 0
    for r,c in network:
        degree = sum((r+dr,c+dc) in network for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)))
        junctions += int(degree > 2)
    return {"route_length": len(route), "turns": turns, "repeated_route_nodes": repeated, "junction_cells": junctions}


def build_map(size: int, layout_index: int, split: str) -> tuple[str, dict[str, object]]:
    seed = size * 10000 + layout_index * 101 + (0 if split == "train" else 900001)
    rng = random.Random(seed)
    route_family = ROUTE_FAMILIES[(layout_index - 1) % len(ROUTE_FAMILIES)]
    structure_family = STRUCTURE_FAMILIES[((layout_index - 1) // len(ROUTE_FAMILIES) + layout_index) % len(STRUCTURE_FAMILIES)]
    route = build_route(size, route_family, layout_index)
    track = set(route)
    h = w = size

    for _attempt in range(80):
        grid = [["." for _ in range(w)] for _ in range(h)]
        for r in range(h): grid[r][0] = grid[r][w-1] = "#"
        for c in range(w): grid[0][c] = grid[h-1][c] = "#"
        add_obstacles(grid, track, structure_family, rng)
        for r,c in track:
            if not (0 < r < h-1 and 0 < c < w-1):
                raise RuntimeError(f"Route leaves map interior: {(r,c)} size={size}")
            grid[r][c] = "T"
        try:
            start, goal = choose_markers(grid, track, rng)
        except RuntimeError:
            continue
        seen = reachable(grid, start)
        if goal not in seen or not track.issubset(seen):
            continue
        grid[start[0]][start[1]] = "A"
        grid[goal[0]][goal[1]] = "G"
        metadata: dict[str, object] = {
            "version": 2,
            "route_family": route_family,
            "structure_family": structure_family,
            "route": [[r,c] for r,c in route],
            "vehicle_cadence": 2 if size <= 16 else 3,
            **route_metrics(route),
        }
        return "\n".join("".join(row) for row in grid) + "\n", metadata
    raise RuntimeError(f"Could not generate solvable {split} map size={size} index={layout_index}")


def main() -> None:
    for split in ("train", "test"):
        out = TASK_ROOT / "maps" / split
        out.mkdir(parents=True, exist_ok=True)
        for p in out.glob("*.txt"): p.unlink()
        for p in out.glob("*.route.json"): p.unlink()

    for size, (n_train, n_test) in SPECS.items():
        for split, count in (("train", n_train), ("test", n_test)):
            out = TASK_ROOT / "maps" / split
            for idx in range(1, count + 1):
                route_family = ROUTE_FAMILIES[(idx - 1) % len(ROUTE_FAMILIES)]
                text, metadata = build_map(size, idx, split)
                stem = f"{split}_{route_family}_{idx:02d}_{size}x{size}"
                (out / f"{stem}.txt").write_text(text, encoding="utf-8")
                (out / f"{stem}.route.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Generated moving-cargo maps under {TASK_ROOT / 'maps'}")


if __name__ == "__main__":
    main()
