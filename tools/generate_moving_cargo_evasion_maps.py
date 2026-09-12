from __future__ import annotations

from collections import deque
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tfp" / "tasks" / "moving_cargo_evasion"

SPECS = {
    12: (15, 5),
    16: (15, 5),
    20: (15, 5),
}
FAMILIES = ("open", "bars", "cross", "islands", "corridor")


def rect_track(h: int, w: int, inset_r: int, inset_c: int) -> set[tuple[int, int]]:
    top, left = inset_r, inset_c
    bottom, right = h - 1 - inset_r, w - 1 - inset_c
    track: set[tuple[int, int]] = set()
    for c in range(left, right + 1):
        track.add((top, c)); track.add((bottom, c))
    for r in range(top, bottom + 1):
        track.add((r, left)); track.add((r, right))
    return track


def reachable(grid: list[list[str]], start: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = len(grid), len(grid[0])
    q = deque([start]); seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            p = (r+dr, c+dc)
            if 0 <= p[0] < h and 0 <= p[1] < w and grid[p[0]][p[1]] != "#" and p not in seen:
                seen.add(p); q.append(p)
    return seen


def add_obstacles(grid: list[list[str]], track: set[tuple[int,int]], family: str, rng: random.Random) -> None:
    h, w = len(grid), len(grid[0])
    protected = set(track)
    # Keep a broad central cross open so every layout preserves multiple shortcut routes.
    cr, cc = h // 2, w // 2
    protected |= {(cr, c) for c in range(1, w-1)} | {(r, cc) for r in range(1, h-1)}

    def wall(r: int, c: int) -> None:
        if 1 <= r < h-1 and 1 <= c < w-1 and (r,c) not in protected:
            grid[r][c] = "#"

    if family == "open":
        for _ in range(max(1, h // 6)):
            r = rng.randint(2, h-3); c = rng.randint(2, w-3)
            wall(r,c)
    elif family == "bars":
        for c in range(3, w-3, 4):
            gap = rng.randint(2, h-3)
            for r in range(2, h-2):
                if abs(r-gap) > 1: wall(r,c)
    elif family == "cross":
        r0 = max(2, cr-2); c0 = max(2, cc-2)
        for c in range(2, w-2):
            if abs(c-cc) > 2 and c % 5 != 0: wall(r0,c)
        for r in range(2, h-2):
            if abs(r-cr) > 2 and r % 5 != 0: wall(r,c0)
    elif family == "islands":
        for _ in range(max(2, h // 4)):
            r = rng.randint(2, h-4); c = rng.randint(2, w-4)
            for dr, dc in ((0,0),(0,1),(1,0),(1,1)):
                wall(r+dr,c+dc)
    elif family == "corridor":
        for r in range(3, h-3, 4):
            gap = rng.randint(2, w-3)
            for c in range(2, w-2):
                if abs(c-gap) > 1: wall(r,c)


def choose_markers(grid: list[list[str]], track: set[tuple[int,int]], rng: random.Random) -> tuple[tuple[int,int], tuple[int,int]]:
    h, w = len(grid), len(grid[0])
    candidates = [(r,c) for r in range(1,h-1) for c in range(1,w-1) if grid[r][c] == "." and (r,c) not in track]
    center = (h//2, w//2)
    candidates.sort(key=lambda p: abs(p[0]-center[0]) + abs(p[1]-center[1]))
    start = candidates[min(len(candidates)-1, rng.randint(0, min(5, len(candidates)-1)))]
    reachable_from_start = reachable(grid, start)
    goals = [p for p in candidates if p in reachable_from_start and p != start]
    goals.sort(key=lambda p: abs(p[0]-start[0]) + abs(p[1]-start[1]), reverse=True)
    goal = goals[min(len(goals)-1, rng.randint(0, min(5, len(goals)-1)))]
    return start, goal


def build_map(size: int, layout_index: int, split: str) -> str:
    seed = size * 10000 + layout_index * 101 + (0 if split == "train" else 900001)
    rng = random.Random(seed)
    h = w = size
    family = FAMILIES[(layout_index - 1) % len(FAMILIES)]
    # The vehicle loop spans most of the map but varies its offset across layouts.
    inset_r = 2 + ((layout_index + size) % 2)
    inset_c = 2 + ((layout_index * 3 + size) % 2)
    track = rect_track(h, w, inset_r, inset_c)

    for _attempt in range(40):
        grid = [["." for _ in range(w)] for _ in range(h)]
        for r in range(h): grid[r][0] = grid[r][w-1] = "#"
        for c in range(w): grid[0][c] = grid[h-1][c] = "#"
        add_obstacles(grid, track, family, rng)
        for r,c in track: grid[r][c] = "T"
        # marker choice only uses ordinary floor cells
        try:
            start, goal = choose_markers(grid, track, rng)
        except Exception:
            continue
        seen = reachable(grid, start)
        if goal not in seen or not track.issubset(seen):
            continue
        grid[start[0]][start[1]] = "A"
        grid[goal[0]][goal[1]] = "G"
        return "\n".join("".join(row) for row in grid) + "\n"
    raise RuntimeError(f"Could not generate solvable {split} map size={size} index={layout_index}")


def main() -> None:
    for split in ("train", "test"):
        out = TASK_ROOT / "maps" / split
        out.mkdir(parents=True, exist_ok=True)
        for p in out.glob("*.txt"): p.unlink()
    for size, (n_train, n_test) in SPECS.items():
        for split, count in (("train", n_train), ("test", n_test)):
            out = TASK_ROOT / "maps" / split
            for idx in range(1, count + 1):
                family = FAMILIES[(idx - 1) % len(FAMILIES)]
                path = out / f"{split}_{family}_{idx:02d}_{size}x{size}.txt"
                path.write_text(build_map(size, idx, split), encoding="utf-8")
    print(f"Generated maps under {TASK_ROOT / 'maps'}")


if __name__ == "__main__":
    main()
