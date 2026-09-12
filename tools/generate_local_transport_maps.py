from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "tfp" / "tasks" / "local_transport" / "maps" / "train"
TEST_DIR = ROOT / "tfp" / "tasks" / "local_transport" / "maps" / "test"
FAMILIES = ("aisle", "partition", "blocks", "mixed", "rooms", "zigzag")


def reachable(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not grid[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return False


def shortest_distance(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    q = deque([(start, 0)])
    seen = {start}
    while q:
        pos, d = q.popleft()
        if pos == goal:
            return d
        r, c = pos
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            nr, nc = nxt
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and not grid[nr, nc] and nxt not in seen:
                seen.add(nxt)
                q.append((nxt, d + 1))
    return grid.size


def carve_guaranteed_route(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], rng: np.random.Generator) -> None:
    """Carve one non-strict Manhattan route after obstacle generation."""
    r, c = start
    gr, gc = goal
    grid[r, c] = 0
    safety = grid.size * 3
    while (r, c) != (gr, gc) and safety > 0:
        safety -= 1
        choices: list[tuple[int, int]] = []
        if r != gr:
            choices.append((r + (1 if gr > r else -1), c))
        if c != gc:
            choices.append((r, c + (1 if gc > c else -1)))
        if rng.random() < 0.18:
            lateral = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            rng.shuffle(lateral)
            for nr, nc in lateral:
                if 1 <= nr < grid.shape[0] - 1 and 1 <= nc < grid.shape[1] - 1:
                    choices.append((nr, nc))
                    break
        nr, nc = choices[int(rng.integers(0, len(choices)))]
        r, c = nr, nc
        grid[r, c] = 0


def _add_aisles(grid: np.ndarray, rng: np.random.Generator, variant: int) -> None:
    """Parallel shelf rows or columns with irregular cross-aisle openings."""
    size = grid.shape[0]
    vertical = variant % 2 == 0
    spacing = 3 if size <= 10 else 4
    positions = list(range(2, size - 2, spacing))
    for idx, pos in enumerate(positions):
        if vertical:
            grid[2:-2, pos] = 1
            door_rows = {2 + ((idx * 3 + variant) % max(1, size - 4)), size // 2}
            for r in door_rows:
                if 1 <= r < size - 1:
                    grid[r, pos] = 0
        else:
            grid[pos, 2:-2] = 1
            door_cols = {2 + ((idx * 3 + variant) % max(1, size - 4)), size // 2}
            for c in door_cols:
                if 1 <= c < size - 1:
                    grid[pos, c] = 0


def _add_partitions(grid: np.ndarray, rng: np.random.Generator, variant: int) -> None:
    """Long room partitions with one or two doors per wall."""
    size = grid.shape[0]
    count = {10: 2, 15: 3, 20: 4}[size]
    for i in range(count):
        horizontal = (i + variant) % 2 == 0
        if horizontal:
            r = 2 + ((i * 4 + variant) % max(1, size - 4))
            grid[r, 1:-1] = 1
            doors = [2 + ((variant + i * 5) % max(1, size - 4))]
            if size >= 15:
                doors.append(size - 3 - ((variant + i * 2) % max(1, size // 3)))
            for c in doors:
                if 1 <= c < size - 1:
                    grid[r, c] = 0
        else:
            c = 2 + ((i * 4 + variant) % max(1, size - 4))
            grid[1:-1, c] = 1
            doors = [2 + ((variant + i * 5) % max(1, size - 4))]
            if size >= 15:
                doors.append(size - 3 - ((variant + i * 2) % max(1, size // 3)))
            for r in doors:
                if 1 <= r < size - 1:
                    grid[r, c] = 0


def _add_blocks(grid: np.ndarray, rng: np.random.Generator, variant: int) -> None:
    """Compact shelf islands that force local left/right routing choices."""
    size = grid.shape[0]
    block_count = {10: 4, 15: 7, 20: 11}[size]
    for i in range(block_count):
        h = int(rng.integers(1, 3 if size <= 10 else 4))
        w = int(rng.integers(1, 4 if size <= 15 else 5))
        r = int(rng.integers(1, max(2, size - 1 - h)))
        c = int(rng.integers(1, max(2, size - 1 - w)))
        grid[r : r + h, c : c + w] = 1
        if i % 3 == 0 and r + h < size - 1:
            grid[r + h, c] = 0


def _add_mixed(grid: np.ndarray, rng: np.random.Generator, variant: int) -> None:
    """Combination of partial partitions and blocks for heterogeneous layouts."""
    size = grid.shape[0]
    segment_count = {10: 2, 15: 4, 20: 6}[size]
    for i in range(segment_count):
        horizontal = (i + variant) % 2 == 0
        if horizontal:
            r = int(rng.integers(2, size - 2))
            c0 = int(rng.integers(1, max(2, size // 3)))
            c1 = int(rng.integers(max(c0 + 2, size // 2), size - 1))
            grid[r, c0:c1] = 1
            grid[r, int(rng.integers(c0, c1))] = 0
        else:
            c = int(rng.integers(2, size - 2))
            r0 = int(rng.integers(1, max(2, size // 3)))
            r1 = int(rng.integers(max(r0 + 2, size // 2), size - 1))
            grid[r0:r1, c] = 1
            grid[int(rng.integers(r0, r1)), c] = 0
    for _ in range({10: 2, 15: 4, 20: 6}[size]):
        h = int(rng.integers(1, 3))
        w = int(rng.integers(1, 4))
        r = int(rng.integers(1, size - 1 - h))
        c = int(rng.integers(1, size - 1 - w))
        grid[r : r + h, c : c + w] = 1



def _add_rooms(grid: np.ndarray, rng: np.random.Generator, variant: int) -> None:
    """Room-like compartments with offset doorways and short interior walls."""
    size = grid.shape[0]
    for r in range(4, size - 2, 5):
        grid[r, 1:-1] = 1
        door = 2 + ((variant + r * 2) % max(1, size - 4))
        grid[r, door] = 0
        if size >= 15 and door + 2 < size - 1:
            grid[r, door + 2] = 0
    for c in range(5, size - 2, 6):
        grid[1:-1, c] = 1
        door = 2 + ((variant * 3 + c) % max(1, size - 4))
        grid[door, c] = 0


def _add_zigzag(grid: np.ndarray, rng: np.random.Generator, variant: int) -> None:
    """Alternating barrier segments that reward local detours without a maze solver."""
    size = grid.shape[0]
    step = 3 if size <= 10 else 4
    flip = variant % 2
    for i, r in enumerate(range(2, size - 2, step)):
        if (i + flip) % 2 == 0:
            grid[r, 1:size-4] = 1
            grid[r, max(2, size-5)] = 0
        else:
            grid[r, 4:size-1] = 1
            grid[r, min(size-3, 4)] = 0

def make_map(size: int, seed: int, variant: int, family: str) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    if family not in FAMILIES:
        raise ValueError(f"Unknown topology family: {family}")
    rng = np.random.default_rng(seed)
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1

    corners = [
        ((1, 1), (size - 2, size - 2)),
        ((1, size - 2), (size - 2, 1)),
        ((size - 2, 1), (1, size - 2)),
        ((size - 2, size - 2), (1, 1)),
    ]
    start, goal = corners[variant % len(corners)]

    if family == "aisle":
        _add_aisles(grid, rng, variant)
    elif family == "partition":
        _add_partitions(grid, rng, variant)
    elif family == "blocks":
        _add_blocks(grid, rng, variant)
    elif family == "mixed":
        _add_mixed(grid, rng, variant)
    elif family == "rooms":
        _add_rooms(grid, rng, variant)
    else:
        _add_zigzag(grid, rng, variant)

    # Keep the task solvable, while retaining enough alternative free space that
    # the benchmark is not merely a one-path maze.
    carve_guaranteed_route(grid, start, goal, rng)
    for _ in range(max(2, size // 4)):
        r = int(rng.integers(1, size - 1))
        c = int(rng.integers(1, size - 1))
        grid[r, c] = 0

    grid[start] = 0
    grid[goal] = 0

    attempts = 0
    while not reachable(grid, start, goal) and attempts < size * size:
        wall_cells = np.argwhere(grid[1:-1, 1:-1] == 1)
        if len(wall_cells) == 0:
            break
        rr, cc = wall_cells[int(rng.integers(0, len(wall_cells)))]
        grid[rr + 1, cc + 1] = 0
        attempts += 1

    # Avoid layouts that collapse to a nearly direct Manhattan route.
    direct = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    if shortest_distance(grid, start, goal) <= direct + 1 and size >= 15:
        mid = size // 2
        if variant % 2 == 0:
            grid[2:-2, mid] = 1
            grid[2 + (variant * 3) % max(1, size - 4), mid] = 0
        else:
            grid[mid, 2:-2] = 1
            grid[mid, 2 + (variant * 3) % max(1, size - 4)] = 0
        carve_guaranteed_route(grid, start, goal, rng)

    return grid, start, goal


def save_map(path: Path, grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> None:
    chars = np.full(grid.shape, ".", dtype="<U1")
    chars[grid == 1] = "#"
    chars[start] = "A"
    chars[goal] = "G"
    path.write_text("\n".join("".join(row) for row in chars) + "\n", encoding="utf-8")


def main() -> None:
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for p in list(TRAIN_DIR.glob("*.txt")) + list(TEST_DIR.glob("*.txt")):
        p.unlink()

    # 54 training layouts: 18 each at 10x10, 15x15, and 20x20.
    train_index = 1
    for size in (10, 15, 20):
        for variant in range(18):
            family = FAMILIES[variant % len(FAMILIES)]
            grid, start, goal = make_map(size, seed=10_000 + size * 100 + variant * 13, variant=variant, family=family)
            save_map(TRAIN_DIR / f"train_{family}_{train_index:02d}_{size}x{size}.txt", grid, start, goal)
            train_index += 1

    # 18 unseen layouts: all six families at each map size. Generator seeds and
    # variants are disjoint from training layouts.
    test_index = 1
    for size in (10, 15, 20):
        for family in FAMILIES:
            variant = 80 + test_index
            grid, start, goal = make_map(size, seed=90_000 + size * 100 + test_index * 17, variant=variant, family=family)
            save_map(TEST_DIR / f"test_{family}_{test_index:02d}_{size}x{size}.txt", grid, start, goal)
            test_index += 1

    print(f"Generated {len(list(TRAIN_DIR.glob('*.txt')))} train maps and {len(list(TEST_DIR.glob('*.txt')))} test maps.")


if __name__ == "__main__":
    main()
