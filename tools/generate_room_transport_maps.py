from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tfp" / "tasks" / "room_transport" / "maps"

TIERS = {
    "small": (15, 2, 2),
    "medium": (19, 2, 3),
    "large": (23, 3, 3),
    "large_plus": (27, 3, 4),
    "large_plusplus": (31, 4, 4),
}


def _partition(total: int, parts: int, rng: random.Random, minimum: int = 4) -> list[int]:
    base = [minimum] * parts
    remaining = total - minimum * parts
    if remaining < 0:
        raise ValueError((total, parts))
    for _ in range(remaining):
        base[rng.randrange(parts)] += 1
    # Small local transfers create visibly non-uniform rooms without changing map size.
    for _ in range(parts * 2):
        i, j = rng.sample(range(parts), 2)
        if base[i] > minimum and rng.random() < 0.65:
            base[i] -= 1
            base[j] += 1
    return base


def make_layout(size: int, room_rows: int, room_cols: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    interior = size - 2
    row_heights = _partition(interior - (room_rows - 1), room_rows, rng)
    col_widths = _partition(interior - (room_cols - 1), room_cols, rng)

    grid = [["#"] * size for _ in range(size)]
    row_spans = []
    cursor = 1
    for height in row_heights:
        row_spans.append((cursor, cursor + height - 1))
        cursor += height + 1
    col_spans = []
    cursor = 1
    for width in col_widths:
        col_spans.append((cursor, cursor + width - 1))
        cursor += width + 1

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    grid[r][c] = "."

    # Every edge in the room lattice has one doorway. Door cells are passable only
    # when the environment has opened them; their initial state is sampled per episode.
    for rr, (r0, r1) in enumerate(row_spans):
        for cc in range(room_cols - 1):
            boundary = col_spans[cc][1] + 1
            candidates = list(range(r0 + 1, r1)) or list(range(r0, r1 + 1))
            grid[rng.choice(candidates)][boundary] = "D"
    for rr in range(room_rows - 1):
        boundary = row_spans[rr][1] + 1
        for cc, (c0, c1) in enumerate(col_spans):
            candidates = list(range(c0 + 1, c1)) or list(range(c0, c1 + 1))
            grid[boundary][rng.choice(candidates)] = "D"

    # A few room-internal blocks add local navigation variation while keeping each
    # room broadly open. Blocks never touch door cells or form full separators.
    room_count = room_rows * room_cols
    obstacle_budget = max(0, room_count // 2)
    for _ in range(obstacle_budget):
        rr = rng.randrange(room_rows)
        cc = rng.randrange(room_cols)
        r0, r1 = row_spans[rr]
        c0, c1 = col_spans[cc]
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        r = rng.randint(r0 + 1, r1 - 1)
        c = rng.randint(c0 + 1, c1 - 1)
        if grid[r][c] == ".":
            grid[r][c] = "#"

    # Fixed start marker in the first room; object and goal are episode-seeded.
    r0, r1 = row_spans[0]
    c0, c1 = col_spans[0]
    start_candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if grid[r][c] == "."]
    sr, sc = start_candidates[len(start_candidates) // 2]
    grid[sr][sc] = "A"
    return ["".join(row) for row in grid]


def write_split(split: str, count_per_tier: int, seed_offset: int) -> None:
    out = TASK_ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    for tier_index, (tier, (size, rr, cc)) in enumerate(TIERS.items(), start=1):
        for index in range(1, count_per_tier + 1):
            seed = seed_offset + tier_index * 1000 + index * 37
            lines = make_layout(size, rr, cc, seed)
            name = f"{split}_{tier}_{index:02d}_{size}x{size}.txt"
            (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_split("train", 10, 17000)
    write_split("test", 4, 29000)
    print("generated", len(list((TASK_ROOT/'train').glob('*.txt'))), "train and", len(list((TASK_ROOT/'test').glob('*.txt'))), "test maps")
