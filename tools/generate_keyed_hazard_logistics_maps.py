from __future__ import annotations

import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tfp" / "tasks" / "keyed_hazard_logistics" / "maps"

# Five long-horizon tiers requested by the task design. The room lattice grows while
# individual room sizes remain deliberately irregular.
TIERS = {
    "medium": (19, 2, 3),
    "large": (23, 3, 3),
    "large_plus": (27, 3, 4),
    "large_plusplus": (31, 4, 4),
    "large_plusplusplus": (35, 4, 5),
}


def _partition(total: int, parts: int, rng: random.Random, minimum: int = 4) -> list[int]:
    values = [minimum] * parts
    remaining = total - minimum * parts
    if remaining < 0:
        raise ValueError((total, parts, minimum))
    for _ in range(remaining):
        values[rng.randrange(parts)] += 1
    for _ in range(parts * 4):
        i, j = rng.sample(range(parts), 2)
        if values[i] > minimum and rng.random() < 0.7:
            values[i] -= 1
            values[j] += 1
    return values


def make_layout(size: int, room_rows: int, room_cols: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    interior = size - 2
    row_heights = _partition(interior - (room_rows - 1), room_rows, rng)
    col_widths = _partition(interior - (room_cols - 1), room_cols, rng)

    grid = [["#"] * size for _ in range(size)]
    row_spans: list[tuple[int, int]] = []
    cursor = 1
    for height in row_heights:
        row_spans.append((cursor, cursor + height - 1))
        cursor += height + 1
    col_spans: list[tuple[int, int]] = []
    cursor = 1
    for width in col_widths:
        col_spans.append((cursor, cursor + width - 1))
        cursor += width + 1

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    grid[r][c] = "."

    # One structural door on every room-lattice edge. Episode reset decides which
    # subset becomes colored/key-locked; the rest are ordinary stateful doors.
    for rr, (r0, r1) in enumerate(row_spans):
        for cc in range(room_cols - 1):
            boundary = col_spans[cc][1] + 1
            choices = list(range(r0 + 1, r1)) or list(range(r0, r1 + 1))
            grid[rng.choice(choices)][boundary] = "D"
    for rr in range(room_rows - 1):
        boundary = row_spans[rr][1] + 1
        for cc, (c0, c1) in enumerate(col_spans):
            choices = list(range(c0 + 1, c1)) or list(range(c0, c1 + 1))
            grid[boundary][rng.choice(choices)] = "D"

    # Local structure varies independently of the room graph. Short wall bars and
    # isolated blocks create occlusion and route ambiguity without sealing a room.
    room_count = room_rows * room_cols
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            budget = 1 + int(rng.random() < 0.55 and room_count >= 9)
            for _ in range(budget):
                if r1 - r0 < 4 or c1 - c0 < 4:
                    continue
                if rng.random() < 0.5:
                    r = rng.randint(r0 + 1, r1 - 1)
                    c = rng.randint(c0 + 1, max(c0 + 1, c1 - 2))
                    length = min(rng.randint(1, 2), c1 - c)
                    for k in range(length):
                        if grid[r][c + k] == ".":
                            grid[r][c + k] = "#"
                else:
                    c = rng.randint(c0 + 1, c1 - 1)
                    r = rng.randint(r0 + 1, max(r0 + 1, r1 - 2))
                    length = min(rng.randint(1, 2), r1 - r)
                    for k in range(length):
                        if grid[r + k][c] == ".":
                            grid[r + k][c] = "#"

    r0, r1 = row_spans[0]
    c0, c1 = col_spans[0]
    starts = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if grid[r][c] == "."]
    sr, sc = starts[len(starts) // 2]
    grid[sr][sc] = "A"
    return ["".join(row) for row in grid]


def write_split(split: str, count_per_tier: int, seed_offset: int) -> None:
    out = TASK_ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    for tier_index, (tier, (size, rr, cc)) in enumerate(TIERS.items(), start=1):
        for index in range(1, count_per_tier + 1):
            seed = seed_offset + tier_index * 1901 + index * 53
            lines = make_layout(size, rr, cc, seed)
            (out / f"{split}_{tier}_{index:02d}_{size}x{size}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_split("train", 10, 41000)
    write_split("test", 4, 73000)
    print("generated", len(list((TASK_ROOT / "train").glob("*.txt"))), "train and", len(list((TASK_ROOT / "test").glob("*.txt"))), "test maps")
