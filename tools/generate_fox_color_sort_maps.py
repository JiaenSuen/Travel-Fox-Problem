from __future__ import annotations

from collections import deque
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "tfp" / "tasks" / "fox_color_sort" / "maps"
SIZES = (8, 10, 12)
FAMILIES = ("open", "sparse", "pillars", "lanes", "mixed")


def connected(grid: list[list[str]], start=(1, 1)) -> bool:
    h, w = len(grid), len(grid[0])
    free = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != "#"}
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            p = (r + dr, c + dc)
            if p in free and p not in seen:
                seen.add(p); q.append(p)
    return seen == free


def make_layout(size: int, family: str, seed: int) -> list[str]:
    rng = random.Random(seed)
    for _ in range(200):
        g = [["#" if r in (0, size - 1) or c in (0, size - 1) else "." for c in range(size)] for r in range(size)]
        if family == "sparse":
            for r in range(2, size - 1):
                for c in range(2, size - 1):
                    if rng.random() < 0.055:
                        g[r][c] = "#"
        elif family == "pillars":
            candidates = [(r, c) for r in range(2, size - 2, 2) for c in range(2, size - 2, 2)]
            rng.shuffle(candidates)
            for r, c in candidates[: max(1, size // 4)]:
                g[r][c] = "#"
        elif family == "lanes":
            if size >= 8:
                c = rng.randint(3, size - 4)
                for r in range(2, size - 2):
                    if r not in {size // 2, size // 2 + 1}:
                        g[r][c] = "#"
        elif family == "mixed":
            for _ in range(max(2, size // 3)):
                r, c = rng.randint(2, size - 3), rng.randint(2, size - 3)
                g[r][c] = "#"
            if size >= 10:
                r = rng.randint(3, size - 4)
                for c in range(3, size - 3):
                    if c != size // 2:
                        g[r][c] = "#"
        g[1][1] = "A"
        if connected(g, (1, 1)):
            return ["".join(row) for row in g]
    raise RuntimeError(f"Could not create connected {family} {size}x{size} layout")


def generate(split: str, per_size: int, seed_base: int) -> None:
    out = TASK_ROOT / split
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.txt"):
        old.unlink()
    idx = 1
    for size in SIZES:
        for j in range(per_size):
            family = FAMILIES[j % len(FAMILIES)]
            lines = make_layout(size, family, seed_base + size * 100 + j)
            name = f"{split}_{family}_{idx:02d}_{size}x{size}.txt"
            (out / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
            idx += 1


if __name__ == "__main__":
    generate("train", per_size=15, seed_base=2000)  # 45 train layouts
    generate("test", per_size=5, seed_base=9000)    # 15 unseen layouts
    print("Generated 45 train + 15 test Color Sort layouts.")
