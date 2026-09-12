from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfp.tasks import create_task_env, get_task
from tfp.utils import discover_maps


def main() -> None:
    task = get_task("TFP-KeyedHazardLogistics")
    maps = discover_maps(task.test_map_dir)
    total = solvable = 0
    tiers: Counter[str] = Counter()
    for map_path in maps:
        for seed in task.default_eval_seeds:
            env = create_task_env(
                task.env_id, [map_path], view_size=task.default_view_size,
                seed=seed, reward_module=task.default_reward, wolves_enabled=False,
            )
            _, info = env.reset(seed=seed, map_path=map_path)
            total += 1
            tiers[str(info["size_tier"])] += 1
            solvable += int(env.structurally_solvable())
    print(f"Task: {task.code}")
    print(f"Test episodes: {total}")
    print(f"Structurally solvable: {solvable}/{total}")
    print("Tier cases:", dict(tiers))
    if solvable != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
