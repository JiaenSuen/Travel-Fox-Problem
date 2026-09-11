from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfp.tasks import create_task_env, discover_tasks
from tfp.utils import discover_maps


def main() -> None:
    total_checked = 0
    for task_id, task in discover_tasks().items():
        train_maps = discover_maps(task.train_map_dir)
        test_maps = discover_maps(task.test_map_dir)
        env = create_task_env(
            task_id,
            test_maps,
            observation_mode=task.default_observation_mode,
            view_size=task.default_view_size,
            reward_module=task.default_reward,
        )
        checked = 0
        for map_path in test_maps:
            for seed in task.default_eval_seeds:
                obs_a, info_a = env.reset(seed=seed, map_path=map_path)
                obs_b, info_b = env.reset(seed=seed, map_path=map_path)
                assert obs_a.shape == obs_b.shape == env.observation_shape
                assert info_a["oracle_steps"] == info_b["oracle_steps"]
                assert info_a["items_total"] == info_b["items_total"]
                assert env.oracle_steps > 0
                checked += 1
        train_sizes = Counter(path.name.split("_")[-1].replace(".txt", "") for path in train_maps)
        test_sizes = Counter(path.name.split("_")[-1].replace(".txt", "") for path in test_maps)
        print(f"Task: {task.env_id} [{task.code}]")
        print(f"  Training maps: {len(train_maps)} {dict(train_sizes)}")
        print(f"  Test maps: {len(test_maps)} {dict(test_sizes)}")
        print(f"  Views: {task.supported_view_sizes}; deterministic reset cases: {checked}")
        total_checked += checked
    print(f"Validated {total_checked} task/map/seed cases. Benchmark validation passed.")


if __name__ == "__main__":
    main()
