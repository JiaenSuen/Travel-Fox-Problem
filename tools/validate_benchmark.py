from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tfp.envs import TransportEnv
from tfp.tasks import get_task
from tfp.utils import discover_maps


def main() -> None:
    task = get_task("TFP-FoxTransport-Local")
    train_maps = discover_maps(task.train_map_dir)
    test_maps = discover_maps(task.test_map_dir)
    env = TransportEnv(test_maps, observation_mode=task.default_observation_mode, view_size=task.default_view_size)

    checked = 0
    for map_path in test_maps:
        for seed in task.default_eval_seeds:
            obs_a, info_a = env.reset(seed=seed, map_path=map_path)
            first_object = info_a["initial_object_pos"]
            obs_b, info_b = env.reset(seed=seed, map_path=map_path)
            assert obs_a.shape == (10, 5, 5)
            assert obs_b.shape == (10, 5, 5)
            assert first_object == info_b["initial_object_pos"]
            assert env.oracle_steps > 0
            checked += 1

    train_sizes = Counter(path.name.split("_")[-1].replace(".txt", "") for path in train_maps)
    test_sizes = Counter(path.name.split("_")[-1].replace(".txt", "") for path in test_maps)
    families = Counter(path.name.split("_")[1] for path in test_maps)

    print(f"Task: {task.env_id} [{task.code}]")
    print(f"Training maps: {len(train_maps)} {dict(train_sizes)}")
    print(f"Test maps: {len(test_maps)} {dict(test_sizes)}")
    print(f"Test topology families: {dict(families)}")
    print(f"Validated deterministic map-seed episodes: {checked}")
    print("Benchmark validation passed.")


if __name__ == "__main__":
    main()
