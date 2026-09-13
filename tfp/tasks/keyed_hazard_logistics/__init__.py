from pathlib import Path

from tfp.tasks.registry import TaskSpec, register_task

TASK_ROOT = Path(__file__).resolve().parent
TRAIN_MAP_DIR = TASK_ROOT / "maps" / "train"
TEST_MAP_DIR = TASK_ROOT / "maps" / "test"
DEFAULT_EVAL_SEEDS = (131, 257, 383, 509, 641)

TASK_SPEC = TaskSpec(
    env_id="TFP-KeyedHazardLogistics",
    code="KEYED-HAZARD-LOGISTICS",
    display_name="Keyed Multi-Cargo Logistics",
    description=(
        "Long-horizon multi-room logistics with colored access dependencies, one-to-three cargo objects, "
        "capacity-one delivery, and two roaming wolf hazards."
    ),
    package=__name__,
    env_class="tfp.tasks.keyed_hazard_logistics.environment.KeyedHazardLogisticsEnv",
    train_map_dir=TRAIN_MAP_DIR,
    test_map_dir=TEST_MAP_DIR,
    default_eval_seeds=DEFAULT_EVAL_SEEDS,
    default_observation_mode="local",
    default_view_size=7,
    supported_view_sizes=(5, 7),
    default_model="001_dependency_film_shield",
    default_reward="002_predictive_hazard_potential",
)

try:
    register_task(TASK_SPEC)
except ValueError:
    pass
