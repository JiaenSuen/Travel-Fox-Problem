from pathlib import Path

from tfp.tasks.registry import TaskSpec, register_task

TASK_ROOT = Path(__file__).resolve().parent
TRAIN_MAP_DIR = TASK_ROOT / "maps" / "train"
TEST_MAP_DIR = TASK_ROOT / "maps" / "test"
DEFAULT_EVAL_SEEDS = (103, 223, 311, 419, 521)

TASK_SPEC = TaskSpec(
    env_id="TFP-ColorSort",
    code="COLOR-SORT",
    display_name="Color-Matched Sorting",
    description="Pick 2–5 colored objects and deliver each to the matching colored destination.",
    package=__name__,
    env_class="tfp.tasks.color_sort.environment.ColorSortEnv",
    train_map_dir=TRAIN_MAP_DIR,
    test_map_dir=TEST_MAP_DIR,
    default_eval_seeds=DEFAULT_EVAL_SEEDS,
    default_observation_mode="local",
    default_view_size=5,
    supported_view_sizes=(5, 7),
    default_model="001_color_cnn",
    default_reward="001_dense_color_sort",
)

try:
    register_task(TASK_SPEC)
except ValueError:
    pass
