from pathlib import Path

from tfp.tasks.registry import TaskSpec, register_task

TASK_ROOT = Path(__file__).resolve().parent
TRAIN_MAP_DIR = TASK_ROOT / "maps" / "train"
TEST_MAP_DIR = TASK_ROOT / "maps" / "test"
DEFAULT_EVAL_SEEDS = (109, 233, 347, 461, 577)

TASK_SPEC = TaskSpec(
    env_id="TFP-MovingCargoEvasion",
    code="MOVING-CARGO-EVASION",
    display_name="Moving Cargo & Predator Avoidance",
    description="Intercept cargo on a cyclic moving carrier, deliver it to the goal, and avoid a roaming white-wolf hazard.",
    package=__name__,
    env_class="tfp.tasks.moving_cargo_evasion.environment.MovingCargoEvasionEnv",
    train_map_dir=TRAIN_MAP_DIR,
    test_map_dir=TEST_MAP_DIR,
    default_eval_seeds=DEFAULT_EVAL_SEEDS,
    default_observation_mode="local",
    default_view_size=7,
    supported_view_sizes=(5, 7),
    default_model="001_intercept_safety_cnn",
    default_reward="001_intercept_safety_potential",
)

try:
    register_task(TASK_SPEC)
except ValueError:
    pass
