from pathlib import Path

from tfp.tasks.registry import TaskSpec, register_task

TASK_ROOT = Path(__file__).resolve().parent
TRAIN_MAP_DIR = TASK_ROOT / "maps" / "train"
TEST_MAP_DIR = TASK_ROOT / "maps" / "test"
DEFAULT_EVAL_SEEDS = (107, 227, 337, 449, 557)

TASK_SPEC = TaskSpec(
    env_id="TFP-RoomTransport",
    code="ROOM-DOOR-TRANSPORT",
    display_name="Multi-Room Door Transport",
    description="Find one object across a multi-room layout, operate doors, and deliver it to a seeded destination room.",
    package=__name__,
    env_class="tfp.tasks.room_transport.environment.RoomTransportEnv",
    train_map_dir=TRAIN_MAP_DIR,
    test_map_dir=TEST_MAP_DIR,
    default_eval_seeds=DEFAULT_EVAL_SEEDS,
    default_observation_mode="local",
    default_view_size=7,
    supported_view_sizes=(5, 7),
    default_model="001_route_prior_cnn",
    default_reward="001_actionable_geodesic",
)

try:
    register_task(TASK_SPEC)
except ValueError:
    pass
