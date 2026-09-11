from pathlib import Path

from tfp.tasks.registry import TaskSpec, register_task

TASK_ROOT = Path(__file__).resolve().parent
TRAIN_MAP_DIR = TASK_ROOT / "maps" / "train"
TEST_MAP_DIR = TASK_ROOT / "maps" / "test"
DEFAULT_EVAL_SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009)

TASK_SPEC = TaskSpec(
    env_id="TFP-FoxTransport-Local",
    code="FOX-TR-L1",
    display_name="Fox Transport · Local",
    description="Local-perception single-cargo transport across multi-scale indoor maps.",
    package=__name__,
    env_class="tfp.envs.transport_env.TransportEnv",
    train_map_dir=TRAIN_MAP_DIR,
    test_map_dir=TEST_MAP_DIR,
    default_eval_seeds=DEFAULT_EVAL_SEEDS,
    default_observation_mode="local",
    default_view_size=5,
    supported_view_sizes=(5, 7),
    default_model="001_simple_cnn",
    default_reward="001_dense_transport",
)

try:
    register_task(TASK_SPEC)
except ValueError:
    pass
