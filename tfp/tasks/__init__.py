"""Named task registry and task-scoped experiment assets."""

from .registry import TaskSpec, create_task_env, default_task_id, discover_tasks, get_task
from . import local_transport as _local_transport
from . import color_sort as _color_sort
from . import room_transport as _room_transport
from . import moving_cargo_evasion as _moving_cargo_evasion
from . import keyed_hazard_logistics as _keyed_hazard_logistics

__all__ = ["TaskSpec", "create_task_env", "default_task_id", "discover_tasks", "get_task"]
