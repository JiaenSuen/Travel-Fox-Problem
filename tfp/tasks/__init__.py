"""Named task registry and task-scoped experiment assets."""

from .registry import TaskSpec, create_task_env, default_task_id, discover_tasks, get_task
from . import fox_transport_local as _fox_transport_local  # register built-in task 1
from . import fox_color_sort as _fox_color_sort  # register built-in task 2

__all__ = ["TaskSpec", "create_task_env", "default_task_id", "discover_tasks", "get_task"]
