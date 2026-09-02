"""Named task registry and fixed benchmark assets for TFP."""

from .registry import TaskSpec, discover_tasks, get_task
from . import fox_transport_local as _fox_transport_local  # register built-in task

__all__ = ["TaskSpec", "discover_tasks", "get_task"]
