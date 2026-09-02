from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    code: str
    display_name: str
    description: str
    train_map_dir: Path
    test_map_dir: Path
    default_eval_seeds: tuple[int, ...]
    default_observation_mode: str = "local"
    default_view_size: int = 5


_TASKS: dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> None:
    if spec.env_id in _TASKS:
        raise ValueError(f"Task id is already registered: {spec.env_id}")
    _TASKS[spec.env_id] = spec


def discover_tasks() -> dict[str, TaskSpec]:
    return dict(sorted(_TASKS.items()))


def get_task(task_id: str) -> TaskSpec:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        raise KeyError(f"Unknown TFP task: {task_id}. Available: {', '.join(_TASKS)}") from exc
