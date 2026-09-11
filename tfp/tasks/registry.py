from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class TaskSpec:
    env_id: str
    code: str
    display_name: str
    description: str
    package: str
    env_class: str
    train_map_dir: Path
    test_map_dir: Path
    default_eval_seeds: tuple[int, ...]
    default_observation_mode: str = "local"
    default_view_size: int = 5
    supported_view_sizes: tuple[int, ...] = (5, 7)
    default_model: str = "001_simple_cnn"
    default_reward: str = "001_dense_transport"

    @property
    def model_package(self) -> str:
        return f"{self.package}.models"

    @property
    def reward_package(self) -> str:
        return f"{self.package}.rewards"


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


def default_task_id() -> str:
    if not _TASKS:
        raise RuntimeError("No TFP tasks are registered.")
    return next(iter(_TASKS))


def create_task_env(
    task_id: str,
    map_paths: Sequence[str | Path],
    **kwargs: Any,
):
    """Instantiate the environment declared by a named task.

    Core training/evaluation code calls this factory instead of importing a concrete
    environment. Adding a new task therefore does not require trainer changes.
    """
    spec = get_task(task_id)
    if int(kwargs.get("view_size", spec.default_view_size)) not in spec.supported_view_sizes:
        raise ValueError(
            f"Task {task_id} supports view sizes {spec.supported_view_sizes}, "
            f"got {kwargs.get('view_size')}."
        )
    module_name, class_name = spec.env_class.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls(map_paths=map_paths, **kwargs)
