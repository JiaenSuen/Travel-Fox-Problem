from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Protocol

from tfp.tasks import default_task_id, get_task


@dataclass(frozen=True)
class RewardSpec:
    key: str
    display_name: str
    description: str = ""


class RewardFunction(Protocol):
    def compute(self, transition: dict[str, object]) -> float: ...


LEGACY_REWARD_ALIASES = {
    "dense_transport_001": "001_dense_transport",
    "sparse_transport_001": "001_sparse_transport",
}


def normalize_reward_name(module_name: str) -> str:
    return LEGACY_REWARD_ALIASES.get(str(module_name), str(module_name))


def _module_to_spec(module: ModuleType) -> RewardSpec | None:
    raw = getattr(module, "REWARD_SPEC", None)
    factory = getattr(module, "create_reward", None)
    if raw is None or not callable(factory):
        return None
    return raw if isinstance(raw, RewardSpec) else RewardSpec(**raw)


def _task_id(task_id: str | None) -> str:
    return task_id or default_task_id()


def discover_reward_plugins(task_id: str | None = None) -> dict[str, RewardSpec]:
    spec = get_task(_task_id(task_id))
    package = importlib.import_module(spec.reward_package)
    output: dict[str, RewardSpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{spec.reward_package}.{module_info.name}"
        importlib.invalidate_caches()
        module = importlib.reload(sys.modules[full_name]) if full_name in sys.modules else importlib.import_module(full_name)
        reward_spec = _module_to_spec(module)
        if reward_spec is not None:
            output[module_info.name] = reward_spec
    return dict(sorted(output.items()))


def load_reward_plugin(
    module_name: str,
    task_id: str | None = None,
) -> tuple[RewardSpec, Callable[[], RewardFunction]]:
    module_name = normalize_reward_name(module_name)
    spec = get_task(_task_id(task_id))
    full_name = f"{spec.reward_package}.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name == full_name:
            raise ValueError(
                f"Reward '{module_name}' is not registered for task '{spec.env_id}'. "
                f"Available: {', '.join(discover_reward_plugins(spec.env_id))}"
            ) from exc
        raise
    reward_spec = _module_to_spec(module)
    if reward_spec is None:
        raise ValueError(f"{full_name} must define REWARD_SPEC and create_reward().")
    return reward_spec, getattr(module, "create_reward")
