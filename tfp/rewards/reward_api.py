from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Protocol


@dataclass(frozen=True)
class RewardSpec:
    key: str
    display_name: str
    description: str = ""


class RewardFunction(Protocol):
    def compute(self, transition: dict[str, object]) -> float: ...


def _module_to_spec(module: ModuleType) -> RewardSpec | None:
    raw = getattr(module, "REWARD_SPEC", None)
    factory = getattr(module, "create_reward", None)
    if raw is None or not callable(factory):
        return None
    return raw if isinstance(raw, RewardSpec) else RewardSpec(**raw)


def discover_reward_plugins() -> dict[str, RewardSpec]:
    package = importlib.import_module("tfp.rewards")
    output: dict[str, RewardSpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_") or module_info.name == "reward_api":
            continue
        full_name = f"tfp.rewards.{module_info.name}"
        importlib.invalidate_caches()
        module = importlib.reload(sys.modules[full_name]) if full_name in sys.modules else importlib.import_module(full_name)
        spec = _module_to_spec(module)
        if spec is not None:
            output[module_info.name] = spec
    return dict(sorted(output.items()))


def load_reward_plugin(module_name: str) -> tuple[RewardSpec, Callable[[], RewardFunction]]:
    module = importlib.import_module(f"tfp.rewards.{module_name}")
    spec = _module_to_spec(module)
    if spec is None:
        raise ValueError(f"tfp.rewards.{module_name} must define REWARD_SPEC and create_reward().")
    return spec, getattr(module, "create_reward")
