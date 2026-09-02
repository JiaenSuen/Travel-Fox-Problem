from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    algorithm: str = "ppo"
    recurrent: bool = False
    memory_type: str = "none"
    description: str = ""


def _module_to_spec(module: ModuleType) -> ModelSpec | None:
    raw = getattr(module, "MODEL_SPEC", None)
    factory = getattr(module, "create_model", None)
    if raw is None or not callable(factory):
        return None
    if isinstance(raw, ModelSpec):
        return raw
    return ModelSpec(**raw)


def discover_model_plugins() -> dict[str, ModelSpec]:
    """Discover user model files placed directly in ``tfp/models``.

    A plugin only needs ``MODEL_SPEC`` and ``create_model``. Framework files that do
    not expose those names are ignored automatically.
    """
    package = importlib.import_module("tfp.models")
    output: dict[str, ModelSpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"tfp.models.{module_info.name}"
        importlib.invalidate_caches()
        module = importlib.reload(sys.modules[full_name]) if full_name in sys.modules else importlib.import_module(full_name)
        spec = _module_to_spec(module)
        if spec is not None:
            output[module_info.name] = spec
    return dict(sorted(output.items()))


def load_model_plugin(module_name: str) -> tuple[ModelSpec, Callable[[tuple[int, int, int], int], nn.Module]]:
    module = importlib.import_module(f"tfp.models.{module_name}")
    spec = _module_to_spec(module)
    if spec is None:
        raise ValueError(
            f"tfp.models.{module_name} is not a model plugin. "
            "Define MODEL_SPEC and create_model(observation_shape, action_count)."
        )
    return spec, getattr(module, "create_model")


def rollout_forward(model: nn.Module, x, env_ids):
    """Forward during environment interaction with optional model-owned memory.

    A stateful plugin may expose ``act_forward(x, env_ids, update=True)`` and
    return ``(logits, value, context_before)``. For example, Action-Memory CNN 003
    returns the recent action-token history. Stateless models require no changes
    and continue to use ``forward(x)``.
    """
    hook = getattr(model, "act_forward", None)
    if callable(hook):
        return hook(x, tuple(int(v) for v in env_ids), update=True)
    logits, value = model(x)
    return logits, value, None


def peek_forward(model: nn.Module, x, env_ids):
    """Value/logit forward that does not advance optional model-owned memory."""
    hook = getattr(model, "act_forward", None)
    if callable(hook):
        return hook(x, tuple(int(v) for v in env_ids), update=False)
    logits, value = model(x)
    return logits, value, None


def optimization_forward(model: nn.Module, x, context=None):
    """PPO minibatch forward using the model context captured at rollout time."""
    hook = getattr(model, "training_forward", None)
    if callable(hook) and context is not None:
        return hook(x, context)
    return model(x)
