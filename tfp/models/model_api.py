from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable

from torch import nn

from tfp.tasks import default_task_id, get_task


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    algorithm: str = "ppo"
    recurrent: bool = False
    memory_type: str = "none"
    intrinsic_module: str = "none"
    bootstrap_from: str = ""
    description: str = ""


LEGACY_MODEL_ALIASES = {
    "simple_cnn_001": "001_simple_cnn",
    "simple_cnn_tabu_002": "002_simple_cnn_tabu",
    "action_memory_cnn_003": "003_action_memory_cnn",
    "simple_cnn_tabux_004": "004_simple_cnn_tabux",
    "action_memory_tabux_cnn_005": "005_action_memory_tabux_cnn",
    "ppo_gru_staged_006": "006_ppo_gru_bootstrap",
    "ppo_gru_action_memory_008": "007_ppo_gru_action_memory",
    "008_ppo_gru_action_memory": "007_ppo_gru_action_memory",
    "ppo_gru_episodic_count_010": "008_ppo_gru_episodic_count",
    "010_ppo_gru_episodic_count": "008_ppo_gru_episodic_count",
    "ppo_gru_gobi_014": "010_ppo_gru_gobi",
    "014_ppo_gru_gobi": "010_ppo_gru_gobi",
    "012_ppo_gru_gobi": "010_ppo_gru_gobi",
}


def normalize_model_name(module_name: str) -> str:
    return LEGACY_MODEL_ALIASES.get(str(module_name), str(module_name))


def _module_to_spec(module: ModuleType) -> ModelSpec | None:
    raw = getattr(module, "MODEL_SPEC", None)
    factory = getattr(module, "create_model", None)
    if raw is None or not callable(factory):
        return None
    return raw if isinstance(raw, ModelSpec) else ModelSpec(**raw)


def _task_id(task_id: str | None) -> str:
    return task_id or default_task_id()


def discover_model_plugins(task_id: str | None = None) -> dict[str, ModelSpec]:
    """Discover model plugins owned by one task.

    Files live in ``tfp/tasks/<task>/models``. Shared neural building blocks may stay
    in ``tfp.models`` but experiment-facing model definitions are task-scoped.
    """
    spec = get_task(_task_id(task_id))
    package = importlib.import_module(spec.model_package)
    output: dict[str, ModelSpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_"):
            continue
        full_name = f"{spec.model_package}.{module_info.name}"
        importlib.invalidate_caches()
        module = importlib.reload(sys.modules[full_name]) if full_name in sys.modules else importlib.import_module(full_name)
        model_spec = _module_to_spec(module)
        if model_spec is not None:
            output[module_info.name] = model_spec
    return dict(sorted(output.items()))


def load_model_plugin(
    module_name: str,
    task_id: str | None = None,
) -> tuple[ModelSpec, Callable[[tuple[int, int, int], int], nn.Module]]:
    module_name = normalize_model_name(module_name)
    spec = get_task(_task_id(task_id))
    full_name = f"{spec.model_package}.{module_name}"
    try:
        module = importlib.import_module(full_name)
    except ModuleNotFoundError as exc:
        if exc.name == full_name:
            raise ValueError(
                f"Model '{module_name}' is not registered for task '{spec.env_id}'. "
                f"Available: {', '.join(discover_model_plugins(spec.env_id))}"
            ) from exc
        raise
    model_spec = _module_to_spec(module)
    if model_spec is None:
        raise ValueError(f"{full_name} must define MODEL_SPEC and create_model(observation_shape, action_count).")
    return model_spec, getattr(module, "create_model")


def rollout_forward(model: nn.Module, x, env_ids):
    hook = getattr(model, "act_forward", None)
    if callable(hook):
        return hook(x, tuple(int(v) for v in env_ids), update=True)
    logits, value = model(x)
    return logits, value, None


def peek_forward(model: nn.Module, x, env_ids):
    hook = getattr(model, "act_forward", None)
    if callable(hook):
        return hook(x, tuple(int(v) for v in env_ids), update=False)
    logits, value = model(x)
    return logits, value, None


def optimization_forward(model: nn.Module, x, context=None):
    hook = getattr(model, "training_forward", None)
    if callable(hook) and context is not None:
        return hook(x, context)
    return model(x)
