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
    module_name = normalize_model_name(module_name)
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
