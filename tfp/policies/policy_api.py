from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Protocol, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PolicySpec:
    key: str
    display_name: str
    algorithm: str = "ppo"
    description: str = ""


class ActionPolicy(Protocol):
    def sample(self, model: nn.Module, logits: torch.Tensor, mask: torch.Tensor, env_ids: Sequence[int]): ...
    def greedy(self, model: nn.Module, logits: torch.Tensor, mask: torch.Tensor, env_ids: Sequence[int]) -> torch.Tensor: ...
    def observe(self, model: nn.Module, env_ids: Sequence[int], actions: Sequence[int], rewards: Sequence[float], dones: Sequence[bool]) -> None: ...
    def reset(self, model: nn.Module, env_ids: Sequence[int]) -> None: ...


def apply_model_action_rewrite(
    model: nn.Module,
    proposed: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Apply an optional inference-time action controller owned by the model.

    TFP deliberately keeps these rewrites out of PPO sampling. A plugin such as
    ``simple_cnn_tabu_002`` therefore learns from the unmodified neural policy and
    only uses its anti-cycle controller when ``greedy`` inference is requested.
    """
    hook = getattr(model, "rewrite_actions", None)
    if callable(hook):
        return hook(proposed, logits, mask, tuple(int(v) for v in env_ids))
    return proposed


def notify_model_action_memory(
    model: nn.Module,
    env_ids: Sequence[int],
    actions: Sequence[int],
    rewards: Sequence[float],
    dones: Sequence[bool],
) -> None:
    """Feed executed actions to an optional learnable action-memory model.

    This hook runs in both training and evaluation. It is distinct from the Tabu
    behavior controller: action-memory is part of the neural input context and must
    see the same action history used when the rollout was collected.
    """
    hook = getattr(model, "observe_action_memory", None)
    if callable(hook):
        hook(
            tuple(int(v) for v in env_ids),
            tuple(int(v) for v in actions),
            tuple(float(v) for v in rewards),
            tuple(bool(v) for v in dones),
        )


def notify_model_inference_behavior(
    model: nn.Module,
    env_ids: Sequence[int],
    actions: Sequence[int],
    rewards: Sequence[float],
    dones: Sequence[bool],
) -> None:
    """Update an optional inference-only behavior controller such as Tabu."""
    hook = getattr(model, "observe_behavior", None)
    if callable(hook):
        hook(
            tuple(int(v) for v in env_ids),
            tuple(int(v) for v in actions),
            tuple(float(v) for v in rewards),
            tuple(bool(v) for v in dones),
        )


def reset_model_memory(model: nn.Module, env_ids: Sequence[int]) -> None:
    ids = tuple(int(v) for v in env_ids)
    for hook_name in ("reset_action_memory", "reset_behavior_memory", "reset_sequence_memory"):
        hook = getattr(model, hook_name, None)
        if callable(hook):
            hook(ids)


def _module_to_spec(module: ModuleType) -> PolicySpec | None:
    raw = getattr(module, "POLICY_SPEC", None)
    factory = getattr(module, "create_policy", None)
    if raw is None or not callable(factory):
        return None
    return raw if isinstance(raw, PolicySpec) else PolicySpec(**raw)


def discover_policy_plugins() -> dict[str, PolicySpec]:
    package = importlib.import_module("tfp.policies")
    output: dict[str, PolicySpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_") or module_info.name == "policy_api":
            continue
        full_name = f"tfp.policies.{module_info.name}"
        importlib.invalidate_caches()
        module = importlib.reload(sys.modules[full_name]) if full_name in sys.modules else importlib.import_module(full_name)
        spec = _module_to_spec(module)
        if spec is not None:
            output[module_info.name] = spec
    return dict(sorted(output.items()))


def load_policy_plugin(module_name: str) -> tuple[PolicySpec, Callable[[], ActionPolicy]]:
    module = importlib.import_module(f"tfp.policies.{module_name}")
    spec = _module_to_spec(module)
    if spec is None:
        raise ValueError(f"tfp.policies.{module_name} must define POLICY_SPEC and create_policy().")
    return spec, getattr(module, "create_policy")
