from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Callable


@dataclass(frozen=True)
class IntrinsicSpec:
    key: str
    display_name: str
    default_coef: float = 0.0
    trainable: bool = False
    anneal_fraction: float = 0.0
    description: str = ""


LEGACY_INTRINSIC_ALIASES = {
    "episodic_count_001": "001_episodic_count",
}



def normalize_intrinsic_name(module_name: str) -> str:
    return LEGACY_INTRINSIC_ALIASES.get(str(module_name), str(module_name))


def _module_to_spec(module: ModuleType) -> IntrinsicSpec | None:
    raw = getattr(module, "INTRINSIC_SPEC", None)
    factory = getattr(module, "create_intrinsic", None)
    if raw is None or not callable(factory):
        return None
    return raw if isinstance(raw, IntrinsicSpec) else IntrinsicSpec(**raw)


def discover_intrinsic_plugins() -> dict[str, IntrinsicSpec]:
    package = importlib.import_module("tfp.intrinsic")
    output: dict[str, IntrinsicSpec] = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.name.startswith("_") or module_info.name == "intrinsic_api":
            continue
        full_name = f"tfp.intrinsic.{module_info.name}"
        importlib.invalidate_caches()
        module = importlib.reload(sys.modules[full_name]) if full_name in sys.modules else importlib.import_module(full_name)
        spec = _module_to_spec(module)
        if spec is not None:
            output[module_info.name] = spec
    return dict(sorted(output.items()))


def load_intrinsic_plugin(module_name: str) -> tuple[IntrinsicSpec, Callable]:
    if module_name in {"", "none", "None", None}:  # type: ignore[comparison-overlap]
        raise ValueError("'none' does not have an intrinsic reward factory.")
    module_name = normalize_intrinsic_name(module_name)
    module = importlib.import_module(f"tfp.intrinsic.{module_name}")
    spec = _module_to_spec(module)
    if spec is None:
        raise ValueError(f"tfp.intrinsic.{module_name} must define INTRINSIC_SPEC and create_intrinsic().")
    return spec, getattr(module, "create_intrinsic")
