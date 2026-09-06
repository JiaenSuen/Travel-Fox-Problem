from __future__ import annotations

import re
from pathlib import Path

from tfp.models.model_api import LEGACY_MODEL_ALIASES, normalize_model_name
from tfp.tasks import get_task

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"


def _safe_component(value: str) -> str:
    value = Path(str(value)).stem.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value or "unnamed"


def checkpoint_path(task_id: str, model_name: str) -> Path:
    """Canonical *final training endpoint* path for a task + model plugin.

    ``<model>.pt`` is the model users saw at the end of
    training. Validation-selected weights are stored separately as ``<model>.best.pt``.
    This removes the previous surprise where quick evaluation showed the in-memory
    final model but the Evaluate tab silently loaded an earlier checkpoint.
    """
    task = get_task(task_id)
    task_dir = _safe_component(task.code)
    model_name = normalize_model_name(model_name)
    return CHECKPOINT_ROOT / task_dir / f"{_safe_component(model_name)}.pt"


def best_checkpoint_path(task_id: str, model_name: str) -> Path:
    final = checkpoint_path(task_id, model_name)
    return final.with_name(f"{final.stem}.best{final.suffix}")


def last_checkpoint_path(task_id: str, model_name: str) -> Path:
    """Backward-compatible alias; canonical ``.pt`` is already the final model."""
    return checkpoint_path(task_id, model_name)


def _legacy_names_for(new_name: str) -> list[str]:
    new_name = normalize_model_name(new_name)
    return [old for old, new in LEGACY_MODEL_ALIASES.items() if new == new_name]


def resolve_checkpoint_path(task_id: str, model_name: str, role: str = "last") -> Path:
    """Resolve a canonical checkpoint, with read-only fallback to legacy filenames."""
    if role not in {"last", "best"}:
        raise ValueError("checkpoint role must be 'last' or 'best'.")
    preferred = best_checkpoint_path(task_id, model_name) if role == "best" else checkpoint_path(task_id, model_name)
    if preferred.exists():
        return preferred

    task = get_task(task_id)
    task_dir = CHECKPOINT_ROOT / _safe_component(task.code)
    for legacy in _legacy_names_for(model_name):
        # Older releases wrote the final endpoint to <legacy>.last.pt and the validation-best
        # checkpoint to <legacy>.pt. Resolve those roles explicitly so a user can
        # reproduce the exact model that produced the last in-training quick test.
        if role == "last":
            candidate = task_dir / f"{_safe_component(legacy)}.last.pt"
            if candidate.exists():
                return candidate
        else:
            candidate = task_dir / f"{_safe_component(legacy)}.best.pt"
            if candidate.exists():
                return candidate
        # Legacy canonical .pt may be a validation-best fallback. For role=last it
        # is used only when no explicit .last.pt exists. Metadata still exposes the
        # actual role saved by the older trainer.
        candidate = task_dir / f"{_safe_component(legacy)}.pt"
        if candidate.exists():
            return candidate
    return preferred
