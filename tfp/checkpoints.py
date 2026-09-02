from __future__ import annotations

import re
from pathlib import Path

from tfp.tasks import get_task

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_ROOT = ROOT / "checkpoints"


def _safe_component(value: str) -> str:
    """Return a filesystem-safe identifier while preserving readable model/task names."""
    value = Path(str(value)).stem.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value or "unnamed"


def checkpoint_path(task_id: str, model_name: str) -> Path:
    """Canonical checkpoint path for a named TFP task + model plugin.

    Example:
        checkpoints/FOX-TR-L1/simple_cnn_001.pt

    The model plugin filename/module name is the checkpoint basename. In the release build
    this canonical ``.pt`` is the best validation checkpoint; training additionally
    writes ``<model>.last.pt`` for debugging/reproducibility. This keeps evaluation
    deterministic and prevents one model from silently loading another model's weights.
    """
    task = get_task(task_id)
    task_dir = _safe_component(task.code)
    model_file = f"{_safe_component(model_name)}.pt"
    return CHECKPOINT_ROOT / task_dir / model_file


def last_checkpoint_path(task_id: str, model_name: str) -> Path:
    """Return the non-exported final training endpoint for a model plugin."""
    best = checkpoint_path(task_id, model_name)
    return best.with_name(f"{best.stem}.last{best.suffix}")
