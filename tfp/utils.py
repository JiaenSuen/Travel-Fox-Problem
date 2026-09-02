from __future__ import annotations

import random
from pathlib import Path
from typing import List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_maps(map_dir: str | Path, split: str | None = None) -> List[Path]:
    """Discover fixed ASCII maps.

    New task directories store maps directly under ``train/`` or ``test/``. The
    optional split argument keeps compatibility with the original flat project layout.
    """
    root = Path(map_dir)
    pattern = f"{split}_*.txt" if split else "*.txt"
    paths = sorted(root.glob(pattern))
    if not paths:
        label = split if split is not None else root.name
        raise FileNotFoundError(f"No maps found for {label!r} in {root}")
    return paths
