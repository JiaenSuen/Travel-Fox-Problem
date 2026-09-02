from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_CONFIG = ROOT / "config" / "runtime.json"
CONDA_ENV_FILE = ROOT / "config" / "conda_env.txt"


@dataclass(frozen=True)
class RuntimeStatus:
    expected_conda_env: str
    current_conda_env: str
    python_executable: str
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_name: str


def load_runtime_config() -> dict[str, object]:
    defaults: dict[str, object] = {
        "conda_env": "tfp-rl",
        "preferred_device": "cuda",
        "require_cuda_for_training": False,
    }
    if RUNTIME_CONFIG.exists():
        defaults.update(json.loads(RUNTIME_CONFIG.read_text(encoding="utf-8")))

    # ``conda_env.txt`` is deliberately tiny so Windows users can switch the
    # launcher environment with select_tfp_env.bat without editing Python/JSON.
    if CONDA_ENV_FILE.exists():
        selected = CONDA_ENV_FILE.read_text(encoding="utf-8").strip()
        if selected:
            defaults["conda_env"] = selected
    return defaults


def runtime_status() -> RuntimeStatus:
    cfg = load_runtime_config()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available"
    return RuntimeStatus(
        expected_conda_env=str(cfg.get("conda_env", "tfp-rl")),
        current_conda_env=os.environ.get("CONDA_DEFAULT_ENV", "Not a Conda shell"),
        python_executable=sys.executable,
        torch_version=torch.__version__,
        cuda_available=bool(torch.cuda.is_available()),
        cuda_version=str(torch.version.cuda or "None"),
        gpu_name=gpu_name,
    )


def resolve_device(requested: str | None = None) -> torch.device:
    cfg = load_runtime_config()
    requested = (requested or str(cfg.get("preferred_device", "cuda"))).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access an NVIDIA CUDA device in the current environment.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("Device must be 'cuda', 'cpu', or 'auto'.")
