#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="$ROOT/config/conda_env.txt"
TFP_CONDA_ENV="tfp-rl"
if [[ -f "$ENV_FILE" ]]; then
  selected="$(tr -d '\r\n' < "$ENV_FILE")"
  if [[ -n "$selected" ]]; then
    TFP_CONDA_ENV="$selected"
  fi
fi
cd "$ROOT"
echo "[TFP] Launching TFP Studio in Conda env: $TFP_CONDA_ENV"
conda run --no-capture-output -n "$TFP_CONDA_ENV" python tfp_studio.py
