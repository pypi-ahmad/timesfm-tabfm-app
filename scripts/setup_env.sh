#!/usr/bin/env bash
set -euo pipefail

# Many sandboxed environments mount ~/.cache read-only and/or apply small /tmp quotas.
# Keep caches under the project directory by default.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.cache}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${ROOT_DIR}/.cache/hf/hub}"

EXTRAS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_env.sh [--dev] [--models] [--models-jax] [--models-xreg] [--finetune]

Defaults:
  --dev is enabled (pytest)
  --models is enabled (TimesFM torch + TabFM pytorch)

Notes:
  - Uses uv only.
  - Defaults UV/HF caches under the repository (./.uv-cache, ./.cache).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

want_dev=1
want_models=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-dev) want_dev=0; shift ;;
    --no-models) want_models=0; shift ;;
    --dev) want_dev=1; shift ;;
    --models) want_models=1; shift ;;
    --models-jax) EXTRAS+=("--extra" "models-jax"); shift ;;
    --models-xreg) EXTRAS+=("--extra" "models-xreg"); shift ;;
    --finetune) EXTRAS+=("--extra" "finetune"); shift ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

if [[ $want_dev -eq 1 ]]; then
  EXTRAS=("--extra" "dev" "${EXTRAS[@]}")
fi
if [[ $want_models -eq 1 ]]; then
  EXTRAS=("--extra" "models" "${EXTRAS[@]}")
fi

# Prefer CPU torch wheels to avoid downloading CUDA runtimes by default.
PYTORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
uv sync --index "${PYTORCH_CPU_INDEX}" "${EXTRAS[@]}"

echo "OK: uv environment synced."
