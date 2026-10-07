#!/usr/bin/env bash
# Build the venv in /tmp (rerun after every new pod; takes a few minutes).
# Usage: bash ~/ECD/scripts/setup_pod.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-/tmp/venv}"
export PIP_CACHE_DIR=/tmp/pip-cache UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python TMPDIR=/tmp

echo "== disk =="; df -h ~ /tmp | sed 's/^/  /'
echo "== gpu =="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "  no GPU visible"

PY=""
for c in python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "no python >= 3.10; fetching one with uv into /tmp"
  command -v uv >/dev/null || pip install --user uv
  uv python install 3.11
  uv venv --python 3.11 "$VENV"
else
  echo "using $($PY --version)"
  "$PY" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip
# CUDA build of torch (needed for the GPU; the CPU wheel is NOT enough here). V100 = sm70, supported by cu121.
python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu121
python -m pip install -q -r "$ROOT/requirements.txt" pytest
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "bf16 native:", torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False))
PY
echo "done. every new shell:  source $ROOT/scripts/pod_env.sh"
