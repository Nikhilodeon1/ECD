#!/usr/bin/env bash
# Source (don't execute) in every new shell on a pod:  source ~/ECD/scripts/pod_env.sh
# Repo lives in the persistent home; venv + caches live in /tmp (large, wiped on a new pod).
export ECD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export VENV="${VENV:-/tmp/venv}"
export HF_HOME="${HF_HOME:-/tmp/hf}"
export PIP_CACHE_DIR=/tmp/pip-cache
export UV_CACHE_DIR=/tmp/uv-cache
export UV_PYTHON_INSTALL_DIR=/tmp/uv-python
export TMPDIR=/tmp
export TOKENIZERS_PARALLELISM=false
# public mode by default: loaders drop MIMIC rows. Only the cleared RunPod volume
# may run `export ECD_ALLOW_MIMIC=1` after sourcing this file.
unset ECD_ALLOW_MIMIC
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "[pod_env] no venv at $VENV yet -> run: bash $ECD_ROOT/scripts/setup_pod.sh"
fi
cd "$ECD_ROOT/src" 2>/dev/null || true
echo "[pod_env] ROOT=$ECD_ROOT VENV=$VENV HF_HOME=$HF_HOME MIMIC_ALLOWED=${ECD_ALLOW_MIMIC:-no}"
