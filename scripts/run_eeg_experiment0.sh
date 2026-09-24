#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip
# Install AMD's official ROCm PyTorch wheel for your OS first; preserve it here.
if ! python -c 'import torch' >/dev/null 2>&1; then
  echo 'Install ROCm-enabled PyTorch in .venv using AMD instructions, then rerun.' >&2
  exit 1
fi
python -m pip install -e '.[eeg]'
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, 'HIP:', torch.version.hip, 'GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')
PY
python scripts/eeg_experiment0.py "$@"
