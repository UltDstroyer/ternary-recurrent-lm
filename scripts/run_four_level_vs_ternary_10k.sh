#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  python_bin=".venv/Scripts/python.exe"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  python_bin="$(command -v python)"
else
  echo "Python was not found. Install Python 3.10 or newer first." >&2
  exit 1
fi

if ! "$python_bin" -c "import torch" >/dev/null 2>&1; then
  echo "PyTorch is not installed for $python_bin." >&2
  echo "Create the environment and install the project with:" >&2
  echo "  python -m venv .venv" >&2
  echo "  source .venv/bin/activate  # Windows Git Bash: source .venv/Scripts/activate" >&2
  echo "  python -m pip install -e \".[dev]\"" >&2
  exit 1
fi

if ! "$python_bin" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "CUDA is not available to PyTorch. Install a CUDA-enabled PyTorch build and retry." >&2
  echo "The experiment intentionally stops instead of silently running for hours on the CPU." >&2
  exit 1
fi

if [[ ! -f "data/external/wikitext-2/train.txt" || ! -f "data/external/wikitext-2/valid.txt" ]]; then
  echo "WikiText-2 is missing; downloading it now." >&2
  "$python_bin" scripts/download_wikitext2.py
fi

gpu_name="$($python_bin -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "GPU: $gpu_name"
echo "Running 10,000 steps for the 3-level baseline and uniform 4-level model"
echo "Seeds: 19, 23, 29 (six runs total)"
echo "Results checkpoint after every completed run. Re-running this script resumes automatically."

PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$python_bin" -m trlm.experiment \
    --config configs/four_level_vs_ternary_10k.json \
    --output results/four_level_vs_ternary_10k.json \
    --device cuda \
    --variants q3_level_shared_capacity q4_level_shared_capacity \
    --resume \
    "$@"
