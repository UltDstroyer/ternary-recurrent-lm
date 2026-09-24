#!/usr/bin/env bash
# Three-model comparison on an AMD ROCm GPU; PyTorch calls this device "cuda".
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: bash scripts/run_expanded_branch_rocm.sh [--resume] [--wikitext2] [--batch-size N]

Use an existing ROCm-enabled PyTorch environment. Set PYTHON_BIN to its Python
executable if it is not your active Python. A GPU check runs before training;
the script never silently falls back to CPU. It runs a two-step smoke check,
then a 300-step, 352-wide, six-pass comparison with up to eight idea threads.
Results are checkpointed after each model; rerun with --resume after a stop.
EOF
}

resume=0
wikitext2=0
batch_size=4
while (($#)); do
  case "$1" in
    --resume) resume=1; shift ;;
    --wikitext2) wikitext2=1; shift ;;
    --batch-size)
      if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--batch-size requires a positive integer" >&2; exit 2
      fi
      batch_size="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

has_rocm() {
  "$1" -c 'import torch; assert torch.version.hip and torch.cuda.is_available()' >/dev/null 2>&1
}
if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1 && has_rocm python; then
  python_bin=python
elif [[ -x .venv/bin/python ]] && has_rocm .venv/bin/python; then
  python_bin=.venv/bin/python
elif command -v python3 >/dev/null 2>&1 && has_rocm python3; then
  python_bin=python3
else
  echo 'No ROCm-enabled PyTorch Python found. Activate your ROCm environment or set PYTHON_BIN.' >&2
  echo 'AMD setup: https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html' >&2
  exit 1
fi

if ! "$python_bin" - <<'PY'
import numpy
import torch

if not torch.version.hip or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch cannot access the GPU; check the driver and PYTHON_BIN")
x = torch.randn(16, 16, device="cuda")
(x @ x).sum().item()
torch.cuda.synchronize()
print("PyTorch", torch.__version__, "ROCm", torch.version.hip,
      "GPU", torch.cuda.get_device_name(0), flush=True)
PY
then
  echo 'GPU preflight failed. Check ROCm PyTorch, NumPy, and your driver before retrying.' >&2
  exit 1
fi

mkdir -p results
common=(--device cuda --full-size --max-threads 8 --children-per-pass 2 --split-threshold 0.15 --merge-threshold 0.995)
run_probe() {
  local output="$1"
  local log="${output%.json}.log"
  shift
  local extra=()
  if (( resume )) && [[ -f "$output" ]]; then extra+=(--resume); fi
  PYTHON_BIN="$python_bin" bash scripts/run_four_level_branch_probe.sh \
    "${common[@]}" "$@" "${extra[@]}" --output "$output" 2>&1 | tee "$log"
}

echo 'ROCm GPU smoke test: eight-way pointer task'
run_probe results/expanded_branch_rocm_smoke.json \
  --dataset synthetic_pointer --steps 2 --train-count 64 --validation-count 32 --batch-size 2

echo 'Full ROCm GPU comparison: eight-way pointer task'
run_probe results/expanded_branch_rocm_pointer.json \
  --dataset synthetic_pointer --steps 300 --train-count 4096 --validation-count 512 \
  --batch-size "$batch_size"

if (( wikitext2 )); then
  echo 'Full ROCm GPU comparison: WikiText-2 (byte-level, target after prefix)'
  run_probe results/expanded_branch_rocm_wikitext2.json \
    --dataset wikitext2 --steps 300 --train-count 4096 --validation-count 512 \
    --batch-size "$batch_size"
fi

echo 'Finished. See results/expanded_branch_rocm_*.json and matching .log files.'
