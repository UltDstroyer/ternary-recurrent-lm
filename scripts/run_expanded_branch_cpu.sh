#!/usr/bin/env bash
# Run the same three-model comparison on this computer's CPU.
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: bash scripts/run_expanded_branch_cpu.sh [--resume] [--wikitext2] [--skip-install] [--torch-threads N]

Runs a two-step smoke test followed by a 300-step, 352-wide, six-pass CPU
comparison with up to eight idea threads and two splits per pass. WikiText-2
is optional; its byte-level next-token comparison runs after the pointer task.
JSON is checkpointed after each model. After an interruption, rerun with
--resume to keep the completed model results.
EOF
}

resume=0
wikitext2=0
skip_install=0
torch_threads=4
while (($#)); do
  case "$1" in
    --resume) resume=1; shift ;;
    --wikitext2) wikitext2=1; shift ;;
    --skip-install) skip_install=1; shift ;;
    --torch-threads)
      if (($# < 2)) || [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
        echo "--torch-threads requires a positive integer" >&2; exit 2
      fi
      torch_threads="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
else
  if [[ ! -x .venv/bin/python ]]; then
    if command -v python3 >/dev/null 2>&1; then
      python3 -m venv .venv
    else
      python -m venv .venv
    fi
  fi
  python_bin=.venv/bin/python
fi

if (( ! skip_install )); then
  if ! "$python_bin" -c 'import torch' >/dev/null 2>&1; then
    case "$(uname -s)" in
      Linux) "$python_bin" -m pip install torch --index-url https://download.pytorch.org/whl/cpu ;;
      *) "$python_bin" -m pip install torch ;;
    esac
  fi
  "$python_bin" -m pip install 'numpy>=1.26'
fi
"$python_bin" -c 'import torch, numpy; print("Python/PyTorch:", torch.__version__, "CUDA available:", torch.cuda.is_available())'
mkdir -p results
common=(--device cpu --torch-threads "$torch_threads" --full-size --max-threads 8 --children-per-pass 2 --split-threshold 0.15 --merge-threshold 0.995)
run_probe() {
  local output="$1"
  local log="${output%.json}.log"
  shift
  local extra=()
  if (( resume )) && [[ -f "$output" ]]; then extra+=(--resume); fi
  PYTHON_BIN="$python_bin" bash scripts/run_four_level_branch_probe.sh \
    "${common[@]}" "$@" "${extra[@]}" --output "$output" 2>&1 | tee "$log"
}

echo 'Smoke test: eight-way pointer task'
run_probe results/expanded_branch_cpu_smoke.json \
  --dataset synthetic_pointer --steps 2 --train-count 64 --validation-count 32 --batch-size 2

echo 'Full CPU comparison: eight-way pointer task'
run_probe results/expanded_branch_cpu_pointer.json \
  --dataset synthetic_pointer --steps 300 --train-count 4096 --validation-count 512 --batch-size 4

if (( wikitext2 )); then
  echo 'Full CPU comparison: WikiText-2 (byte-level, target after prefix)'
  run_probe results/expanded_branch_cpu_wikitext2.json \
    --dataset wikitext2 --steps 300 --train-count 4096 --validation-count 512 --batch-size 4
fi

echo 'Results: results/expanded_branch_cpu_{smoke,pointer,wikitext2}.json (WikiText-2 optional)'
