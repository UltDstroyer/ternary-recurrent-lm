#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON_BIN:-python}"
if [[ " $* " == *" --dataset wikitext2"* ]] &&
   [[ ! -f data/external/wikitext-2/train.txt ]]; then
  "$python_bin" scripts/download_wikitext2.py
fi
PYTHONPATH=src "$python_bin" scripts/four_level_branch_probe.py "$@"
