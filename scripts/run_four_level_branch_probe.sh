#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
device=cpu
full_size=
dataset=synthetic
steps=
while (($#)); do
  case "$1" in
    --device) device="$2"; shift 2;;
    --full-size) full_size=--full-size; shift;;
    --dataset) dataset="$2"; shift 2;;
    --steps) steps="$2"; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
if [[ "$dataset" == wikitext2 && ! -f data/external/wikitext-2/train.txt ]]; then
  python scripts/download_wikitext2.py
fi
PYTHONPATH=src python scripts/four_level_branch_probe.py --device "$device" --dataset "$dataset" ${full_size:+$full_size} ${steps:+--steps "$steps"}
