#!/usr/bin/env bash
# Pick a free GPU, lower our priority, and start training.
# Safe on a shared box: it never occupies a card someone else is using, and
# nice/OMP settings keep the CPU load down.
#
#   bash run_train.sh                       # auto-pick the emptiest GPU, full run
#   bash run_train.sh --gpu 2               # force cuda:2
#   bash run_train.sh --name myrun --epochs 30 --batch 16
#   bash run_train.sh --ablation region_only

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f env.sh ]] || { echo "run setup.sh first (no env.sh)"; exit 1; }
# shellcheck disable=SC1091
source ./env.sh

GPU=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    *)     ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$GPU" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  # choose the GPU with the most free memory, and refuse to run if all are busy
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -n -r | head -1 | cut -d, -f1 | tr -d ' ')
  FREE=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -n -r | head -1 | cut -d, -f2 | tr -d ' ')
  echo "auto-selected cuda:$GPU with ${FREE} MiB free"
  if (( FREE < 6000 )); then
    echo "No GPU has >=6 GB free. Refusing to start -- check nvidia-smi and re-run." >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2 || true
    exit 1
  fi
fi
[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

# nice 10 so interactive users on the box keep their latency
exec nice -n 10 python -u -m rst.train "${ARGS[@]+"${ARGS[@]}"}"
