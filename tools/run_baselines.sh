#!/usr/bin/env bash
# Table 4-3 baselines: pure-vision backbones (trained) + SigLIP 2 zero-shot.
#
#   bash tools/run_baselines.sh [first_gpu] [backbones...]
#
# Runs are placed round-robin over the given GPUs and share cards with whatever is
# already there -- a baseline needs ~2 GB and the cards have >20 GB free, so GPU
# memory is not the constraint. CPU is: every job is limited by JPEG decode, so
# this caps itself at 6 workers per job and refuses to start more jobs than there
# are spare cores.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

FIRST_GPU="${1:-0}"; shift || true
BACKBONES=("$@")
[[ ${#BACKBONES[@]} -eq 0 ]] && BACKBONES=(resnet50 vit_b_16 swin_b)
EPOCHS="${EPOCHS:-30}"
WORKERS="${WORKERS:-6}"
BATCH="${BATCH:-64}"

# how many cores are already spoken for
load1=$(cut -d' ' -f1 /proc/loadavg | cut -d. -f1)
cores=$(nproc)
spare=$((cores - load1))
echo "[$(date +%H:%M:%S)] load ${load1}/${cores} -> ${spare} spare cores for $(( ${#BACKBONES[@]} + 1 )) job(s)"
if [[ $spare -lt $(( (${#BACKBONES[@]} + 1) * WORKERS / 2 )) ]]; then
  echo "WARNING: few spare cores; these jobs will run slowly and slow the others"
fi

mkdir -p runs
i=$FIRST_GPU
for b in "${BACKBONES[@]}"; do
  g=$((i % 4)); i=$((i + 1))
  free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  if [[ "$free" -lt 4000 ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $b: cuda:$g only has ${free} MiB free"
    continue
  fi
  echo "[$(date +%H:%M:%S)] start $b on cuda:$g (${free} MiB free) -> runs/baseline_$b.log"
  rm -rf "runs/baseline_$b"
  CUDA_VISIBLE_DEVICES="$g" setsid nice -n 15 python -u -m rst.baselines train \
      --backbone "$b" --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" --gpu 0 \
      > "runs/baseline_$b.log" 2>&1 < /dev/null &
done

g=$((i % 4))
echo "[$(date +%H:%M:%S)] start SigLIP2 zero-shot on cuda:$g -> runs/baseline_zeroshot.log"
CUDA_VISIBLE_DEVICES="$g" setsid nice -n 15 python -u -m rst.baselines zeroshot --gpu 0 \
    --workers 4 > runs/baseline_zeroshot.log 2>&1 < /dev/null &

wait
echo "[$(date +%H:%M:%S)] all baseline jobs finished"
