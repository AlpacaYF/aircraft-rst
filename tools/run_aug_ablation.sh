#!/usr/bin/env bash
# Augmentation-component ablation: erase-only vs context-jitter-only, on the
# Region-Only recipe (same base as table 4-5's no-aug / both rows).
#   none      = region_only            (existing, 87.64 test top1)
#   both      = v2_ro_s0               (existing, 90.64 test top1)
#   erase     = this batch             (--erase-prob 0.25, jitter 0)
#   jitter    = this batch             (--expand-jitter 0.10, erase 0)
# Two runs on the two idle cards; test dumps run automatically afterwards.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"
BASE="--z-s-env --proto-learnable --margin-hard --con-hard-weight 2.0 --remine-stage3 --gate-active-only"
TEXT="knowledge/text_features_clipL_en_diff.npz"

launch() {
  local g="$1" name="$2"; shift 2
  echo "[$(date +%H:%M:%S)] starting $name on cuda:$g"
  CUDA_VISIBLE_DEVICES="$g" nice -n 10 "$PY" -u -m rst.train \
      --name "$name" --epochs 50 --batch 16 --workers 8 --scene-cache \
      --text-features "$TEXT" --ablation region_only --seed 0 \
      $BASE "$@" > "runs/${name}.log" 2>&1 &
  local pid=$!
  for n in $(seq 1 40); do
    sleep 15
    local p
    p=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [[ "$p" != "0" ]] && { echo "[$(date +%H:%M:%S)] $name registered"; return 0; }
  done
  echo "[$(date +%H:%M:%S)] $name never registered" >&2
  return 1
}

launch 1 v2_ro_erase_s0  --erase-prob 0.25 &
p1=$!
launch 2 v2_ro_jitter_s0 --expand-jitter 0.10 &
p2=$!
wait "$p1"; r1=$?
wait "$p2"; r2=$?

echo "[$(date +%H:%M:%S)] training finished (rc=$r1/$r2); dumping test"
for r in v2_ro_erase_s0 v2_ro_jitter_s0; do
  [[ -f "runs/$r/best.pt" ]] || { echo "missing best.pt for $r"; continue; }
  CUDA_VISIBLE_DEVICES=1 nice -n 10 "$PY" -m rst.eval_checkpoint \
      --root . --run "runs/$r" --ckpt best.pt --split test --dump-samples \
      --workers 4 --gpu 0 > "runs/dump_${r}.log" 2>&1
  echo "[$(date +%H:%M:%S)] dumped $r"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
