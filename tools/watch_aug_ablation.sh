#!/usr/bin/env bash
# Wait for the two augmentation-ablation runs, then dump test predictions.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"

for n in $(seq 1 120); do
  if [[ -f runs/v2_ro_erase_s0/summary.json && -f runs/v2_ro_jitter_s0/summary.json ]]; then
    echo "[$(date +%H:%M:%S)] both trainings finished"
    break
  fi
  sleep 300
done
for r in v2_ro_erase_s0 v2_ro_jitter_s0; do
  [[ -f "runs/$r/best.pt" ]] || { echo "missing best.pt $r"; continue; }
  CUDA_VISIBLE_DEVICES=1 nice -n 10 "$PY" -m rst.eval_checkpoint \
      --root . --run "runs/$r" --ckpt best.pt --split test --dump-samples \
      --workers 4 --gpu 0 > "runs/dump_${r}_aug.log" 2>&1
  echo "[$(date +%H:%M:%S)] dumped $r"
done
echo done > runs/aug_ablation_dumps_done.txt
echo "[$(date +%H:%M:%S)] ALL DONE"
