#!/usr/bin/env bash
# Re-score the existing checkpoints once, with per-entity predictions on disk.
#
# Every question the diagnosis document asks -- which entities flip between
# configurations, where the gate puts its weight, how large the semantic margin is,
# whether Scene is right where Region is wrong -- is answerable from these npz files
# offline. Re-scoring costs ~45 s per checkpoint and no training, so this is the cheap
# half of the whole investigation and it runs first on purpose.
#
#   GPU=1 nohup bash tools/dump_samples.sh > runs/dump_samples.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
source ./env.sh
export OMP_NUM_THREADS=2
GPU=${GPU:-1}
SPLIT=${SPLIT:-test}

free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "[$(date +%T)] cuda:$GPU has ${free} MiB free"
if [ "$free" -lt 6000 ]; then echo "refusing: < 6000 MiB free on cuda:$GPU" >&2; exit 1; fi

for n in "$@"; do
  if [ ! -f "runs/$n/best.pt" ]; then echo "[$(date +%T)] skip $n (no best.pt)"; continue; fi
  if [ -f "runs/$n/samples_$SPLIT.npz" ]; then echo "[$(date +%T)] have $n already"; continue; fi
  echo "[$(date +%T)] ==== $n ===="
  CUDA_VISIBLE_DEVICES=$GPU nice -n 15 .venv/bin/python -u -m rst.eval_checkpoint \
      --run "runs/$n" --ckpt best.pt --split "$SPLIT" --dump-samples --gpu -1 \
      >> "runs/dump_$n.log" 2>&1
  echo "[$(date +%T)] rc=$?"
done
echo "[$(date +%T)] done"
