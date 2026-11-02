#!/usr/bin/env bash
# Wait for a training run to publish its summary.json, then score its best checkpoint.
#
# Scoring is a ~45 s, ~3 GB job, so it is run at nice 15 on whatever card the training
# used -- it never needs to queue behind another training run. This exists so a run that
# finishes unattended at 03:00 does not sit unscored until someone looks.
#
#   GPU=1 nohup bash tools/watch_and_score_runs.sh region_only_fc region_only_fc_s1 \
#       > runs/score_after_fc.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source ./env.sh
GPU=${GPU:-1}
MIN_FREE=${MIN_FREE:-6000}

for name in "$@"; do
  echo "$(date '+%m-%d %H:%M') waiting for runs/$name/summary.json"
  while [ ! -f "runs/$name/summary.json" ]; do sleep 60; done
  # best.pt must outlive the trainer's CUDA context teardown or the two jobs share a card
  sleep 150
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  procs=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  echo "$(date '+%m-%d %H:%M') scoring $name on cuda:$GPU (${free} MiB free, $procs compute procs)"
  if [ "$free" -lt "$MIN_FREE" ]; then
    echo "  cuda:$GPU too full; waiting for another card"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$GPU nice -n 15 .venv/bin/python -u -m rst.eval_checkpoint \
      --run "runs/$name" --ckpt best.pt >> runs/score_after_$name.log 2>&1
  echo "  rc=$? -- see runs/score_after_$name.log"
done
echo "$(date '+%m-%d %H:%M') all requested runs scored"
