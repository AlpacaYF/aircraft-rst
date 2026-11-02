#!/usr/bin/env bash
# Test-split scoring for the runs produced by the text-tower / fairness investigation.
# Light jobs (~3 GB, a few minutes each) so they are run at nice 15 and may share a
# card with a training job; the card is still checked for headroom first.
#
#   nohup bash tools/score_new_runs.sh > runs/score_new_runs.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
export OMP_NUM_THREADS=2

GPU="${GPU:-0}"
RUNS="${*:-full_cliptext_descdiff full_cliptext_diff region_only_nomask full_leftpad full_regdiff}"

free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
echo "[$(date +%T)] cuda:$GPU has ${free} MiB free; scoring at nice 15"
if [[ "$free" -lt 6000 ]]; then
  echo "refusing: < 6000 MiB free on cuda:$GPU" >&2
  exit 1
fi

for n in $RUNS; do
  if [[ ! -f "runs/$n/best.pt" ]]; then
    echo "[$(date +%T)] skip $n (no best.pt yet)"
    continue
  fi
  echo "[$(date +%T)] ==== $n ===="
  nice -n 15 python -u -m rst.eval_checkpoint --run "runs/$n" --ckpt best.pt \
      --split test --gpu "$GPU" --diagnose --batch 32 --workers 6 \
      > "runs/$n/score.log" 2>&1
  echo "[$(date +%T)] $n rc=$?"
done
echo "[$(date +%T)] score_new_runs done"
