#!/usr/bin/env bash
# Train the three pure-vision baselines WITH the GT-box mask channel, so 表4-2 is
# input-matched against our Region.
#
# 3.1 gives our Region four channels (RGB + the binary mask of the ground-truth box),
# and 4.2.1 states the classification task is solved "在已知 Ground Truth Bounding Box
# 的条件下" -- so the baselines are entitled to that fourth channel too. Denying it to
# them was our choice, and it is the one difference between the two tables that is
# extra *information* rather than extra capacity, which is exactly what a reviewer
# would attack first.
#
# Only the channel count changes: lr, batch, epochs, sampler, class weights, crop
# geometry, splits and metrics are identical to the archived RGB-only baselines, so
# the delta between the two rows is attributable to the mask alone.
#
# Cards are polled, not assumed: a GPU that is free when this script starts may be
# taken by someone else before the job actually allocates, so each launch re-checks
# for zero compute processes and >= 12 GB free, and waits rather than sharing.
#
#   nohup bash tools/run_masked_baselines.sh > runs/masked_baselines.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-64}"
LR="${LR:-3e-4}"
WORKERS="${WORKERS:-6}"
MIN_FREE_MIB="${MIN_FREE_MIB:-12000}"
BACKBONES=("${@:-}")
[[ -z "${BACKBONES[0]}" ]] && BACKBONES=(resnet50 vit_b_16 swin_b)

free_card() {
  local i free procs
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    free=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    procs=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [[ "$procs" == "0" && "$free" -ge "$MIN_FREE_MIB" ]]; then
      echo "$i"; return 0
    fi
  done
  return 1
}

pids=()
for b in "${BACKBONES[@]}"; do
  while ! g=$(free_card); do
    echo "[$(date +%T)] no card with >= ${MIN_FREE_MIB} MiB free and no compute process; waiting"
    sleep 120
  done
  echo "[$(date +%T)] starting $b (4-channel) on cuda:$g"
  CUDA_VISIBLE_DEVICES="$g" nice -n 10 python -u -m rst.baselines train \
      --backbone "$b" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
      --workers "$WORKERS" --gpu 0 --mask-channel --tag mask \
      > "runs/baseline_${b}_mask.log" 2>&1 &
  pids+=($!)
  # give it time to allocate, otherwise the next iteration picks the same card
  sleep 90
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "[$(date +%T)] masked baselines done (fail=$fail)"
for b in "${BACKBONES[@]}"; do
  if [[ -f "runs/baseline_${b}_mask/eval_test.json" ]]; then
    printf '  %-12s %s\n' "$b" "$(tr -d '\n' < "runs/baseline_${b}_mask/eval_test.json" | cut -c1-150)"
  else
    printf '  %-12s (no eval_test.json -- check runs/baseline_%s_mask.log)\n' "$b" "$b"
  fi
done
exit $fail
