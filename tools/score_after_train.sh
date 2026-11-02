#!/usr/bin/env bash
# Wait for this project's own training jobs to finish, then score everything the
# paper's tables need on the test split.
#
#   nohup bash tools/score_after_train.sh > runs/score_after_train.log 2>&1 &
#
# Polls instead of sleeping blind, so it starts the moment a card frees up. It only
# ever looks for `rst.train` processes belonging to this user and only writes inside
# runs/, so it cannot disturb anyone else's job.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

# seed-0 runs are re-scored to add Hard Pair Accuracy (table 4-7); the rest are firsts
RUNS=(align_full_kb region_scene region_only no_hard
      align_full_kb_s1 region_scene_s1 region_only_s1 full_names)

echo "[$(date +%T)] waiting for training to finish"
while pgrep -u "$(id -un)" -f 'rst\.[t]rain' > /dev/null 2>&1; do sleep 60; done
echo "[$(date +%T)] training done; scoring ${#RUNS[@]} runs over 4 cards"

i=0
for n in "${RUNS[@]}"; do
  [[ -f "runs/$n/best.pt" ]] || { echo "skip $n (no best.pt)"; continue; }
  g=$((i % 4)); i=$((i + 1))
  (
    nice -n 10 python -u -m rst.eval_checkpoint --run "runs/$n" --ckpt best.pt \
        --split test --gpu "$g" --diagnose --batch 32 --workers 6 \
        > "runs/$n/score.log" 2>&1
    echo "[$(date +%T)] $n -> runs/$n/eval_test.json (rc=$?)"
  ) &
done
wait
echo "[$(date +%T)] ALL SCORING DONE"
