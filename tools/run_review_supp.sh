#!/usr/bin/env bash
# Review-response batch (2026-10-19): issue 1 (statistical support) + issue 2
# (DEDF gain-source controls). 13 runs, one per idle card, same shared-box
# discipline as run_matrix_v2.sh: idle check, nice 10, wait for the CUDA
# context to register before claiming the next card.
#
#   nohup bash tools/run_review_supp.sh > runs/review_supp_20261019.log 2>&1 &
#
# Composition:
#   seeds 1/2 for ro, rs, rt_adapt, rst_f        (6 runs -> mean+-std; seed ensemble)
#   ro_noaug seed 2                              (completes the 3-seed aug control)
#   同监督 trio (lambda_con 0, no hard, no diff)  (3 runs -> attribution)
#   rst_wide (hidden 1536)                       (1 run -> capacity-matched control)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
[[ -f ./env.sh ]] && source ./env.sh
PY="$ROOT/.venv/bin/python"
EPOCHS="${EPOCHS:-50}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-16}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
TEXT="knowledge/text_features_clipL_en_diff.npz"
V2_AUG="--z-s-env --proto-learnable --margin-hard --con-hard-weight 2.0 --remine-stage3 --erase-prob 0.25 --expand-jitter 0.10 --gate-active-only"
V2_NOAUG="--z-s-env --proto-learnable --margin-hard --con-hard-weight 2.0 --remine-stage3 --gate-active-only"

RUNS=(
  v2_ro_s1 v2_ro_s2 v2_ro_noaug_s2
  v2_rs_s1 v2_rs_s2
  v2_rt_adapt_s1 v2_rt_adapt_s2
  v2_rst_f_s1 v2_rst_f_s2
  v2_rs_nosup_s0 v2_rt_nosup_s0 v2_rst_nosup_s0
  v2_rst_wide_s0
)
declare -A ABL=(
  [v2_ro_s1]=region_only [v2_ro_s2]=region_only [v2_ro_noaug_s2]=region_only
  [v2_rs_s1]=region_scene [v2_rs_s2]=region_scene
  [v2_rt_adapt_s1]=region_text [v2_rt_adapt_s2]=region_text
  [v2_rs_nosup_s0]=region_scene [v2_rt_nosup_s0]=region_text
)
declare -A EXTRA=(
  [v2_ro_s1]="--seed 1"
  [v2_ro_s2]="--seed 2"
  [v2_ro_noaug_s2]="--seed 2"
  [v2_rs_s1]="--seed 1"
  [v2_rs_s2]="--seed 2"
  [v2_rt_adapt_s1]="--seed 1 --adaptive-sem"
  [v2_rt_adapt_s2]="--seed 2 --adaptive-sem"
  [v2_rst_f_s1]="--seed 1"
  [v2_rst_f_s2]="--seed 2"
  [v2_rs_nosup_s0]="--seed 0 --lambda-con 0 --no-use-hard-negative --no-use-diff-loss"
  [v2_rt_nosup_s0]="--seed 0 --adaptive-sem --lambda-con 0 --no-use-hard-negative --no-use-diff-loss"
  [v2_rst_nosup_s0]="--seed 0 --lambda-con 0 --no-use-hard-negative --no-use-diff-loss"
  [v2_rst_wide_s0]="--seed 0 --hidden 1536"
)
NOAUG_SET=" v2_ro_noaug_s2 "

usable_gpu() {
  local i f p
  i="$1"
  f=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  p=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [[ "$p" == "0" && "$f" -ge "$MIN_FREE_MIB" ]]
}

mkdir -p runs
pids=(); names=(); gpus=()
for r in "${RUNS[@]}"; do
  if [[ -f "runs/$r/best.pt" ]]; then
    echo "[$(date +%H:%M:%S)] SKIP $r (best.pt exists)"
    continue
  fi
  g=""
  while [[ -z "$g" ]]; do
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      if usable_gpu "$i"; then
        sleep 150
        if usable_gpu "$i"; then g="$i"; break; fi
      fi
    done
    if [[ -z "$g" ]]; then
      echo "[$(date +%H:%M:%S)] no idle GPU >= ${MIN_FREE_MIB} MiB -- waiting 300 s"
      sleep 300
    fi
  done

  abl="${ABL[$r]:-}"
  ablargs=()
  [[ -n "$abl" ]] && ablargs=(--ablation "$abl")
  flags="$V2_AUG"
  [[ "$NOAUG_SET" == *" $r "* ]] && flags="$V2_NOAUG"
  log="runs/${r}.log"
  echo "[$(date +%H:%M:%S)] starting $r on cuda:$g -> $log"
  CUDA_VISIBLE_DEVICES="$g" nice -n 10 "$PY" -u -m rst.train \
      --name "$r" --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" \
      --scene-cache --text-features "$TEXT" \
      "${ablargs[@]}" $flags ${EXTRA[$r]:-} > "$log" 2>&1 &
  pids+=($!); names+=("$r"); gpus+=("$g")

  ok=0
  for n in $(seq 1 40); do
    sleep 15
    p=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [[ "$p" != "0" ]]; then ok=1; break; fi
  done
  if [[ "$ok" != "1" ]]; then
    echo "[$(date +%H:%M:%S)] $r never registered on cuda:$g -- aborting" >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] $r registered on cuda:$g"
done

fail=0
for j in "${!pids[@]}"; do
  if wait "${pids[$j]}"; then
    echo "[$(date +%H:%M:%S)] DONE   ${names[$j]} (cuda:${gpus[$j]})"
  else
    echo "[$(date +%H:%M:%S)] FAILED ${names[$j]} (cuda:${gpus[$j]}) -- see runs/${names[$j]}.log"
    fail=1
  fi
done
exit $fail
