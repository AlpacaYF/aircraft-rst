#!/usr/bin/env bash
# v2 mechanism matrix: {ro, rs, rt, rst, nohard} x seeds for the monotone-ablation
# experiment. Same shared-box discipline as run_matrix.sh (idle-GPU check, nice 10,
# one run per card), plus a wait loop so more configurations than cards still all
# run instead of oversubscribing.
#
#   nohup bash tools/run_matrix_v2.sh ro rs rt rst nohard > runs/matrix_v2_s0.log 2>&1 &
#   SEED=1 nohup bash tools/run_matrix_v2.sh ... &
#
# v2 flags are the mechanism set under test; override with V2_FLAGS if a run
# should isolate one of them (e.g. V2_FLAGS="--z-s-env" for scene-only).
#
# Two launch bugs this version fixes, both observed on the first attempt:
#  * "${ABL[$c]:+"${ABL[$c]}"}" hands argparse ONE word containing a space
#    ("--ablation region_only"), which argparse rejects -- every ablated config
#    died at startup and only the un-ablated one trained. Ablations are now a
#    plain word in an associative array and split into a real array.
#  * A claimed GPU was re-checked 150 s later, but SigLIP2 weight loading keeps
#    the CUDA context unregistered for longer than that, so every claim in the
    # window judged the first card free and four jobs stacked on one GPU. After
#    starting a job the launcher now waits for that job's compute process to
#    actually appear on the card (up to 10 min) before claiming the next.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

EPOCHS="${EPOCHS:-50}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-16}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
SEED="${SEED:-0}"
TEXT_FEATURES="${TEXT_FEATURES:-knowledge/text_features_clipL_en_diff.npz}"
V2_FLAGS="${V2_FLAGS:---z-s-env --proto-learnable --margin-hard --con-hard-weight 2.0 --remine-stage3 --erase-prob 0.25 --expand-jitter 0.10 --gate-active-only}"

# configuration -> ablation word ("" = full model). The augmentation flags
# (erase/expand-jitter) apply to every row identically; the branch mechanisms
# only take effect where the branch is live, so the rows stay comparable at
# equal training budget. The _f variants re-run the gate-bug-affected rows with
# --gate-active-only (dead branch channels no longer swallow softmax mass).
declare -A ABL=(
  [ro]=region_only
  [rs]=region_scene
  [rt]=region_text
  [rst]=""
  [nohard]=no_hard
  [nodiff]=no_diff
  [rt_f]=region_text
  [rst_f]=""
  [nodiff_f]=no_diff
)
# Route-B few-shot configurations: fs{K}_{base}, each with its own K. They get
# --few-shot K --stage-epochs 10 (30 epochs total: scarce data overfits fast,
# and all three arms share the same shortened budget) and run names without the
# v2_ prefix so they never collide with full-data runs.
declare -A FS_ABL=(
  [fs10_ro]=region_only  [fs10_rt]=region_text  [fs10_rst]=""
  [fs5_ro]=region_only   [fs5_rt]=region_text   [fs5_rst]=""
)
declare -A FS_K=(
  [fs10_ro]=10 [fs10_rt]=10 [fs10_rst]=10
  [fs5_ro]=5   [fs5_rt]=5   [fs5_rst]=5
)
WANT=("$@")
if [[ ${#WANT[@]} -eq 0 ]]; then
  WANT=(ro rs rt rst nohard)
fi
for c in "${WANT[@]}"; do
  if [[ -n "${FS_ABL[$c]:-}" || "$c" == fs10_rst || "$c" == fs5_rst ]]; then continue; fi
  [[ -n "${ABL[$c]:-}" || "$c" == "rst" || "$c" == "rst_f" ]] || { echo "unknown configuration: $c" >&2; exit 1; }
done

usable_gpu() {
  local i f p
  i="$1"
  f=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  p=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [[ "$p" == "0" && "$f" -ge "$MIN_FREE_MIB" ]]
}

mkdir -p runs
pids=(); names=(); gpus=()
for c in "${WANT[@]}"; do
  g=""
  while [[ -z "$g" ]]; do
    for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
      if usable_gpu "$i"; then
        sleep 150                       # survive a racing queue on the same card
        if usable_gpu "$i"; then g="$i"; break; fi
      fi
    done
    if [[ -z "$g" ]]; then
      echo "[$(date +%H:%M:%S)] no idle GPU >= ${MIN_FREE_MIB} MiB -- waiting 300 s"
      sleep 300
    fi
  done

  if [[ -n "${FS_ABL[$c]:-}" || "$c" == fs10_rst || "$c" == fs5_rst ]]; then
    name="${c}_s${SEED}"
    abl="${FS_ABL[$c]}"
    fsargs=(--few-shot "${FS_K[$c]}" --stage-epochs 10)
    echo "[$(date +%H:%M:%S)] $c is a few-shot run (K=${FS_K[$c]}, 30 epochs)"
  else
    name="v2_${c}_s${SEED}"
    abl="${ABL[$c]}"
    fsargs=()
  fi
  log="runs/${name}.log"
  ablargs=()
  [[ -n "$abl" ]] && ablargs=(--ablation "$abl")

  echo "[$(date +%H:%M:%S)] starting $c (seed $SEED) on cuda:$g -> $log"
  CUDA_VISIBLE_DEVICES="$g" nice -n 10 python -u -m rst.train \
      --name "$name" --seed "$SEED" --epochs "$EPOCHS" --batch "$BATCH" \
      --workers "$WORKERS" --scene-cache --text-features "$TEXT_FEATURES" \
      "${ablargs[@]}" "${fsargs[@]}" $V2_FLAGS > "$log" 2>&1 &
  pids+=($!); names+=("$name"); gpus+=("$g")

  # hold this loop until the job's CUDA context registers on $g (weight loading
  # is CPU-side and shows nothing on nvidia-smi for 1-3 min). Without this, the
  # next claim re-judges the same card free and stacks on it.
  ok=0
  for n in $(seq 1 40); do              # 40 x 15 s = 10 min
    sleep 15
    p=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [[ "$p" != "0" ]]; then ok=1; break; fi
  done
  if [[ "$ok" != "1" ]]; then
    echo "[$(date +%H:%M:%S)] job on cuda:$g never registered in 10 min -- aborting before more jobs launch" >&2
    exit 1
  fi
  echo "[$(date +%H:%M:%S)] $c registered on cuda:$g"
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

echo; echo "summaries:"
for j in "${!names[@]}"; do
  n="${names[$j]}"
  if [[ -f "runs/$n/summary.json" ]]; then
    printf '  %-16s %s\n' "$n" "$(tr -d '\n' < "runs/$n/summary.json")"
  else
    printf '  %-16s (no summary.json)\n' "$n"
  fi
done
exit $fail
