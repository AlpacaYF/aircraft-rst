#!/usr/bin/env bash
# Run the table 4-6 / 4-7 configurations in parallel, one per idle GPU.
#
#   bash tools/run_matrix.sh                      # all five, incl. the full model
#   bash tools/run_matrix.sh region_only no_hard  # only these
#
# Shared-box discipline, enforced here rather than left to the caller:
#   * a GPU is only used if it has no compute process AND >= 8 GB free
#   * every process runs at nice 10
#   * 8 dataloader workers per run -- measured optimum; 16 and 24 are slower
#     because the bottleneck is CPU-side region cropping, not the GPU
#   * if fewer idle GPUs than configurations, the extras queue instead of
#     oversubscribing a card someone else is using
#
# A run needs ~5.6 GB, so a 24 GB card could hold three, but 72 cores / 8 workers
# caps sensible concurrency near 8 runs. One run per card keeps everyone else's
# interactive latency intact.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

EPOCHS="${EPOCHS:-50}"
WORKERS="${WORKERS:-8}"
BATCH="${BATCH:-16}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
# the full model is what run_pipeline.sh step 6 reads stage2.pt from
RUN_NAME_full="${RUN_NAME_full:-align_full_kb}"
# SEED + RUN_SUFFIX exist because run-to-run noise is 2.05 Macro-F1 points: a single
# ablation pair cannot support a claim, so repeats get their own run name.
SEED="${SEED:-}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
# EXTRA_ARGS is appended to every run in a batch, e.g. the class-name-only prototype
# control: EXTRA_ARGS="--text-features knowledge/text_features_names.npz"
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

WANT=("$@")
if [[ ${#WANT[@]} -eq 0 ]]; then
  WANT=(full region_only region_scene no_hard no_diff)
fi

ALL=(full region_only region_scene no_hard no_diff no_mask)
for c in "${WANT[@]}"; do
  ok=0; for a in "${ALL[@]}"; do [[ "$c" == "$a" ]] && ok=1; done
  [[ $ok -eq 1 ]] || { echo "unknown configuration: $c (one of: ${ALL[*]})" >&2; exit 1; }
done

# ---- which GPUs are actually free ------------------------------------------
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
FREE_GPUS=()
while read -r line; do
  idx="${line%%|*}"; rest="${line#*|}"
  free="${rest%%|*}"; procs="${rest##*|}"
  idx="$(echo "$idx" | tr -d ' ')"; free="$(echo "$free" | tr -d ' ')"
  procs="$(echo "$procs" | tr -d ' ')"
  if [[ "$procs" == "0" && "$free" -ge "$MIN_FREE_MIB" ]]; then
    FREE_GPUS+=("$idx")
    echo "gpu $idx usable (${free} MiB free, no compute process)"
  else
    echo "gpu $idx SKIPPED (${free} MiB free, $procs compute process(es)) -- in use"
  fi
done < <(
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    f=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits)
    p=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    echo "$i|$f|$p"
  done
)

if [[ ${#FREE_GPUS[@]} -eq 0 ]]; then
  echo "no idle GPU with >= ${MIN_FREE_MIB} MiB free -- refusing to start" >&2
  exit 1
fi
echo "${#FREE_GPUS[@]} usable GPU(s) for ${#WANT[@]} configuration(s)"

mkdir -p runs
pids=(); names=(); gpus=()
i=0
for c in "${WANT[@]}"; do
  g="${FREE_GPUS[$((i % ${#FREE_GPUS[@]}))]}"
  i=$((i + 1))
  if [[ "$c" == "full" ]]; then
    name="${RUN_NAME_full}"; abl=()
  else
    name="$c"; abl=(--ablation "$c")
  fi
  name="${name}${RUN_SUFFIX}"
  seedarg=(); [[ -n "$SEED" ]] && seedarg=(--seed "$SEED")
  log="runs/${name}.log"
  echo "[$(date +%H:%M:%S)] starting $c on cuda:$g -> $log"
  CUDA_VISIBLE_DEVICES="$g" nice -n 10 python -u -m rst.train \
      --name "$name" --epochs "$EPOCHS" --batch "$BATCH" --workers "$WORKERS" \
      --scene-cache "${abl[@]+"${abl[@]}"}" ${seedarg[@]+"${seedarg[@]}"} \
      "${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"}" > "$log" 2>&1 &
  pids+=($!); names+=("$name"); gpus+=("$g")
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

echo
echo "summaries:"
for j in "${!pids[@]}"; do
  n="${names[$j]}"
  if [[ -f "runs/$n/summary.json" ]]; then
    printf '  %-16s %s\n' "$n" "$(cat "runs/$n/summary.json" | tr -d '\n')"
  else
    printf '  %-16s (no summary.json)\n' "$n"
  fi
done
exit $fail
