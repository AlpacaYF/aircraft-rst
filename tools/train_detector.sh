#!/usr/bin/env bash
# Fine-tune a class-agnostic single-class aircraft detector for 表4-8.
#
# Weights come from the project HF cache, never from github: this box cannot reach
# github.com (timeouts), and ultralytics' default is to fetch release assets from
# there, which would fail halfway through a long run.
#
#   bash tools/train_detector.sh 0 yolo11x 30 640
#   bash tools/train_detector.sh 1 rtdetr-l 20 640     (once RT-DETR weights are cached)
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

GPU="${1:-0}"
MODEL="${2:-yolo11x}"
EPOCHS="${3:-30}"
IMGSZ="${4:-640}"
BATCH="${BATCH:-16}"
NAME="${NAME:-${MODEL}_${IMGSZ}}"

free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
busy=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
echo "cuda:$GPU -> ${free} MiB free, ${busy} other compute process(es)"
if [[ "$free" -lt 12000 ]]; then
  echo "refusing: cuda:$GPU has < 12000 MiB free -- someone else is using it" >&2
  exit 1
fi

if [[ "$MODEL" == rtdetr* ]]; then
  # fetched by tools/fetch_rtdetr_weights.sh; the mirror does not carry these and
  # github is slow here, so they are resolved from the project weights dir
  WEIGHTS="$ROOT/weights/${MODEL}.pt"
  [[ -s "$WEIGHTS" ]] || { echo "missing $WEIGHTS -- run tools/fetch_rtdetr_weights.sh" >&2; exit 1; }
else
  WEIGHTS=$(.venv-det/bin/python - "$MODEL" <<'PY'
import sys
from huggingface_hub import hf_hub_download
name = sys.argv[1]
repo = "Ultralytics/YOLO11" if name.startswith("yolo11") else "Ultralytics/YOLOv8"
print(hf_hub_download(repo, name + ".pt"))
PY
) || { echo "could not resolve weights for $MODEL" >&2; exit 1; }
fi
echo "weights: $WEIGHTS"

# offline: stop ultralytics phoning home for update checks and font assets, and keep
# its config dir inside the project instead of ~/.config
export ULTRALYTICS_OFFLINE=True YOLO_OFFLINE=True YOLO_CONFIG_DIR="$ROOT/.ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"
# FRACTION < 1 gives a smoke test: 0.02 trains on ~375 images and still exercises the
# dataset scan, the AMP check, one optimizer loop and a full validation pass.
FRACTION="${FRACTION:-1.0}"
# RT-DETR needs lr0=0.0001 (AdamW); YOLO's default lr0=0.01 diverges it -- a run at
# 0.01 validated zero for 10 straight epochs and early-stopped with best.pt from
# epoch 1, which cost 3.2 GPU-hours to discover. Pass the recipe per model.
LR0="${LR0:-}"
PATIENCE="${PATIENCE:-10}"
lrarg=()
[[ -n "$LR0" ]] && lrarg=(lr0="$LR0")
exec nice -n 10 .venv-det/bin/yolo detect train \
    model="$WEIGHTS" data=det_dataset/data.yaml \
    epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" device="$GPU" fraction="$FRACTION" \
    patience="$PATIENCE" project=det_runs name="$NAME" exist_ok=True \
    "${lrarg[@]+"${lrarg[@]}"}" \
    workers=8 optimizer=auto cos_lr=True seed=0 verbose=True
