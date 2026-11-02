#!/usr/bin/env bash
# Re-score the fine-tuned detectors under the SAME operating point as the VLM rows.
#
# Entity P/R/F1 needs a confidence floor, and a fixed floor is not comparable across
# detectors: Grounding DINO's highest score on this test set is 0.58, so the 0.5 floor
# used for YOLO/RT-DETR would leave it with almost nothing and the row would measure
# score calibration rather than detection quality. Every row therefore keeps the
# top-3 boxes per image (ground truth averages 1.8), which is detector-agnostic.
#
#   bash tools/score_entity_top3.sh 1
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
GPU="${1:-0}"
K="${K:-3}"

for pair in "yolo11x_640_b16:entity_yolo_top${K}.json" \
            "rtdetr-l_640:entity_rtdetr_top${K}.json"; do
  tag="${pair%%:*}"; out="${pair##*:}"
  boxes="det_runs/${tag}_test_boxes.json"
  [[ -f "$boxes" ]] || { echo "skip $tag (no $boxes)"; continue; }
  echo "==== $tag : top-$K per image, no score floor ===="
  nice -n 10 python -u -m rst.entity_extract --boxes "$boxes" --run runs/align_full_kb \
      --gpu "$GPU" --batch 32 --workers 8 --score 0 --max-per-image "$K" \
      --out "runs/align_full_kb/$out" 2>&1 | grep -aE "detections:|Entity|Detect|Recognition|wrote"
done
echo "score_entity_top3 done"
