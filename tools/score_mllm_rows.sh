#!/usr/bin/env bash
# Score the two rows of 表4-3 whose detector is not an ultralytics model:
#   Grounding DINO + VLM   (open-vocabulary detector, recogniser is a separate VLM)
#   Qwen3-VL               (one grounding model does both)
#
# Both are measured under the same uniform candidate budget as the YOLO and RT-DETR
# rows -- at most TOPK boxes per image, ranked by the detector's own confidence --
# because Grounding DINO's scores top out at 0.58 and Qwen3-VL emits none, so a
# confidence floor would rank their calibration rather than their capability.
#
# Run it any time: a row is skipped until its upstream file exists.
set -uo pipefail
cd "$(dirname "$0")/.."
TOPK=${TOPK:-3}
RST=.venv/bin/python        # entity metrics: no GPU, no model load
DET=.venv-det/bin/python    # pycocotools lives here
export CUDA_VISIBLE_DEVICES=

echo "=== Grounding DINO + VLM ==="
if [ -f det_runs/gdino_tiny_test_boxes_vlm.json ]; then
  nice -n 19 $RST -u -m rst.entity_extract \
      --boxes det_runs/gdino_tiny_test_boxes.json \
      --pred-json det_runs/gdino_tiny_test_boxes_vlm.json \
      --score 0 --max-per-image "$TOPK" \
      --out "det_runs/entity_gdino_top$TOPK.json" 2>&1 | tail -8
else
  echo "  det_runs/gdino_tiny_test_boxes_vlm.json not written yet"
fi

echo "=== Qwen3-VL ==="
if [ -f det_runs/qwen3vl_test_boxes.json ]; then
  # emission order is the model's own confidence ranking (tools/coco_ap.py)
  nice -n 19 $DET -u -m tools.coco_ap --boxes det_runs/qwen3vl_test_boxes.json --rank-score
  # ALL boxes it emitted, no top-k trim: it averages 1.5 boxes/image on its own, which
  # is already at the real density, so trimming would only punish it for emitting few.
  nice -n 19 $RST -u -m rst.entity_extract \
      --boxes det_runs/qwen3vl_test_boxes.json \
      --pred-json det_runs/qwen3vl_test_boxes.json \
      --score 0 --max-per-image 0 \
      --out "det_runs/entity_qwen_all.json" 2>&1 | tail -8
else
  echo "  det_runs/qwen3vl_test_boxes.json not written yet"
fi
