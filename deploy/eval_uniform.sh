#!/usr/bin/env bash
# Entity-level metrics for the DEPLOYED (uniform-supervision) recipe, on the
# same YOLO boxes and threshold as the paper's table: score 0.5, IoU 0.5.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"
B="det_runs/yolo11x_640_b16_test_boxes.json"

CUDA_VISIBLE_DEVICES=1 nice -n 10 "$PY" -m rst.entity_extract --root . --boxes "$B" \
  --arch dedf --models runs/v2_rs_s0,runs/v2_rt_nosup_s0,runs/v2_rst_nosup_s0 \
  --score 0.5 --gpu 0 --workers 4 --out det_runs/entity_yolo_uniform_dedf.json \
  > runs/eval_uniform_dedf.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 nice -n 10 "$PY" -m rst.entity_extract --root . --boxes "$B" \
  --arch rst --run runs/v2_rst_nosup_s0 \
  --score 0.5 --gpu 0 --workers 4 --out det_runs/entity_yolo_uniform_rst.json \
  > runs/eval_uniform_rst.log 2>&1 &
p2=$!
wait "$p1"; wait "$p2"
echo ALL DONE
