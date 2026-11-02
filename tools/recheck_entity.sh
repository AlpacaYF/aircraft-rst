#!/usr/bin/env bash
# Re-score table 4-7 rows from the SAME boxes file at a SINGLE threshold, with the
# current entity_extract code, to check the published numbers and build the
# matched/unmatched/duplicate table M4 asks for. Inference only.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"
B_YOLO="det_runs/yolo11x_640_b16_test_boxes.json"
B_RTDETR="det_runs/rtdetr-l_640_test_boxes.json"
MODELS="runs/v2_rs_s0,runs/v2_rt_adapt_s0,runs/v2_rst_f_s0"

one() {
  local g="$1" out="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$g" nice -n 10 "$PY" -m rst.entity_extract --root . "$@" \
      --score 0.5 --gpu 0 --workers 4 --out "det_runs/$out" \
      > "runs/recheck_${out%.json}.log" 2>&1
  echo "[$(date +%H:%M:%S)] $out rc=$?"
}

one 1 recheck_yolo_rst.json  --boxes "$B_YOLO" --arch rst --run runs/align_full_kb &
p1=$!
one 2 recheck_yolo_dedf.json --boxes "$B_YOLO" --arch dedf --models "$MODELS" &
p2=$!
one 3 recheck_yolo_swin.json --boxes "$B_YOLO" --arch swin_fair --run runs/baseline_swin_b_fair384 &
p3=$!
wait "$p1" "$p2" "$p3"

one 1 recheck_rtdetr_rst.json  --boxes "$B_RTDETR" --arch rst --run runs/align_full_kb &
p1=$!
one 2 recheck_rtdetr_dedf.json --boxes "$B_RTDETR" --arch dedf --models "$MODELS" &
p2=$!
one 3 recheck_rtdetr_swin.json --boxes "$B_RTDETR" --arch swin_fair --run runs/baseline_swin_b_fair384 &
p3=$!
wait "$p1" "$p2" "$p3"
echo "[$(date +%H:%M:%S)] ALL DONE"
