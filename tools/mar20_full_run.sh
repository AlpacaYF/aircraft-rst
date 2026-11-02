#!/usr/bin/env bash
# Assemble the FULL MAR20 test split (2,511 images) from the subset zip plus the
# remaining-images zip, then run GT-box classification, detection and end-to-end.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"
DET_PY="$ROOT/.venv-det/bin/python"
MODELS="runs/v2_rs_s0,runs/v2_rt_nosup_s0,runs/v2_rst_nosup_s0"

rm -rf mar20_full
mkdir -p mar20_full
unzip -oq mar20_subset.zip -d mar20_full
unzip -oq mar20_rest.zip -d mar20_full

n_img=$(ls mar20_full/JPEGImages/*.jpg 2>/dev/null | wc -l)
n_xml=$(ls "mar20_full/Annotations/Horizontal Bounding Boxes"/*.xml 2>/dev/null | wc -l)
echo "assembled: images=$n_img xml=$n_xml"
if [[ "$n_img" -ne 2511 || "$n_xml" -ne 2511 ]]; then
  echo "ERROR: expected 2511/2511" >&2
  exit 1
fi

echo "[$(date +%H:%M:%S)] GT-box classification"
CUDA_VISIBLE_DEVICES=1 nice -n 10 "$PY" tools/mar20_eval.py --mar20 mar20_full \
  --ids-file mar20_full/test.txt --arch dedf --models "$MODELS" --gpu 0 --batch 32 \
  --out mar20/results/full_dedf > runs/mar20_full_gt.log 2>&1 &
p1=$!

echo "[$(date +%H:%M:%S)] detection"
CUDA_VISIBLE_DEVICES=2 nice -n 10 "$DET_PY" tools/mar20_detect.py --mar20 mar20_full \
  --ids-file mar20_full/test.txt --out mar20/results/full_boxes.json --conf 0.5 \
  > runs/mar20_full_det.log 2>&1 &
p2=$!

wait "$p1"; r1=$?
wait "$p2"; r2=$?
echo "[$(date +%H:%M:%S)] GT rc=$r1 det rc=$r2"

echo "[$(date +%H:%M:%S)] end-to-end"
CUDA_VISIBLE_DEVICES=1 nice -n 10 "$PY" tools/mar20_eval.py --mar20 mar20_full \
  --ids-file mar20_full/test.txt --arch dedf --models "$MODELS" --gpu 0 --batch 32 \
  --det-json mar20/results/full_boxes.json --out mar20/results/full_e2e \
  > runs/mar20_full_e2e.log 2>&1
echo "[$(date +%H:%M:%S)] ALL DONE"
