#!/usr/bin/env bash
# Pack the deployable artifacts for offline use (~3.7 GB):
#   bash deploy/export_models.sh
# Produces deploy/export/rst_dedf_demo_models_<date>.tar.gz
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
mkdir -p deploy/export
TAR="deploy/export/rst_dedf_demo_models_$(date +%Y%m%d).tar.gz"
tar -czf "$TAR" \
  runs/detect/det_runs/yolo11x_640_b16/weights/best.pt \
  runs/v2_rs_s0/best.pt \
  runs/v2_rt_nosup_s0/best.pt \
  runs/v2_rst_nosup_s0/best.pt \
  knowledge/text_features_clipL_en_diff.npz \
  index/class_meta.json \
  deploy/demo.py deploy/README.md deploy/manifest.json
ls -lh "$TAR"
echo "extract and run:  .venv/bin/python deploy/demo.py --image <img> --root ."
