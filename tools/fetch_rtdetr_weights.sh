#!/usr/bin/env bash
# Get ultralytics-format RT-DETR weights onto this box.
#
# 表4-8 names RT-DETR as the main detector, and reusing ultralytics' trainer for it is
# worth a lot: the same code path, the same augmentation and the same COCO-style AP
# evaluation as the YOLO row, so the two rows are directly comparable. That needs
# rtdetr-l.pt, which ultralytics normally fetches from github releases.
#
# This box reaches github at ~25 KB/s and hf-mirror intermittently, so try the mirror
# first and fall back to a patient github pull. Everything lands inside the project.
#
#   nohup bash tools/fetch_rtdetr_weights.sh > runs/fetch_rtdetr_weights.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
[[ -f env.sh ]] && source ./env.sh
mkdir -p weights

for name in rtdetr-l rtdetr-x; do
  if [[ -s "weights/${name}.pt" ]]; then echo "already have weights/${name}.pt"; continue; fi
  if .venv-det/bin/python - "$name" <<'PY'
import sys
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download("Ultralytics/RT-DETR", sys.argv[1] + ".pt")
shutil.copy(p, "weights/%s.pt" % sys.argv[1])
print("mirror ok")
PY
  then echo "[$(date +%T)] ${name}.pt from hf-mirror"; continue; fi

  echo "[$(date +%T)] mirror failed for ${name}; pulling from github (slow, ~25 KB/s)"
  curl -sSL --retry 8 --retry-delay 20 --max-time 5400 -o "weights/${name}.pt" \
      "https://github.com/ultralytics/assets/releases/download/v8.3.0/${name}.pt" \
    && echo "[$(date +%T)] ${name}.pt from github ($(du -h weights/${name}.pt | cut -f1))" \
    || echo "[$(date +%T)] FAILED to fetch ${name}.pt"
done

# ultralytics refuses a truncated checkpoint with a confusing error, so verify the file
# actually loads before declaring success
.venv-det/bin/python - <<'PY'
import os
for n in ("rtdetr-l", "rtdetr-x"):
    p = "weights/%s.pt" % n
    if not os.path.exists(p):
        continue
    try:
        from ultralytics import RTDETR
        RTDETR(p)
        print("VERIFIED %s loads (%.1f MB)" % (p, os.path.getsize(p) / 1e6))
    except Exception as exc:
        print("UNUSABLE %s: %s" % (p, exc))
        os.remove(p)
PY
echo "fetch_rtdetr_weights done"
