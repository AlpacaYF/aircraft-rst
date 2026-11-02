#!/usr/bin/env bash
# A SECOND project-local venv for the detection and MLLM rows of tables 4-3 / 4-8.
#
# Why not just install into .venv: the training environment is pinned at
# transformers 4.49.0, which cannot load Qwen3-VL, and upgrading it in place would
# (a) risk the jobs that are running and (b) change how the frozen SigLIP 2 text
# tower is called, silently invalidating knowledge/text_features.npz and everything
# 3.3.2/3.3.3 derived from it. So the new stack lives in .venv-det and the two never
# touch. Everything stays inside /home/wangyangfan/dmt/aircraft_rst.
#
#   bash tools/setup_det_env.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
[[ -f env.sh ]] && source ./env.sh

PY="${SYSTEM_PYTHON:-python3}"
if [[ ! -x .venv-det/bin/python ]]; then
  echo "creating .venv-det from $($PY -V 2>&1)"
  "$PY" -m venv .venv-det
fi
V=.venv-det/bin
"$V/pip" install -U pip wheel
# torch first and pinned to the CUDA build the driver supports, so ultralytics does
# not pull a CPU-only or mismatched wheel behind our back
"$V/pip" install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url "${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
"$V/pip" install ultralytics "transformers>=4.57" accelerate qwen-vl-utils pycocotools

echo
echo "=== versions ==="
"$V/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
import ultralytics
print("ultralytics", ultralytics.__version__)
PY
echo "=== weight download reachability ==="
for u in https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt \
         https://hf-mirror.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -L "$u" || echo FAIL)
  echo "  $code  $u"
done
echo "setup_det_env done"
