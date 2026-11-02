#!/usr/bin/env bash
# Run the 表4-3 Qwen3-VL pass as two shards, one per free GPU, and merge when both land.
#
# A single process needs ~1669 x 5.4 s = 150 min; two shards finish in ~80. Each shard
# checkpoints every 25 images and starts with --resume, so a preemption or a crash costs
# the tail rather than the run -- that is what lost the previous 2 hours (one image,
# 448x280, was the only one on the 28-px grid and it killed the loop before anything
# had been written).
#
# The merge writes det_runs/qwen3vl_test_boxes.json, which is the file
# tools/watch_and_score_mllm.sh is waiting on, so scoring happens without another step.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv-det/bin/python
SHARD_DIR=det_runs

for g in 2 3; do
  free=$($PY -c "import torch;print(int(torch.cuda.mem_get_info('cuda:$g')[0]/2**20))" 2>/dev/null || echo 0)
  if [ "$free" -lt 18000 ]; then
    echo "GPU$g has only ${free} MiB free -- another user is on it, aborting before launching"
    exit 1
  fi
done

nice -n 19 $PY -u -m tools.qwen_ground --gpu 2 --tag qwen3vl \
      --num-shards 2 --shard 0 --resume > runs/qwen_s0.log 2>&1 &
p0=$!
nice -n 19 $PY -u -m tools.qwen_ground --gpu 3 --tag qwen3vl \
      --num-shards 2 --shard 1 --resume > runs/qwen_s1.log 2>&1 &
p1=$!
echo "launched shard0(pid $p0, GPU2) shard1(pid $p1, GPU3)"

wait $p0; r0=$?
wait $p1; r1=$?
echo "shards exited: $r0 $r1"
if [ "$r0" != "0" ] || [ "$r1" != "0" ]; then
  echo "NOT merged -- a shard failed, rerun this script (it resumes from the checkpoints)"
  exit 1
fi
$PY -u -m tools.qwen_ground --tag qwen3vl --merge "$SHARD_DIR/qwen3vl_test_boxes_shard*.json"
