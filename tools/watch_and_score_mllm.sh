#!/usr/bin/env bash
# Wait for each of the two 表4-3 jobs to write its output, then score it immediately.
#
# Scoring is CPU-only and nice'd to 19, so it never competes with the GPU jobs that are
# still running. tools/score_mllm_rows.sh skips whatever is not on disk yet, which is why
# it can be run speculatively once per arrival instead of waiting for both.
set -uo pipefail
cd "$(dirname "$0")/.."

for f in det_runs/gdino_tiny_test_boxes_vlm.json det_runs/qwen3vl_test_boxes.json; do
  echo "$(date '+%H:%M') waiting for $f"
  while [ ! -f "$f" ]; do sleep 60; done
  echo "$(date '+%H:%M') $f appeared, scoring"
  sleep 20                      # let the writer finish flushing and close
  bash tools/score_mllm_rows.sh
  echo "$(date '+%H:%M') scored after $f"
done
echo "$(date '+%H:%M') all rows scored"
