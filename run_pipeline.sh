#!/usr/bin/env bash
# The full offline chain, in the order the paper needs it. Each step is
# idempotent and prints its own progress.
#
#   bash run_pipeline.sh                 # everything, including training run A
#   bash run_pipeline.sh --skip-train    # stop after the offline artifacts
#   bash run_pipeline.sh --skip-run-b    # do not start the filtered-KB run
#
# 3.3.2 runs AFTER 3.5.2, not before it. A frozen SigLIP 2 has no usable
# text->image signal on these 114 fine-grained types (own-class rank median 56
# against a chance median of 57.5, top-1 0.7%), so verifying the knowledge base
# against frozen features measures noise. The verification therefore uses the
# stage-2 checkpoint, where the contrastive objective of eq. 3-19 has already
# pulled the visual side and the text side into one space.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source ./env.sh

BATCH_EMB="${BATCH_EMB:-256}"
NGPU="${NGPU:-4}"
EPOCHS="${EPOCHS:-30}"
RUN_A="${RUN_A:-align_full_kb}"
RUN_B="${RUN_B:-align_filtered_kb}"
SKIP_TRAIN=0
SKIP_RUN_B=0
for a in "$@"; do
  case "$a" in
    --skip-train) SKIP_TRAIN=1 ;;
    --skip-run-b) SKIP_RUN_B=1 ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done

step() { echo; echo "======== $* ========"; }

step "0/7 preflight"
python -u -m rst.check_env --download-model

# --verdict points at a path that does not exist so this build is unfiltered even
# on a re-run: build_text_features applies knowledge/kb_verification.json whenever
# that file is present, and a stale verdict from an earlier attempt would silently
# delete sentences from what is supposed to be the complete knowledge base.
step "1/7 Text Features from the complete knowledge base (3.3.4)"
python -u -m rst.build_text_features --batch 64 --verdict knowledge/.no_verdict

step "2/7 region embeddings, frozen encoder (input to 3.3.3)"
python -u -m rst.extract_embeddings --split all --batch "$BATCH_EMB" \
  --out index/region_embeddings.npz

step "3/7 mine hard pairs from real visual similarity (3.3.3)"
# must run before verify_knowledge: its hard-negative contrast criterion reads
# index/hard_table.npy, and without it falls back to the weaker rank-1 test
python -u -m rst.mine_hard_pairs --topk 8

step "4/7 Scene feature cache across $NGPU GPU(s) (3.2.3 / 3.5.4)"
# Mandatory, not an optimisation: the live scene path peaks at 19.5 GB for a
# batch of 4x20 tiles and runs at 4.9 entities/s (1.9 h per epoch), and batch 8
# OOMs a 24 GB card.
bash tools/build_scene_cache.sh "$NGPU"

if [[ $SKIP_TRAIN -eq 1 ]]; then
  echo; echo "offline artifacts ready; skipping training"; exit 0
fi

step "5/7 training run A: all three stages on the complete knowledge base"
# One invocation for all three stages on purpose. total_steps is derived from
# len(stages), so splitting stage 3 into a second --resume call would restart the
# cosine schedule with start_step already past its own horizon and collapse the
# learning rate to ~1e-9.
bash run_train.sh --name "$RUN_A" --epochs "$EPOCHS"

STAGE2="runs/$RUN_A/stage2.pt"
if [[ ! -f $STAGE2 ]]; then
  echo "missing $STAGE2 -- cannot run 3.3.2 in the aligned space" >&2
  exit 1
fi

step "6/7 visual consistency verification of the KB (3.3.2), post-3.5.2"
python -u -m rst.extract_embeddings --split all --batch "$BATCH_EMB" --ckpt "$STAGE2" \
  --out index/region_embeddings_stage2.npz
python -u -m rst.verify_knowledge \
  --embeddings index/region_embeddings_stage2.npz \
  --out knowledge/kb_verification.json

# A keep rate at or below the chance baseline means even the aligned encoder
# carries no usable signal for this criterion. Rebuilding on that verdict would
# delete most of the knowledge base on noise, so run B is gated on the filter
# actually beating chance by a clear margin.
step "6b read the 3.3.2 verdict"
VERDICT=knowledge/kb_verification.json
./.venv/bin/python - "$VERDICT" <<'PY' > /tmp/verdict.txt 2>&1 || true
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))["meta"]
keep, chance = m["overall_keep_rate"], m["chance_keep_rate"]
print("keep %.1f%% vs chance %.1f%% (space: %s)" % (100 * keep, 100 * chance, m.get("space", "?")))
print("PASS" if keep >= 1.5 * chance else "SKIP")
PY
head -1 /tmp/verdict.txt

if ! grep -q "^PASS" /tmp/verdict.txt; then
  tail -1 /tmp/verdict.txt
  echo "  Run B SKIPPED: the criterion does not beat its chance baseline, so the"
  echo "  complete knowledge base is kept and run A is the reported result."
  echo "  3.3.2 must then be written up as a negative/diagnostic result."
  exit 0
fi

if [[ $SKIP_RUN_B -eq 1 ]]; then
  echo "verdict is informative; --skip-run-b given, so stopping before run B"
  exit 0
fi

step "7/7 training run B: all three stages on the filtered knowledge base"
echo "3.3.2 filter is informative -- rebuilding Text Features without the dropped sentences"
python -u -m rst.build_text_features --batch 64 --verdict "$VERDICT"
bash run_train.sh --name "$RUN_B" --epochs "$EPOCHS"
echo
echo "compare runs/$RUN_A/summary.json against runs/$RUN_B/summary.json:"
echo "that difference is the measured contribution of 3.3.2."
