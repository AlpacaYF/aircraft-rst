#!/usr/bin/env bash
# Test-split dumps for the review-response batch (2026-10-19 runs). Three
# workers, one per idle card (cuda:1/2/3; cuda:0 belongs to another user and is
# never touched). Also produces the multi-scale TTA control for v2_rs_s0.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"

queue() {
  local g="$1"; shift
  for r in "$@"; do
    echo "[$(date +%H:%M:%S)] gpu$g dump $r"
    CUDA_VISIBLE_DEVICES="$g" nice -n 10 "$PY" -m rst.eval_checkpoint \
        --root . --run "runs/$r" --ckpt best.pt --split test --dump-samples \
        --workers 4 --gpu 0 > "runs/dump_${r}_review.log" 2>&1
    echo "[$(date +%H:%M:%S)] gpu$g done  $r (rc=$?)"
  done
}

queue 1 v2_ro_s1 v2_rs_s1 v2_rt_adapt_s1 v2_rst_f_s1 v2_rs_nosup_s0 &
p1=$!
queue 2 v2_ro_s2 v2_rs_s2 v2_rt_adapt_s2 v2_rst_f_s2 v2_rt_nosup_s0 &
p2=$!
queue 3 v2_ro_noaug_s2 v2_rst_nosup_s0 v2_rst_wide_s0 &
p3=$!
wait "$p1" "$p2" "$p3"

# ---- multi-scale TTA on the RS s0 model (inference-cost-matched control) ----
cd "$ROOT" || exit 1
D=runs/v2_rs_s0
if [[ -f "$D/samples_test.npz" ]]; then
  cp -n "$D/samples_test.npz" "$D/samples_test_exp025.npz"
  cp -n "$D/eval_test.json" "$D/eval_test_default.json"
fi
for e in 0.15 0.35; do
  tag=$(echo "$e" | tr -d '.')
  echo "[$(date +%H:%M:%S)] gpu3 TTA expand=$e -> $D/samples_test_exp${tag}.npz"
  CUDA_VISIBLE_DEVICES=3 nice -n 10 "$PY" -m rst.eval_checkpoint \
      --root . --run "$D" --ckpt best.pt --split test --dump-samples \
      --region-expand "$e" --out "$D/eval_test_exp${tag}.json" \
      --workers 4 --gpu 0 > "runs/tta_rs_exp${tag}.log" 2>&1
  [[ -f "$D/samples_test.npz" ]] && mv "$D/samples_test.npz" "$D/samples_test_exp${tag}.npz"
  echo "[$(date +%H:%M:%S)] gpu3 TTA done expand=$e"
done
# restore the default-expansion dump as the canonical samples_test.npz
[[ -f "$D/samples_test_exp025.npz" ]] && cp "$D/samples_test_exp025.npz" "$D/samples_test.npz"
echo "[$(date +%H:%M:%S)] ALL DONE"
