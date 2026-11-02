#!/usr/bin/env bash
# Multi-scale TTA for the RS expert. Uses v2_rs_s1 rather than the s0
# checkpoint: s0 predates the sem_gate module, so strict state_dict loading
# fails there. s1 is the same recipe and pairs with DEDF seed 1.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh
PY="$ROOT/.venv/bin/python"
D=runs/v2_rs_s1

# stale files from the failed s0 attempt (mine, not anyone else's)
rm -f runs/v2_rs_s0/samples_test_exp015.npz runs/v2_rs_s0/samples_test_exp025.npz

cp -n "$D/samples_test.npz" "$D/samples_test_default.npz"
for e in 0.15 0.35; do
  tag=$(echo "$e" | tr -d '.')
  echo "[$(date +%H:%M:%S)] TTA expand=$e"
  CUDA_VISIBLE_DEVICES=3 nice -n 10 "$PY" -m rst.eval_checkpoint \
      --root . --run "$D" --ckpt best.pt --split test --dump-samples \
      --region-expand "$e" --out "$D/eval_test_exp${tag}.json" \
      --workers 4 --gpu 0 > "runs/tta_rs_s1_exp${tag}.log" 2>&1
  [[ -f "$D/samples_test.npz" ]] && mv "$D/samples_test.npz" "$D/samples_test_exp${tag}.npz"
done
cp "$D/samples_test_default.npz" "$D/samples_test_exp025.npz"
cp "$D/samples_test_default.npz" "$D/samples_test.npz"
echo "[$(date +%H:%M:%S)] TTA DONE"
ls -la "$D"/samples_test*.npz
