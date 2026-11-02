#!/usr/bin/env bash
# Build the Scene feature cache across every idle GPU, then merge the shards.
#
# 3.5.4 caches one scene encoding per image per orientation. The live path is not
# an option on this data: a single 4x20-tile step peaks at 19.5 GB and runs at
# 4.9 entities/s (1.9 h per epoch), and batch 8 OOMs a 24 GB card.
#
# Usage:  bash tools/build_scene_cache.sh [ngpus]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NG="${1:-4}"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source env.sh
export OMP_NUM_THREADS=8

mkdir -p cache
echo "[$(date +%H:%M:%S)] building the scene cache on $NG GPU(s)"
pids=()
for i in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$i nice -n 5 python -m rst.cache_scene_features \
      --shard "$i" --nshards "$NG" --out-dir cache > "cache/shard$i.log" 2>&1 &
  pids+=($!)
  echo "  shard $i -> pid ${pids[-1]} on GPU $i (log cache/shard$i.log)"
done

fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
if [[ $fail -ne 0 ]]; then
  echo "[$(date +%H:%M:%S)] a shard FAILED -- see cache/shard*.log"
  exit 1
fi
echo "[$(date +%H:%M:%S)] all shards done, merging"
python -m rst.cache_scene_features --merge --nshards "$NG" --out-dir cache || exit 1
rm -f cache/scene_tokens.shard*.bin cache/scene_index.shard*.npz
echo "[$(date +%H:%M:%S)] cache ready:"
ls -la cache/scene_tokens.bin cache/scene_index.npz
df -h "$HOME" | tail -1
