#!/usr/bin/env bash
# Start the scene feature cache build detached, with the project environment
# actually applied. Kept as a script because doing it in an inline ssh command
# invites quoting mistakes that silently drop `source env.sh` -- which then
# stalls on huggingface.co retries instead of using the local model cache.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source ./env.sh
echo "HF_ENDPOINT=${HF_ENDPOINT:-<unset>}  HF_HOME=${HF_HOME:-<unset>}"
mkdir -p cache
nohup ./.venv/bin/python -u -m rst.cache_scene_features \
    --batch "${BATCH:-64}" --tile-tokens "${TILE_TOKENS:-16}" \
    >> cache_scene.log 2>&1 < /dev/null &
echo "started pid=$!  log=cache_scene.log"
