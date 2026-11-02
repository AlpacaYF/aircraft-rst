#!/usr/bin/env bash
# Build the upload bundle for the lab server: code + index + knowledge, no data
# and no local caches. Run this from Git Bash on Windows.
#
#   bash package_for_server.sh              # -> ../aircraft_rst_code.tar.gz
#   bash package_for_server.sh --with-index # also include index/*.jsonl (default)
#   bash package_for_server.sh --slim       # code only, regenerate index on server

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(dirname "$ROOT")"
NAME="$(basename "$ROOT")"
OUT="$PARENT/${NAME}_rst_code.tar.gz"
SLIM=0
[[ "${1:-}" == "--slim" ]] && SLIM=1

EXCLUDES=(
  --exclude "$NAME/.venv"
  --exclude "$NAME/.cache"
  --exclude "$NAME/runs"
  --exclude "$NAME/__pycache__"
  --exclude "$NAME/rst/__pycache__"
  --exclude "$NAME/scripts/__pycache__"
  --exclude "$NAME/dataset"
  --exclude "$NAME/crop"
  --exclude "$NAME/models"
  --exclude "$NAME/*.docx"
  --exclude "$NAME/~\$*"
  --exclude "$NAME/*.tar.gz"
)
if [[ $SLIM -eq 1 ]]; then
  EXCLUDES+=(--exclude "$NAME/index/instances.jsonl" --exclude "$NAME/index/images.jsonl")
fi

cd "$PARENT"
tar "${EXCLUDES[@]}" -czf "$(basename "$OUT")" "$NAME"

echo "wrote $OUT"
ls -la "$OUT" | awk '{printf "size: %.1f MB\n", $5/1e6}'
echo
echo "contents:"
tar -tzf "$OUT" | sed "s|^$NAME/||" | grep -v '/$' | sort | head -40
echo
if [[ $SLIM -eq 1 ]]; then
  echo "SLIM bundle: on the server rebuild the index first with"
  echo "  python scripts/build_dataset_index.py && python scripts/build_metadata.py"
  echo "(needs labels_with_split.csv and dataset/ present)"
else
  echo "Bundle includes index/ and knowledge/. Only dataset/ needs to be present"
  echo "on the server (or symlinked) before running."
fi
