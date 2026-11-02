#!/usr/bin/env bash
# Does removing the shared direction from the text prototypes make them usable?
#
# Every frozen text tower measured here is strongly anisotropic: the 114 class
# prototypes share one dominant direction (mean pairwise cosine 0.92-0.97), which
# carries no class information and swamps the differences between types. Centering is
# the standard remedy and costs nothing, so before spending GPU-hours retraining with
# a given prototype set, measure it both ways.
#
#   nohup bash tools/probe_whiten_matrix.sh > runs/probe_whiten.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

GPU="${GPU:-0}"
N="${N:-600}"

run() {  # <npz> <encoder> <label> [extra args]
  echo "##### $3  ($1, $2)"
  nice -n 15 ./.venv/bin/python -u -m tools.probe_text_prototypes \
      --text-features "knowledge/$1.npz" --encoder "$2" --n "$N" --gpu "$GPU" \
      --workers 6 "${@:4}" 2>&1 | grep -aE "pairwise cosine|class-mean|max-sentence"
}

run text_features_clipL_en        clip    "CLIP-L 英文知识库 全类型"
run text_features_clipL_en        clip    "CLIP-L 英文知识库 全类型 + 白化" --whiten
run text_features_clipL_en_diff   clip    "CLIP-L 英文知识库 仅diff"
run text_features_clipL_en_diff   clip    "CLIP-L 英文知识库 仅diff + 白化" --whiten
run text_features_leftpad         siglip2 "SigLIP2 中文知识库 左填充"
run text_features_leftpad         siglip2 "SigLIP2 中文知识库 左填充 + 白化" --whiten
run text_features_names           siglip2 "SigLIP2 仅类名"
run text_features_names           siglip2 "SigLIP2 仅类名 + 白化" --whiten
echo "probe_whiten_matrix done"
