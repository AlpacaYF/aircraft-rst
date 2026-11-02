#!/usr/bin/env bash
# Complete the 2(tower) x 2(language) x 2(sentence-kind) x 2(anchor) factorial at the
# PROBE level, so the text-tower decision costs minutes instead of GPU-hours.
#
# Why: full_cliptext_diff (88.37 Top-1 / 85.87 Macro-F1) changed three things at once
# relative to the archived full model -- tower SigLIP2->CLIP-L, language zh->en, and
# sentence kinds all->diff. Without the SigLIP2 x English cells there is no way to say
# which of those produced the gain, and 4.3 of the thesis states the text encoder is
# SigLIP 2. The missing cells are built and scored here; no training is involved.
#
#   nohup bash tools/probe_text_matrix.sh > runs/probe_text_matrix.log 2>&1 &
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

GPU="${GPU:-0}"
EN=knowledge/kb_strings_en.json
PY=./.venv/bin/python

build() {  # <out> [extra build args...]
  local out="$1"; shift
  if [[ -s "$out" ]]; then echo "[skip] $out exists"; return; fi
  echo "[build] $out  ($*)"
  nice -n 15 $PY -u -m rst.build_text_features --text-encoder siglip2 --strings "$EN" \
      --out "$out" --batch 64 "$@" 2>&1 | grep -aE "restricted|anchored|text tokens|wrote"
}

probe() {  # <npz> <encoder> <label>
  echo
  echo "##### $3"
  nice -n 15 $PY -u -m tools.probe_text_prototypes --text-features "$1" --encoder "$2" \
      --n 600 --gpu "$GPU" --workers 6 2>&1 \
    | grep -aE "sentences each|pairwise cosine|class-mean|max-sentence"
}

build knowledge/text_features_siglip2_en.npz
build knowledge/text_features_siglip2_en_diff.npz          --proto-kinds diff
build knowledge/text_features_siglip2_en_anchored.npz      --anchor-names
build knowledge/text_features_siglip2_en_diff_anchored.npz --proto-kinds diff --anchor-names

probe knowledge/text_features_siglip2_en.npz              siglip2 "P1 SigLIP2 · 英 · 全部句类"
probe knowledge/text_features_siglip2_en_diff.npz         siglip2 "P2 SigLIP2 · 英 · 仅diff"
probe knowledge/text_features_siglip2_en_anchored.npz     siglip2 "P3 SigLIP2 · 英 · 全部 · 类名锚定"
probe knowledge/text_features_siglip2_en_diff_anchored.npz siglip2 "P3b SigLIP2 · 英 · 仅diff · 类名锚定"
probe knowledge/text_features_clipL_en_desc_diff.npz      clip    "P4 CLIP-L · 英 · desc+diff（补测已训练过的那版）"

echo
echo "decision rule: any SigLIP2 row >= 8% -> keep SigLIP2 (4.3 unchanged);"
echo "               all <= 3%             -> the tower is the binding constraint, use CLIP-L for text"
echo "probe_text_matrix done"
