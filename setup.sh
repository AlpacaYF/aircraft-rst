#!/usr/bin/env bash
# Isolated environment for the RST project on a shared lab server.
#
# Everything lives under this project directory:
#   .venv/   the Python environment (created with python3 -m venv, no conda)
#   .cache/  pip, HuggingFace and torch caches -- nothing is written to ~/.cache
#
# It never runs "pip install --user", never activates or modifies a conda
# environment, and never edits your shell rc file. Uninstall = rm -rf .venv .cache
#
# Usage:
#   bash setup.sh                  # auto-detect CUDA from nvidia-smi
#   bash setup.sh --torch cu121    # force a wheel variant (cpu|cu118|cu121|cu124)
#   bash setup.sh --dry-run        # show what it would do

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_VARIANT="auto"
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torch)    TORCH_VARIANT="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
    *)          echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

VENV="$ROOT/.venv"
CACHE="$ROOT/.cache"
PYBIN="${PYTHON:-python3}"

# ---- pick a PyPI mirror ------------------------------------------------------
# pypi.org is reachable from many campus networks but drops connections
# mid-download, which kills `pip install` for a 800 MB torch wheel. Prefer a
# domestic mirror when one answers, and always retry hard.
PIP_MIRROR="${PIP_MIRROR:-auto}"
if [[ "$PIP_MIRROR" == "auto" ]]; then
  for m in https://pypi.tuna.tsinghua.edu.cn/simple \
           https://mirrors.aliyun.com/pypi/simple \
           https://pypi.org/simple; do
    if curl -sS -m 12 -o /dev/null "$m/" 2>/dev/null; then PIP_MIRROR="$m"; break; fi
  done
  PIP_MIRROR="${PIP_MIRROR:-https://pypi.org/simple}"
fi
PIP_OPTS=(--no-input --retries 20 --timeout 120 --index-url "$PIP_MIRROR")
echo "pypi mirror: $PIP_MIRROR"

if [[ ! -f "$ROOT/requirements.txt" ]]; then
  echo "requirements.txt not found next to setup.sh" >&2; exit 1
fi

# ---- pick the torch wheel ----------------------------------------------------
if [[ "$TORCH_VARIANT" == "auto" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    MAJ=$(printf '%s' "$DRV" | cut -d. -f1)
    echo "nvidia-smi reports driver $DRV"
    if [[ "${MAJ:-0}" -ge 525 ]]; then TORCH_VARIANT=cu124
    elif [[ "${MAJ:-0}" -ge 450 ]]; then TORCH_VARIANT=cu121
    elif [[ "${MAJ:-0}" -ge 418 ]]; then TORCH_VARIANT=cu118
    else TORCH_VARIANT=cpu; fi
  else
    echo "no nvidia-smi -> CPU wheels"
    TORCH_VARIANT=cpu
  fi
fi
echo "torch variant: $TORCH_VARIANT"

case "$TORCH_VARIANT" in
  cpu)   INDEX="https://download.pytorch.org/whl/cpu" ;;
  cu118) INDEX="https://download.pytorch.org/whl/cu118" ;;
  cu121) INDEX="https://download.pytorch.org/whl/cu121" ;;
  cu124) INDEX="https://download.pytorch.org/whl/cu124" ;;
  *)     echo "bad --torch value: $TORCH_VARIANT" >&2; exit 1 ;;
esac

if [[ $DRY -eq 1 ]]; then
  echo "would create venv at $VENV"
  echo "would install requirements.txt with --index-url $INDEX"
  echo "caches would go to $CACHE/{pip,hf,torch,tmp}"
  exit 0
fi

# ---- create the venv ---------------------------------------------------------
# Minimal cloud images often lack python3-venv, so `python3 -m venv` fails with
# "ensurepip is not available". Probe for any interpreter that can actually
# create a self-contained venv; the resulting .venv is independent of whichever
# interpreter bootstrapped it.
pick_python() {
  local cands=("$PYBIN")
  [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]] && cands+=("$CONDA_PREFIX/bin/python")
  for p in /data/miniconda/bin/python /opt/conda/bin/python \
           "$HOME/miniconda3/bin/python" "$HOME/anaconda3/bin/python" \
           /usr/local/bin/python3.12 /usr/local/bin/python3.11 python3.12 python3.11 python3.10; do
    command -v "$p" >/dev/null 2>&1 && cands+=("$p")
  done
  for p in "${cands[@]}"; do
    "$p" -c 'import ensurepip' >/dev/null 2>&1 && { echo "$p"; return 0; }
  done
  return 1
}

if [[ ! -d "$VENV" ]]; then
  BOOTSTRAP_PY=$(pick_python) || {
    echo "No interpreter on this machine can create a venv." >&2
    echo "Either install the venv support for your system python:" >&2
    echo "    sudo apt-get install -y python3.10-venv" >&2
    echo "or point setup.sh at a conda/miniconda python:" >&2
    echo "    PYTHON=/path/to/conda/bin/python bash setup.sh" >&2
    exit 1
  }
  echo "bootstrapping venv with: $BOOTSTRAP_PY ($($BOOTSTRAP_PY -V 2>&1))"
  "$BOOTSTRAP_PY" -m venv "$VENV"
  echo "created venv: $VENV"
else
  echo "reusing existing venv: $VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# ---- keep every cache inside the project ------------------------------------
mkdir -p "$CACHE"/{pip,hf,torch,tmp}
export PIP_CACHE_DIR="$CACHE/pip"
export HF_HOME="$CACHE/hf"
export TORCH_HOME="$CACHE/torch"
export TMPDIR="$CACHE/tmp"

python -m pip install --no-input --retries 20 --timeout 120 --only-binary :all: \
  --index-url "$INDEX" --extra-index-url "$PIP_MIRROR" \
  torch==2.4.1 torchvision==0.19.1
python -m pip install "${PIP_OPTS[@]}" -r "$ROOT/requirements.txt"

# ---- a project-local env file, nothing global --------------------------------
cat > "$ROOT/env.sh" <<EOF
# Generated by setup.sh. Source this before running anything:  source env.sh
# It only affects the current shell; it is not added to any rc file.
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:\$PATH"
export PYTHONUNBUFFERED=1
export PIP_CACHE_DIR="$CACHE/pip"
export HF_HOME="$CACHE/hf"
export TORCH_HOME="$CACHE/torch"
export TMPDIR="$CACHE/tmp"
# huggingface.co is unreachable from many campus networks; the mirror is used
# unless you already set HF_ENDPOINT yourself.
export HF_ENDPOINT="\${HF_ENDPOINT:-https://hf-mirror.com}"
# Reduce allocator fragmentation, which matters on an 8 GB card where a few
# hundred MiB of unusable gaps is the difference between running and OOM.
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Be a good neighbour on a shared box: do not grab every core.
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
# Keep later "pip install" calls on the mirror that worked during setup.
export PIP_INDEX_URL="$PIP_MIRROR"
export PIP_RETRIES=20
export PIP_DEFAULT_TIMEOUT=120
EOF

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print("  cuda:%d %s  %.1f/%.1f GB free" % (i, torch.cuda.get_device_name(i),
                                                   free / 1e9, total / 1e9))
PY

echo
echo "Done. Then run:"
echo "  source env.sh"
echo "  python -m rst.check_env --download-model"
