#!/usr/bin/env bash
# Run one command as soon as a GPU is genuinely free, then exit.
#
# Exists because "the card was free when I looked" is not the same as "the card was
# free when my job started": today a re-scoring job was killed by another user's
# 20 GB process that appeared between the check and the allocation, and a matrix
# launcher put two of our own jobs on one card for the same reason. So the check
# happens immediately before exec, and it requires BOTH zero compute processes and
# enough free memory.
#
#   nohup bash tools/queue_run.sh 12000 -- python -u -m rst.train --name x ... \
#       > runs/x.log 2>&1 &
#
# Everything after `--` is the command; it runs inside the project with env.sh sourced
# and at nice 10, and CUDA_VISIBLE_DEVICES is set to the card that was chosen (so the
# command should ask for cuda:0).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
# shellcheck disable=SC1091
source ./env.sh

MIN_FREE_MIB="${1:-12000}"; shift || true
[[ "${1:-}" == "--" ]] && shift
if [[ $# -eq 0 ]]; then echo "usage: queue_run.sh <min_free_mib> -- <command...>" >&2; exit 2; fi

while :; do
  for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    free=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    procs=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    if [[ "$procs" == "0" && "$free" -ge "$MIN_FREE_MIB" ]]; then
      # Re-check after a pause: more than one queue can be waiting for the same card,
      # and two jobs landing on one GPU is exactly what the polling is meant to avoid.
      # The wait has to outlast CUDA context registration -- a job that has been
      # "started" still shows no compute process for its first 60-90 s (weight loading
      # and dataset scan happen on the CPU), which is how three runs ended up sharing
      # one card while three others sat idle.
      sleep 150
      procs2=$(nvidia-smi -i "$i" --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
      free2=$(nvidia-smi -i "$i" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
      if [[ "$procs2" != "0" || "$free2" -lt "$MIN_FREE_MIB" ]]; then
        echo "[$(date +%T)] cuda:$i taken while waiting (${procs2} proc, ${free2} MiB); retrying"
        continue
      fi
      echo "[$(date +%T)] cuda:$i free (${free2} MiB, no compute process) -> $*"
      export CUDA_VISIBLE_DEVICES="$i"
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      exec nice -n 10 "$@"
    fi
  done
  echo "[$(date +%T)] no card with >= ${MIN_FREE_MIB} MiB free and no compute process; waiting"
  sleep 120
done
