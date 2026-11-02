#!/usr/bin/env bash
# watcher: wait for both encoders to finish, then run the real fusion matrix
cd ~/dmt/aircraft_rst
echo "[$(date +%H:%M:%S)] watcher started"
while :; do
  n=$(ps -u wangyangfan -o pid,args | grep emb_probe_encode | grep -v grep | wc -l)
  [ "$n" = "0" ] && break
  sleep 300
done
echo "[$(date +%H:%M:%S)] encoders done; files:"
ls -la index/emb_qwen3vle_region_all.npz index/emb_qwen3vle_scene_all.npz

source ./env.sh
CUDA_VISIBLE_DEVICES=3 nice -n 10 .venv/bin/python -u -m tools.emb_probe_fusion_matrix \
    --region index/emb_qwen3vle_region_all.npz \
    --scene index/emb_qwen3vle_scene_all.npz \
    --text index/emb_qwen3vle_text_classes.npz \
    --out runs/emb_fusion_matrix_full.json \
    > runs/emb_fusion_matrix_full.log 2>&1
echo "[$(date +%H:%M:%S)] fusion matrix rc=$?"
tail -30 runs/emb_fusion_matrix_full.log
