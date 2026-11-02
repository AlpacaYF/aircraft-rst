#!/usr/bin/env python
"""Dump frozen per-modality features from a TRAINED RST checkpoint (fine-tuned
SigLIP2), for the fusion-mode matrix (early/mid/late x R/RS/RT/RST).

  region : mean-pooled encoder tokens of the context-expanded crop   [768]
  scene  : masked mean of the cached pooled scene tokens             [768]
  text   : class prototypes from the paper's text_features (diff)    [114,768]

No augmentation (train=False datasets), no dropout, no gradient. Run:
  .venv/bin/python -m tools.dump_siglip_features --run runs/v2_rs_s0 --gpu 0
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rst.data import EntityDataset, RSTCollator  # noqa: E402
from rst.eval_checkpoint import cfg_from_ckpt  # noqa: E402
from rst.model import RSTModel  # noqa: E402
from rst.train import load_semantic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--run", default="runs/v2_rs_s0")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out-prefix", default="index/siglipft_rs")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = args.run if os.path.isabs(args.run) else os.path.join(root, args.run)
    ck = torch.load(os.path.join(run_dir, args.ckpt), map_location=device,
                    weights_only=False)
    cfg = cfg_from_ckpt(ck, root)
    cfg.num_workers = args.workers
    print("checkpoint %s | scene=%s text=%s" % (run_dir, cfg.use_scene, cfg.use_text))

    protos, _, _ = load_semantic(cfg)
    sd = ck["model"]
    model = RSTModel(cfg, protos, attr_dim=0)
    missing = model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # text modality: masked-mean of the class prototype tokens
    toks = model.proto                       # [C, N, 768]
    mask = model.proto_mask.unsqueeze(-1).float()
    text_feat = (toks * mask).sum(1) / mask.sum(1).clamp(min=1)
    np.savez_compressed(args.out_prefix + "_text.npz",
                        features=text_feat.float().cpu().numpy(),
                        classes=np.array(json.load(open(
                            os.path.join(root, "index", "class_meta.json"),
                            encoding="utf-8"))["classes"], dtype=object))
    print("text features:", tuple(text_feat.shape))

    reg_f, reg_y, reg_ids, reg_sp = [], [], [], []
    scene_feat, scene_ids = {}, []

    for split in [s.strip() for s in args.splits.split(",")]:
        ds = EntityDataset(cfg, split, train=False)
        dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=cfg.num_workers,
                        collate_fn=RSTCollator(ds),
                        pin_memory=(device.type == "cuda"))
        with torch.no_grad():
            for bi, b in enumerate(dl):
                b = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in b.items()}
                with torch.autocast(device_type=device.type, enabled=cfg.amp):
                    rtok, _ = model.region_enc.pooled_tokens(
                        b["region"], cfg.region_query_pool)
                reg_f.append(rtok.mean(1).float().cpu())
                reg_y.append(b["class"].cpu())
                reg_ids.extend(b["instance_ids"])
                reg_sp.extend([split] * len(b["instance_ids"]))
                # scene: masked mean of the cached pooled tokens
                if "scene_tokens" in b:
                    st, meta = b["scene_tokens"], b["scene_tile_meta"]
                    T = max(1, meta.size(1))
                    slots = st.size(1) // T
                    near = meta[..., 2] > 0.5
                    valid = torch.ones(st.size(0), T, slots, dtype=torch.bool,
                                       device=st.device)
                    kf = min(cfg.scene_far_pool, int(round(slots ** 0.5)))
                    valid[..., kf * kf:] = False
                    valid |= near.unsqueeze(-1)
                    valid &= ~(meta[..., 3] > 0.5).unsqueeze(-1)
                    vm = valid.reshape(st.size(0), -1).unsqueeze(-1).float()
                    sf = (st.float() * vm).sum(1) / vm.sum(1).clamp(min=1)
                    for img, vec in zip(b["images"], sf.cpu().numpy()):
                        if img not in scene_feat:
                            scene_feat[img] = vec
                if bi % 50 == 0:
                    print("  %s %d/%d" % (split, (bi + 1) * args.batch, len(ds)),
                          flush=True)

    Xr = torch.cat(reg_f).numpy().astype(np.float32)
    np.savez_compressed(args.out_prefix + "_region.npz", features=Xr,
                        labels=torch.cat(reg_y).numpy(),
                        ids=np.array(reg_ids),
                        splits=np.array(reg_sp))
    imgs = sorted(scene_feat)
    np.savez_compressed(args.out_prefix + "_scene.npz",
                        features=np.stack([scene_feat[i] for i in imgs]).astype(np.float32),
                        images=np.array(imgs))
    print("region %s | scene %s | text %s"
          % (Xr.shape,
             np.stack([scene_feat[i] for i in imgs]).shape, tuple(text_feat.shape)))
    print("wrote %s_{region,scene,text}.npz" % args.out_prefix)


if __name__ == "__main__":
    main()
