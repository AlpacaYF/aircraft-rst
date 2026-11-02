#!/usr/bin/env python
"""Pure recognition-time profiler over the whole test split.

Same boxes and same per-image loop as the integrated pipeline, but ONLY the
recognition stage is timed (image read -> region/scene build -> forward ->
logits), so it can be added to the measured detection time for an end-to-end
per-image number. Supported recognisers:

  rst   : one RST model (default v2_rs_s0)          -> per-image time
  swin  : the fair-384 Swin-B baseline classifier   -> per-image time
  dedf  : three RST experts + logit average         -> per-image time
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rst.config import Config                                    # noqa: E402
from rst.data import RSTCollator                                 # noqa: E402
from rst.entity_extract import DetectionSet, load_rst_expert     # noqa: E402
from rst.eval_checkpoint import cfg_from_ckpt                    # noqa: E402
from rst.train import load_semantic                              # noqa: E402
from tools.infer_image import build_items_for_image, ImageStore  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boxes", default="det_runs/testset_boxes.json")
    ap.add_argument("--arch", default="rst", choices=["rst", "swin", "dedf"])
    ap.add_argument("--run", default="runs/v2_rs_s0")
    ap.add_argument("--experts", default="runs/v2_rs_s0,runs/v2_rt_adapt_s0,runs/v2_rst_f_s0")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)
    device = torch.device("cuda:%s" % args.gpu if torch.cuda.is_available() else "cpu")
    meta = json.load(open("index/class_meta.json", encoding="utf-8"))
    det = json.load(open(args.boxes))["images"]

    cfg = Config()
    cfg.root = root
    cfg.use_mask_channel = True
    cfg.use_scene_cache = False
    cfg.num_workers = 0

    if args.arch == "swin":
        from rst.baselines import RegionCollator, VisionBaseline
        ck = torch.load(os.path.join(root, "runs/baseline_swin_b_fair384/best.pt"),
                        map_location=device, weights_only=False)
        model = VisionBaseline("swin_b", len(meta["classes"]), dropout=0.1,
                               in_channels=int(ck.get("in_channels", 4)),
                               input_size=int(ck.get("native_input", 384)))
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        models = [(model, cfg)]
        collate = lambda ds: RegionCollator(ds, mask_channel=True)
        print("recogniser: swin_b fair384")
    else:
        dirs = args.experts.split(",") if args.arch == "dedf" else [args.run]
        models = []
        for mdir in dirs:
            md = mdir if os.path.isabs(mdir) else os.path.join(root, mdir)
            p = torch.load(os.path.join(md, "best.pt"), map_location=device,
                           weights_only=False)
            c = cfg_from_ckpt(p, root)
            c.use_scene_cache = False
            pr, _, _ = load_semantic(c)
            models.append((load_rst_expert(p, c, pr, device, tag=os.path.basename(md)), c))
        collate = RSTCollator
        print("recogniser: %s" % args.arch, [os.path.basename(d) for d in dirs])

    times, n_ent = [], 0
    for ii, (key, dv) in enumerate(sorted(det.items())):
        boxes = dv["boxes"]
        if not boxes:
            continue
        t0 = time.time()
        if args.arch == "swin":
            items, order, (W, H), kk = build_items_for_image(root, dv["file"], boxes, cfg)
            store = ImageStore(meta, kk, dv["file"], W, H, boxes)
            ds = DetectionSet(cfg, store, items)
            with torch.no_grad():
                loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                                    collate_fn=collate(ds))
                for b in loader:
                    b = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in b.items()}
                    models[0][0](b)
        else:
            items, order, (W, H), kk = build_items_for_image(root, dv["file"], boxes, cfg)
            store = ImageStore(meta, kk, dv["file"], W, H, boxes)
            ds = DetectionSet(cfg, store, items)
            with torch.no_grad():
                loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                                    collate_fn=collate(ds))
                for b in loader:
                    b = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in b.items()}
                    for m, _ in models:
                        m(b)
        times.append(time.time() - t0)
        n_ent += len(boxes)
        if ii % 200 == 0:
            print("  %d  %.3fs" % (ii, times[-1]), flush=True)

    st = sorted(times)
    n = len(st)
    res = {"arch": args.arch, "n_images": n, "n_entities": n_ent,
           "mean": round(sum(st) / n, 4), "median": round(st[n // 2], 4),
           "p90": round(st[int(n * 0.9)], 4), "min": round(st[0], 4),
           "max": round(st[-1], 4)}
    print(json.dumps(res, indent=2))
    out = args.out or ("runs/time_recog_%s.json" % args.arch)
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
