#!/usr/bin/env python
"""Qwen3-VL-Embedding-8B frozen-feature probe: encode aircraft regions.

Encodes the full test split (3002) plus a train subset (--train-per-class) with
the official Qwen3VLEmbedder (kept unmodified, imported from tools/), using the
same context-expanded bbox geometry our RST Region uses. Output npz feeds
tools/emb_probe_classify.py.

Run with the user's own qwen3vl_env (transformers 5.14) on an idle GPU:
  CUDA_VISIBLE_DEVICES=1 nice -n 10 ~/qwen3vl_env/bin/python -u tools/emb_probe_encode.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

Image.MAX_IMAGE_PIXELS = None

INSTRUCTION = ("Identify the type of the aircraft shown in this image, "
               "focusing on fine-grained structural details.")


def load_records(root, train_per_class, splits, modal):
    """modal=region -> one record per instance (bbox crop);
    modal=scene  -> one record per unique image (whole image, deduplicated)."""
    recs = []
    per_class = {}
    seen_img = set()
    with open(os.path.join(root, "index", "instances.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("split") not in splits:
                continue
            if modal == "scene":
                if d["image"] in seen_img:
                    continue
                seen_img.add(d["image"])
                recs.append(d)
                continue
            if d["split"] == "train" and train_per_class > 0:
                c = d["class"]
                if per_class.get(c, 0) >= train_per_class:
                    continue
                per_class[c] = per_class.get(c, 0) + 1
            recs.append(d)
    return recs


def crop_region(root, rec, margin=0.0):
    img = Image.open(os.path.join(root, rec["scene_file"])).convert("RGB")
    x0, y0, x1, y1 = [float(v) for v in rec["bbox_expanded"]]
    if margin:
        w, h = x1 - x0, y1 - y0
        x0, y0 = max(0, x0 - w * margin), max(0, y0 - h * margin)
        x1, y1 = min(img.width, x1 + w * margin), min(img.height, y1 + h * margin)
    sub = img.crop((int(max(0, x0)), int(max(0, y0)),
                    int(min(img.width, x1)), int(min(img.height, y1))))
    img.close()
    if sub.width < 16 or sub.height < 16:
        sub = sub.resize((max(16, sub.width), max(16, sub.height)))
    return sub


def scene_image(root, rec):
    """Whole image, pre-shrunk on the long side so the 8k-px originals do not
    spend decode time we would only resize away in the processor anyway."""
    img = Image.open(os.path.join(root, rec["scene_file"])).convert("RGB")
    long_side = max(img.size)
    if long_side > 1600:
        s = 1600.0 / long_side
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.BILINEAR)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-VL-Embedding-8B"))
    ap.add_argument("--train-per-class", type=int, default=20)
    ap.add_argument("--modal", default="region", choices=["region", "scene"])
    ap.add_argument("--splits", default="test,train", help="comma list, e.g. test or test,train,validation")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    if not args.out:
        suffix = "all" if len(splits) == 3 else splits[0]
        args.out = "index/emb_probe_qwen3vle_%s_%s.npz" % (args.modal, suffix)

    recs = load_records(root, args.train_per_class, splits, args.modal)
    n_by_split = {}
    for r in recs:
        n_by_split[r["split"]] = n_by_split.get(r["split"], 0) + 1
    print("modal=%s splits=%s records: %d %s"
          % (args.modal, splits, len(recs), n_by_split), flush=True)

    model = Qwen3VLEmbedder(model_name_or_path=args.model, torch_dtype="bfloat16")
    print("embedder ready", flush=True)

    feats, labels, ids, sp_list = [], [], [], []
    t0 = time.time()
    for i in range(0, len(recs), args.batch):
        chunk = recs[i:i + args.batch]
        inputs = []
        for r in chunk:
            try:
                img = (crop_region(root, r) if args.modal == "region"
                       else scene_image(root, r))
            except Exception as e:
                print("  image fail %s: %s" % (r["image"], e))
                img = Image.new("RGB", (64, 64), (127, 127, 127))
            inputs.append({"image": img, "instruction": INSTRUCTION})
        emb = model.process(inputs, normalize=True)
        emb = emb.float().cpu().numpy()
        feats.append(emb)
        for r in chunk:
            labels.append(r["class"])
            ids.append("%s#%d" % (r["image"], r["instance_index"]))
            sp_list.append(r["split"])
        done = min(i + args.batch, len(recs))
        if (i // args.batch) % 10 == 0:
            sp = (time.time() - t0) / max(1, done)
            eta = sp * (len(recs) - done) / 60
            print("  %d/%d  %.2f s/img  eta %.1f min" % (done, len(recs), sp, eta),
                  flush=True)

    X = np.concatenate(feats, 0).astype(np.float32)
    np.savez_compressed(os.path.join(root, args.out),
                        features=X, labels=np.array(labels),
                        ids=np.array(ids), splits=np.array(sp_list),
                        model=args.model, instruction=INSTRUCTION,
                        modal=args.modal, train_per_class=args.train_per_class)
    print("wrote %s  features %s  (%.1f min)"
          % (args.out, X.shape, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
