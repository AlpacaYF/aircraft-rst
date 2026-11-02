"""Precompute SigLIP 2 embeddings for every aircraft region.

Feeds 3.3.2 (visual consistency of the knowledge base) and 3.3.3 (mining hard
pairs from real visual similarity rather than from what the teacher happens to
name). Pass --scene to also cache scene tile features for training.

  python -m rst.extract_embeddings --split all
"""

import argparse
import os
import time

import numpy as np
import torch
from PIL import Image

from .config import Config
from .data import EntityStore, build_region
from .encoders import SigLIP2Encoder

try:
    from tqdm import tqdm
except ImportError:          # pragma: no cover
    def tqdm(x, **kw):
        return x


def run(cfg, args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = SigLIP2Encoder(args.model or cfg.vision_model, freeze=True,
                         cache_dir=(os.path.join(cfg.root, args.cache_dir) if args.cache_dir else None),
                         load_text=False).to(device)
    enc.eval()
    store = EntityStore(os.path.join(cfg.root, cfg.index_dir), None)
    recs = [r for r in store.instances
            if args.split in ("all", r["split"])]
    if args.limit:
        recs = recs[:args.limit]
    print("encoding %d regions on %s" % (len(recs), device))

    feats = np.zeros((len(recs), enc.embed_dim), dtype=np.float16)
    t0 = time.time()
    with torch.no_grad():
        for start in tqdm(range(0, len(recs), args.batch), desc="regions"):
            chunk = recs[start:start + args.batch]
            px = []
            for rec in chunk:
                img = Image.open(os.path.join(cfg.root, rec["scene_file"])).convert("RGB")
                region, _ = build_region(img, rec["image_size"][0], rec["image_size"][1],
                                         rec, False, cfg, jitter=False)
                img.close()
                px.append(region[:3])
            batch = torch.stack(px).to(device)
            f = enc.image_feature(batch)
            f = torch.nn.functional.normalize(f, dim=-1)
            feats[start:start + len(chunk)] = f.float().cpu().numpy().astype(np.float16)

    out = os.path.join(cfg.root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        features=feats,
        classes=np.array([r["class"] for r in recs], dtype=object),
        class_index=np.array([store.class_index[r["class"]] for r in recs], dtype=np.int32),
        splits=np.array([r["split"] for r in recs], dtype=object),
        instance_ids=np.array(["%s#%d" % (r["image"], r["instance_index"]) for r in recs],
                              dtype=object),
        images=np.array([r["image"] for r in recs], dtype=object),
        model=np.array([args.model or cfg.vision_model], dtype=object))
    print("wrote %s in %.1f min (%.1f img/s)"
          % (out, (time.time() - t0) / 60, len(recs) / max(1e-6, time.time() - t0)))


def run_trained(cfg, args):
    """3.3.2 as revised: export the representation that stage 2 actually aligned.

    A frozen SigLIP 2 carries no usable text->image signal on these 114 types
    (own-class rank median 56 against a chance median of 57.5, top-1 0.7%), which
    is why the verification moved to after the contrastive alignment of 3.5.2.
    This exports z_R -- the pooled F_RS of eq. 3-13, i.e. the `v` that eq. 3-19
    pulls toward the correct prototype -- together with every knowledge sentence
    pushed through the same trained text projection, so both sides of the
    similarity live in one aligned space instead of two unrelated ones.
    """
    from torch.utils.data import DataLoader

    from .data import EntityDataset, RSTCollator
    from .model import RSTModel

    run_dir = os.path.dirname(os.path.abspath(args.ckpt))
    cfg = Config.load(os.path.join(run_dir, "config.json"))
    cfg.root = os.path.abspath(args.root)
    cfg.num_workers = args.workers
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    txt = np.load(os.path.join(cfg.root, "knowledge", "text_features.npz"),
                  allow_pickle=True)
    protos = {"tokens": torch.from_numpy(txt["tokens"]),
              "mask": torch.from_numpy(txt["mask"]), "class_of_token": None}
    model = RSTModel(cfg, protos, attr_dim=int(txt["attr_matrix"].shape[1]))
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print("loaded %s (stage cfg use_scene=%s use_text=%s)"
          % (args.ckpt, cfg.use_scene, cfg.use_text))

    ds = EntityDataset(cfg, None, train=False)
    recs = ds.records
    if args.split != "all":
        keep = [i for i, r in enumerate(recs) if r["split"] == args.split]
        ds.records = [recs[i] for i in keep]
        recs = ds.records
    if args.limit:
        ds.records = recs[:args.limit]
        recs = ds.records
    print("encoding %d regions on %s (batch %d)" % (len(recs), device, args.batch))

    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=RSTCollator(ds),
                        pin_memory=(str(device).startswith("cuda")),
                        persistent_workers=cfg.num_workers > 0)
    chunks, t0 = [], time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="regions"):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            z = model(batch, need_features=True)["z_region"]
            z = torch.nn.functional.normalize(z.float(), dim=-1)
            chunks.append(z.cpu().numpy().astype(np.float16))
    feats = np.concatenate(chunks) if chunks else np.zeros((0, cfg.hidden), np.float16)
    assert feats.shape[0] == len(recs), "%d features for %d records" % (
        feats.shape[0], len(recs))

    with torch.no_grad():
        raw = torch.from_numpy(txt["features"].astype(np.float32)).to(device)
        tp = torch.nn.functional.normalize(model.text_proj(raw).float(), dim=-1)
    text_projected = tp.cpu().numpy().astype(np.float16)
    print("projected %d knowledge sentences into the aligned space (dim %d)"
          % (text_projected.shape[0], text_projected.shape[1]))

    store = ds.store
    out = os.path.join(cfg.root, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(
        out,
        features=feats,
        text_projected=text_projected,
        text_rows=txt["rows"].astype(np.int32),
        classes=np.array([r["class"] for r in recs], dtype=object),
        class_index=np.array([store.class_index[r["class"]] for r in recs], dtype=np.int32),
        splits=np.array([r["split"] for r in recs], dtype=object),
        instance_ids=np.array(["%s#%d" % (r["image"], r["instance_index"]) for r in recs],
                              dtype=object),
        images=np.array([r["image"] for r in recs], dtype=object),
        space=np.array(["trained_z_region"], dtype=object),
        ckpt=np.array([os.path.abspath(args.ckpt)], dtype=object),
        model=np.array([cfg.vision_model], dtype=object))
    print("wrote %s in %.1f min (%.1f img/s)"
          % (out, (time.time() - t0) / 60, len(recs) / max(1e-6, time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="all", choices=["all", "train", "validation", "test"])
    ap.add_argument("--model", default="")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", default="",
                    help="defaults to $HF_HOME set by env.sh")
    ap.add_argument("--ckpt", default="",
                    help="a training checkpoint (runs/<name>/best.pt); switches to the "
                         "post-3.5.2 aligned space instead of frozen SigLIP 2 features")
    ap.add_argument("--out", default="index/region_embeddings.npz")
    args = ap.parse_args()

    cfg = Config()
    cfg.root = os.path.abspath(args.root)
    if args.ckpt:
        run_trained(cfg, args)
    else:
        run(cfg, args)


if __name__ == "__main__":
    main()
