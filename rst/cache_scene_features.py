"""Precompute Scene features for every image (the caching 3.5.4 already assumes).

Why this exists: each scene tile is a full 384x384 ViT-B forward, and with the
backbone trainable an 8 GB card cannot fit even batch 4 once tiles enter the
graph. The proposal's own inference design says the scene is encoded once per
image and shared by that image's entities -- making that literal, with a frozen
encoder, removes the scene branch from the training graph entirely.

What is stored is the *pooled patch tokens*, before the trainable projection, so
`scene_proj` and `tile_pos` still learn normally. Tiles are stored in the same
rectangular [T, kn*kn] layout the live path builds: an entity-bearing tile keeps
all kn*kn of them, a background tile keeps its first kf*kf and the rest are zero
and masked by the reader (3.2.3's two-level budget).

  python -m rst.cache_scene_features
  # spread over 4 idle GPUs, then merge:
  for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python -m rst.cache_scene_features \
      --shard $i --nshards 4 & done; wait
  python -m rst.cache_scene_features --merge --nshards 4

Writes cache/scene_tokens.bin (fp16) and cache/scene_index.npz.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import torch
from PIL import Image

from .config import Config
from .data import EntityStore, build_scene_tiles
from .encoders import SigLIP2Encoder

try:
    from tqdm import tqdm
except ImportError:                                    # pragma: no cover
    def tqdm(x, **kw):
        return x

Image.MAX_IMAGE_PIXELS = None
MAGIC = "RSTSCENE"


def variant_id(key, flip):
    """Cache entries are keyed per orientation so flipped regions never pair
    with unflipped scene features."""
    return "%s#%d" % (key, int(flip))


def load_existing(bin_path, idx_path, want_fp):
    if not (os.path.exists(bin_path) and os.path.exists(idx_path)):
        return [], None, 0
    idx = np.load(idx_path, allow_pickle=True)
    got = str(idx["fingerprint"][0]) if "fingerprint" in idx.files else ""
    if got != want_fp:
        raise RuntimeError(
            "scene cache was built with %r but this run needs %r -- the pooling or "
            "tile cap changed, so re-run with --no-resume" % (got, want_fp))
    done = [str(k) for k in idx["images"]]
    return done, idx, len(done)


def merge_shards(out_dir, nshards):
    """Concatenate per-GPU shard caches into scene_tokens.bin / scene_index.npz.

    Shards are independent runs over disjoint variant lists, so the bins simply
    concatenate in shard order and each shard's row_ptr is offset by the running
    token total.
    """
    keys, row_ptr, n_tiles, meta_rows = [], [0], [], []
    off = 0
    parts = []
    for i in range(nshards):
        ip = os.path.join(out_dir, "scene_index.shard%d.npz" % i)
        bp = os.path.join(out_dir, "scene_tokens.shard%d.bin" % i)
        if not (os.path.exists(ip) and os.path.exists(bp)):
            raise SystemExit("shard %d missing (%s)" % (i, bp))
        idx = np.load(ip, allow_pickle=True)
        parts.append(bp)
        keys.extend(str(k) for k in idx["images"])
        n_tiles.extend(int(v) for v in idx["n_tiles"])
        meta_rows.append(idx["meta"])
        rp = idx["row_ptr"].astype(np.int64)
        row_ptr.extend(int(v) + off for v in rp[1:])
        off += int(rp[-1])
        print("  shard %d: %d variants, %d tokens" % (i, len(idx["images"]), int(rp[-1])))
    bin_path = os.path.join(out_dir, "scene_tokens.bin")
    with open(bin_path, "wb") as out:
        for p in parts:
            with open(p, "rb") as fh:
                shutil.copyfileobj(fh, out, 1 << 22)
    first = np.load(os.path.join(out_dir, "scene_index.shard0.npz"), allow_pickle=True)
    np.savez_compressed(
        os.path.join(out_dir, "scene_index.npz"),
        images=np.array(keys, dtype=object),
        row_ptr=np.array(row_ptr, dtype=np.int64),
        n_tiles=np.array(n_tiles, dtype=np.int32),
        meta=np.concatenate(meta_rows).astype(np.float16),
        tile_tokens=first["tile_tokens"], near_pool=first["near_pool"],
        far_pool=first["far_pool"], dim=first["dim"], model=first["model"],
        max_tiles=first["max_tiles"], fingerprint=first["fingerprint"])
    print("merged %d variants, %d tokens, %.2f GB -> %s"
          % (len(keys), row_ptr[-1], os.path.getsize(bin_path) / 1e9, bin_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="")
    ap.add_argument("--batch", type=int, default=32, help="tiles per encoder call")
    ap.add_argument("--near-pool", type=int, default=0,
                    help="pooled grid side for entity-bearing tiles (0 = cfg)")
    ap.add_argument("--far-pool", type=int, default=0,
                    help="pooled grid side for background tiles (0 = cfg)")
    ap.add_argument("--max-tiles", type=int, default=0,
                    help="tiles kept per image (0 = cfg.scene_max_tiles)")
    ap.add_argument("--out-dir", default="cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-both-flips", dest="both_flips", action="store_false",
                    help="cache only the unflipped orientation (half the disk, but then"
                         " disable --augment-flip at train time)")
    ap.add_argument("--shard", type=int, default=0, help="this run's shard index")
    ap.add_argument("--nshards", type=int, default=1,
                    help="split the variants this many ways, one process per GPU")
    ap.add_argument("--merge", action="store_true",
                    help="concatenate --nshards shard caches, then exit")
    ap.set_defaults(both_flips=True)
    args = ap.parse_args()

    out_dir = os.path.join(os.path.abspath(args.root), args.out_dir)
    if args.merge:
        merge_shards(out_dir, args.nshards)
        return
    if not 0 <= args.shard < args.nshards:
        raise SystemExit("--shard must be in [0, %d)" % args.nshards)

    root = os.path.abspath(args.root)
    cfg = Config()
    cfg.root = root
    if args.max_tiles:
        cfg.scene_max_tiles = args.max_tiles
    if args.near_pool:
        cfg.scene_near_pool = args.near_pool
    if args.far_pool:
        cfg.scene_far_pool = args.far_pool
    kn = cfg.scene_near_pool
    kf = min(cfg.scene_far_pool, kn)
    slots = kn * kn
    # the same invariant RSTModel enforces: the budget must hold when every tile
    # that survives the cap turns out to carry an entity
    if cfg.scene_max_tiles * slots > cfg.scene_token_cap:
        raise SystemExit("%d tiles x %d tokens = %d, over scene_token_cap=%d"
                         % (cfg.scene_max_tiles, slots,
                            cfg.scene_max_tiles * slots, cfg.scene_token_cap))
    model_name = args.model or cfg.vision_model
    fp = "%s|near%d|far%d|tiles%d" % (model_name, kn, kf, cfg.scene_max_tiles)

    suffix = "" if args.nshards <= 1 else ".shard%d" % args.shard
    bin_path = os.path.join(out_dir, "scene_tokens%s.bin" % suffix)
    idx_path = os.path.join(out_dir, "scene_index%s.npz" % suffix)
    os.makedirs(out_dir, exist_ok=True)

    store = EntityStore(os.path.join(root, cfg.index_dir), None)
    images = sorted(store.scenes)
    if args.limit:
        images = images[:args.limit]

    done, prev, n_done = (([], None, 0) if args.no_resume
                          else load_existing(bin_path, idx_path, fp))
    done_set = set(done)
    flips = (False, True) if args.both_flips else (False,)
    all_variants = [(k, f) for k in images for f in flips]
    if args.nshards > 1:
        all_variants = [v for i, v in enumerate(all_variants)
                        if i % args.nshards == args.shard]
    todo = [v for v in all_variants if variant_id(*v) not in done_set]
    print("%d images x %d orientation(s) = %d variants; %d cached, %d to encode"
          % (len(images), len(flips), len(all_variants), n_done, len(todo)))
    if not todo:
        print("cache complete")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = SigLIP2Encoder(model_name, freeze=True, load_text=False).to(device).eval()

    # Both orientations are cached so that horizontal-flip augmentation stays
    # available: a flipped region paired with an unflipped scene would teach the
    # attention to retrieve mirrored context. Costs 2x disk, nothing at train time.
    variants = todo
    print("encoding %d scene variants" % len(variants))

    mode = "ab" if n_done else "wb"
    tokens_out = open(bin_path, mode)
    meta_rows, row_ptr, n_tiles, keys = [], [0], [], list(done)
    if n_done and prev is not None:
        meta_rows = [list(r) for r in prev["meta"]]
        row_ptr = list(prev["row_ptr"])[:len(keys) + 1]
        n_tiles = list(prev["n_tiles"])

    t0 = time.time()
    with torch.no_grad():
        for key, flip in tqdm(variants, desc="scene cache"):
            scen = store.scenes[key]
            path = os.path.join(root, "dataset", key + ".jpg")
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            W, H = img.size
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            tiles, tmeta = build_scene_tiles(img, W, H, scen, flip, cfg)
            img.close()
            flat = tiles.reshape(-1, *tiles.shape[1:]).to(device)
            raws = []
            for i in range(0, flat.size(0), args.batch):
                raws.append(enc.patch_tokens(flat[i:i + args.batch]))
            raw = torch.cat(raws, 0)                       # [T, g*g, d]
            g = enc.grid_side(raw)
            # Same rectangular layout encode_scene rebuilds at train time: every
            # tile gets a kn*kn-wide row and a background tile fills only its
            # first kf*kf slots, the rest staying zero and masked. Storing it
            # pre-padded keeps the reader and the collator oblivious to near/far.
            buf = raw.new_zeros(raw.size(0), slots, enc.embed_dim)
            near = tmeta[:, 2] > 0.5
            p_near = enc.pool_grid(raw, g, kn)
            if bool(near.any()):
                buf[near] = p_near[near]
            if bool((~near).any()):
                p_far = enc.pool_grid(raw, g, kf)
                buf[~near, :kf * kf] = p_far[~near]
            arr = buf.float().cpu().numpy().astype(np.float16)
            tokens_out.write(arr.reshape(-1, enc.embed_dim).tobytes())
            meta_rows.extend(tmeta.numpy().tolist())
            row_ptr.append(row_ptr[-1] + arr.shape[0] * arr.shape[1])
            n_tiles.append(arr.shape[0])
            keys.append(variant_id(key, flip))

    tokens_out.close()
    np.savez_compressed(
        idx_path,
        images=np.array(keys, dtype=object),
        row_ptr=np.array(row_ptr, dtype=np.int64),
        n_tiles=np.array(n_tiles, dtype=np.int32),
        meta=np.array(meta_rows, dtype=np.float16),
        tile_tokens=np.array([slots], dtype=np.int32),
        near_pool=np.array([kn], dtype=np.int32),
        far_pool=np.array([kf], dtype=np.int32),
        dim=np.array([enc.embed_dim], dtype=np.int32),
        model=np.array([model_name], dtype=object),
        max_tiles=np.array([cfg.scene_max_tiles], dtype=np.int32),
        fingerprint=np.array([fp], dtype=object))
    el = time.time() - t0
    print("cached %d images in %.1f min (%.1f img/s)" % (len(keys), el / 60, len(keys) / max(1e-6, el)))
    print("tokens.bin %.2f GB" % (os.path.getsize(bin_path) / 1e9))
    print("index: %d images, %d tokens total" % (len(keys), row_ptr[-1]))


if __name__ == "__main__":
    main()
