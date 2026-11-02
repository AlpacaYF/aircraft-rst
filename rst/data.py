"""Dataset and batching for entity-level aircraft recognition.

Three design points matter:

* Geometry consistency. Region and Scene of the same image must agree, so the
  horizontal flip is decided per *image* from a hash of (image, epoch) and both
  branches read that same decision. Colour jitter is region-only, which is what
  keeps scene features cacheable.

* One decode per image. ``__getitem__`` returns only metadata; the collator
  opens each distinct image in the batch once and cuts every region and every
  scene tile from it.

* Scene shared across entities, as in 3.5.4: a batch carries a deduplicated
  scene list plus an entity -> scene index.
"""

import hashlib
import json
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image, ImageEnhance

Image.MAX_IMAGE_PIXELS = None


def variant_id(image_key, flip):
    """Scene-cache key: one entry per orientation."""
    return "%s#%d" % (image_key, int(flip))


def stable_flip(image_key, epoch, enabled=True):
    if not enabled:
        return False
    h = hashlib.md5(("%s|%d" % (image_key, epoch)).encode()).digest()
    return bool(h[0] & 1)


def letterbox(img, size, fill=127):
    """Resize keeping aspect onto a square canvas; also return the map so boxes
    can be carried through it unchanged."""
    w, h = img.size
    s = min(size / max(1, w), size / max(1, h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    out = Image.new("RGB", (size, size), (fill, fill, fill))
    out.paste(img.resize((nw, nh), Image.BILINEAR), ((size - nw) // 2, (size - nh) // 2))
    return out, s, (size - nw) // 2, (size - nh) // 2


def to_tensor(img):
    arr = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    arr = arr.view(img.size[1], img.size[0], 3).permute(2, 0, 1).float()
    return arr.mul_(1 / 127.5).sub_(1.0)          # SigLIP expects [-1, 1]


class EntityStore:
    """Instance index plus class metadata, shared by train/val/test loaders.

    `shots` caps the TRAIN split at K entities per class, sampled once with a
    fixed seed: every run that passes the same (shots, shot_seed) sees the
    identical subset, so R/RT/RST few-shot comparisons are paired on data, not
    on sampling luck. Val/test stores are built with shots=None by EntityDataset
    and are never capped.
    """

    def __init__(self, index_dir, split=None, shots=None, shot_seed=12345):
        with open(os.path.join(index_dir, "class_meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        self.classes = meta["classes"]
        self.class_index = meta["class_index"]
        self.domain_of = meta["domain"]
        self.params = meta["params"]
        self.weights = meta["weights"]
        self.military_index = [i for i, c in enumerate(self.classes)
                               if self.domain_of[c] == "military"]
        self.civilian_index = [i for i, c in enumerate(self.classes)
                               if self.domain_of[c] == "civilian"]

        all_rows = self._read(os.path.join(index_dir, "instances.jsonl"))
        self.instances = [r for r in all_rows if split is None or r["split"] == split]
        if shots and split == "train":
            by_class = {}
            for r in self.instances:
                by_class.setdefault(r["class"], []).append(r)
            rng = random.Random(shot_seed)
            kept = []
            for c in sorted(by_class):
                rows = by_class[c]
                order = list(range(len(rows)))
                rng.shuffle(order)
                kept.extend(rows[i] for i in order[:shots])
            print("few-shot: capped train split at %d/class (seed %d) -> %d entities "
                  "(was %d)" % (shots, shot_seed, len(kept), len(self.instances)))
            self.instances = kept
        self.by_image = {}
        for r in self.instances:
            self.by_image.setdefault(r["image"], []).append(r)

        self.scenes = {}
        for rec in self._read(os.path.join(index_dir, "images.jsonl")):
            plan = rec["scene"]
            rects = rec["tile_rects"]
            plan["grid_nx"] = (max((t["gi"] for t in rects), default=0) + 1)
            plan["grid_ny"] = (max((t["gj"] for t in rects), default=0) + 1)
            self.scenes[rec["image"]] = {"plan": plan, "rects": rects}

    @staticmethod
    def _read(path):
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    def class_weight_tensor(self, scheme="inverse_freq"):
        return torch.tensor([self.weights[scheme][c] for c in self.classes],
                            dtype=torch.float32)


class EntityDataset(Dataset):
    """Yields per-entity metadata only; pixels are produced by RSTCollator."""

    def __init__(self, cfg, split, train=False):
        self.cfg = cfg
        self.train = train
        # few-shot capping is a TRAIN-split device only: the val/test stores must
        # always see every entity, including when eval_checkpoint rebuilds them
        shots = getattr(cfg, "few_shot", 0) if train else 0
        self.store = EntityStore(os.path.join(cfg.root, cfg.index_dir), split,
                                 shots=(shots or None),
                                 shot_seed=getattr(cfg, "few_shot_seed", 12345))
        self.records = self.store.instances
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        flip = stable_flip(rec["image"], self.epoch, self.cfg.augment_flip and self.train)
        return {"idx": i, "rec": rec, "flip": flip,
                "class": self.store.class_index[rec["class"]],
                "domain": 0 if self.store.domain_of[rec["class"]] == "military" else 1,
                "instance_id": "%s#%d" % (rec["image"], rec["instance_index"])}


def build_region(img, W, H, rec, flip, cfg, jitter, erase=False, expand_jit=0.0):
    crop = [float(v) for v in rec["bbox_expanded"]]
    gt = [float(rec["bbox"][0]), float(rec["bbox"][1]),
          float(rec["bbox"][2]), float(rec["bbox"][3])]
    if flip:
        crop = [W - crop[2], crop[1], W - crop[0], crop[3]]
        gt = [W - gt[2], gt[1], W - gt[0], gt[3]]
    if expand_jit > 0:
        # sample the context expansion per entity instead of always 25%: a fixed
        # ring makes the boundary between target and context perfectly
        # predictable, which is one more thing the model can overfit. The jitter
        # is around cfg.region_expand, clamped to stay positive and <= 1.0.
        e = cfg.region_expand + random.uniform(-expand_jit, expand_jit)
        e = min(1.0, max(0.05, e))
        w, h = gt[2] - gt[0], gt[3] - gt[1]
        crop = [gt[0] - w * e, gt[1] - h * e, gt[2] + w * e, gt[3] + h * e]
    crop = [min(max(v, 0.0), float(W)) for v in crop]
    if crop[2] - crop[0] < 2 or crop[3] - crop[1] < 2:
        crop = list(gt)

    S = cfg.region_input
    sub = img.crop([int(math.floor(crop[0])), int(math.floor(crop[1])),
                    int(math.ceil(crop[2])), int(math.ceil(crop[3]))])
    if jitter:
        if random.random() < 0.5:
            sub = ImageEnhance.Brightness(sub).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            sub = ImageEnhance.Contrast(sub).enhance(random.uniform(0.85, 1.15))
    boxed, s, px, py = letterbox(sub, S)

    def m(v, axis):
        o = crop[0] if axis == 0 else crop[1]
        p = px if axis == 0 else py
        return min(max((v - o) * s + p, 0.0), float(S))

    gt_local = [m(gt[0], 0), m(gt[1], 1), m(gt[2], 0), m(gt[3], 1)]
    region = to_tensor(boxed)
    if cfg.use_mask_channel:
        mask = torch.zeros(1, S, S)
        x1, y1 = int(round(gt_local[0])), int(round(gt_local[1]))
        x2, y2 = max(x1 + 1, int(round(gt_local[2]))), max(y1 + 1, int(round(gt_local[3])))
        mask[0, y1:min(y2, S), x1:min(x2, S)] = 1.0
        region = torch.cat([region, mask], dim=0)
    if erase:
        # one grey rectangle over 2-15% of the canvas, all channels: RGB goes to
        # the letterbox grey (0.0 after to_tensor's [0,255]->[-1,1] map) and the
        # mask channel to background. Balanced sampling drove train Top-1 to
        # 1.000 in the v1 runs, which zeroed every discriminative loss that was
        # supposed to shape stage 3; this keeps the task from being memorised.
        a = random.uniform(0.02, 0.15) * S * S
        r = random.uniform(0.3, 1 / 0.3)
        ew, eh = max(2, int((a * r) ** 0.5)), max(2, int((a / r) ** 0.5))
        ex = random.randint(0, S - ew)
        ey = random.randint(0, S - eh)
        region[:, ey:ey + eh, ex:ex + ew] = 0.0
    return region, torch.tensor(gt_local, dtype=torch.float32)


def build_scene_tiles(img, W, H, scene, flip, cfg):
    """Returns (tiles tensor [T,3,S,S], meta [T,4]=(gi,gj,near,padding))."""
    rects = scene["rects"]
    plan = scene["plan"]
    if plan.get("mode") != "tiled":
        rects = [{"x": 0, "y": 0, "w": W, "h": H, "near": True, "gi": 0, "gj": 0}]
    if len(rects) > cfg.scene_max_tiles:
        # keep entity-bearing tiles first, then fill the remainder spatially
        near = [t for t in rects if t["near"]]
        far = [t for t in rects if not t["near"]]
        keep = near[:cfg.scene_max_tiles]
        room = cfg.scene_max_tiles - len(keep)
        if room > 0 and far:
            stride = max(1, len(far) // room)
            keep.extend(far[::stride][:room])
        rects = keep
    S = cfg.scene_input
    gx = plan.get("grid_nx", 1) or 1
    tensors, meta = [], []
    for t in rects:
        x, y, w, h, gi = t["x"], t["y"], t["w"], t["h"], t["gi"]
        if flip:
            x = W - (x + w)
            gi = (gx - 1) - gi
        sub = img.crop((x, y, x + w, y + h)).resize((S, S), Image.BILINEAR)
        tensors.append(to_tensor(sub))
        meta.append([gi, t["gj"], 1.0 if t["near"] else 0.0, 0.0])
    if not tensors:
        # an image with no recorded tiles still needs one (blank) slot so the
        # batch has a well-defined scene tensor
        tensors = [torch.zeros(3, S, S)]
        meta = [[0, 0, 1.0, 0.0]]
    return torch.stack(tensors), torch.tensor(meta, dtype=torch.float32)


class RSTCollator:
    def __init__(self, dataset):
        self.ds = dataset
        self.cfg = dataset.cfg
        self.root = dataset.cfg.root
        self._cache = None

    @property
    def cache(self):
        # built once per worker so forked DataLoader processes each hold their own
        # np.memmap rather than sharing a closed handle
        if self._cache is None:
            from .scene_cache import SceneCache
            d = os.path.join(self.root, self.cfg.scene_cache_dir)
            self._cache = SceneCache(os.path.join(d, "scene_tokens.bin"),
                                     os.path.join(d, "scene_index.npz"))
        return self._cache

    def __call__(self, items):
        cfg = self.cfg
        by_image = {}
        for pos, it in enumerate(items):
            by_image.setdefault((it["rec"]["image"], it["flip"]), []).append((pos, it))

        regions = [None] * len(items)
        gt_boxes = [None] * len(items)
        scene_keys = []
        scene_items, scene_meta = [], []
        entity_to_scene = [0] * len(items)

        for (key, flip), members in by_image.items():
            rec0 = members[0][1]["rec"]
            W, H = rec0["image_size"]
            img = Image.open(os.path.join(self.root, rec0["scene_file"])).convert("RGB")
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            slot = len(scene_keys)
            scene_keys.append(key)
            if cfg.use_scene_cache:
                got = self.cache.get(variant_id(key, flip))
                if got is None:
                    raise KeyError("image %s flip=%s missing from the scene cache; "
                                   "re-run rst.cache_scene_features" % (key, flip))
                tok, meta = got
                scene_items.append(torch.from_numpy(tok))
                scene_meta.append(torch.from_numpy(np.asarray(meta, dtype=np.float32)))
            else:
                scene = self.ds.store.scenes.get(key, {"plan": {"mode": "whole"}, "rects": []})
                tiles, tmeta = build_scene_tiles(img, W, H, scene, flip, cfg)
                scene_items.append(tiles)
                scene_meta.append(tmeta)

            for pos, it in members:
                region, gt_local = build_region(
                    img, W, H, it["rec"], flip, cfg, self.ds.train and cfg.augment_color,
                    erase=(self.ds.train and cfg.erase_prob > 0
                           and random.random() < cfg.erase_prob),
                    expand_jit=(cfg.expand_jitter if self.ds.train else 0.0))
                regions[pos] = region
                gt_boxes[pos] = gt_local
                entity_to_scene[pos] = slot
            img.close()

        out = {
            "region": torch.stack(regions),
            "class": torch.tensor([it["class"] for it in items], dtype=torch.long),
            "domain": torch.tensor([it["domain"] for it in items], dtype=torch.long),
            "gt_box": torch.stack(gt_boxes),
            "entity_to_scene": torch.tensor(entity_to_scene, dtype=torch.long),
            "images": scene_keys,
            "instance_ids": [it["instance_id"] for it in items],
        }
        if cfg.use_scene_cache:
            n_img = len(scene_items)
            max_tok = max(t.size(0) for t in scene_items)
            max_t = max(m.size(0) for m in scene_meta)
            dim = scene_items[0].size(1)
            d_pad = torch.zeros(n_img, max_tok, dim)
            m_pad = torch.zeros(n_img, max_t, 4)
            m_pad[..., 3] = 1.0
            for i, (t, m) in enumerate(zip(scene_items, scene_meta)):
                d_pad[i, :t.size(0)] = t
                m_pad[i, :m.size(0)] = m
            out["scene_tokens"] = d_pad
            out["scene_tile_meta"] = m_pad
        else:
            n_img = len(scene_items)
            max_tiles = max(t.size(0) for t in scene_items)
            S = scene_items[0].shape[-1]
            d_pad = torch.zeros(n_img, max_tiles, 3, S, S)
            m_pad = torch.zeros(n_img, max_tiles, 4)
            m_pad[..., 3] = 1.0
            for i, (t, m) in enumerate(zip(scene_items, scene_meta)):
                d_pad[i, :t.size(0)] = t
                m_pad[i, :m.size(0)] = m
            out["scene_tiles"] = d_pad
            out["scene_tile_meta"] = m_pad
        return out


class BalancedEntitySampler(Sampler):
    """Repeats small classes so each contributes a comparable number of entities
    per epoch. The paper's long tail is 17..1560, so natural sampling starves the
    rare types entirely."""

    def __init__(self, dataset, seed=0, target=None):
        self.ds = dataset
        self.seed = seed
        self.epoch = 0
        buckets = {}
        for i, rec in enumerate(dataset.records):
            buckets.setdefault(rec["class"], []).append(i)
        self.buckets = list(buckets.values())
        self.target = target or max(1, len(dataset.records) // len(self.buckets))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = random.Random(self.seed * 100003 + self.epoch)
        out = []
        for bucket in self.buckets:
            reps = max(1, round(self.target / max(1, len(bucket))))
            for _ in range(reps):
                order = bucket[:]
                g.shuffle(order)
                out.extend(order)
        g.shuffle(out)
        return iter(out)

    def __len__(self):
        return sum(max(1, round(self.target / max(1, len(b)))) * len(b)
                   for b in self.buckets)
