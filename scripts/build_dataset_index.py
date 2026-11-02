"""Build the machine-readable index the RST pipeline reads from.

Pure geometry over labels_with_split.csv, so it runs and is verifiable on this
machine without a GPU. Produces:

  index/instances.jsonl   one line per aircraft entity (region box, expanded
                          region box, scene reference, class, domain, split)
  index/images.jsonl      one line per image (scene tiling plan under 3.2.3)
  index/class_meta.json   class vocabulary, counts, domain, sampling weights
  index/build_report.json what the geometry actually costs, incl. whether the
                          table 4-2 token budget is satisfiable
"""

import csv
import json
import math
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "index")
CSV_PATH = os.path.join(ROOT, "labels_with_split.csv")
DOMAIN_PATH = os.path.join(ROOT, "knowledge", "domain_map.json")

# --- table 4-2 ---
REGION_EXPAND = 0.25      # context added on each side of the GT box (3.1)
SCENE_TRIGGER = 896       # long side above which tiling replaces whole-image resize
TILE = 448                # tile side in source pixels
STRIDE = 336              # tile step -> ~25% overlap
SCENE_TOKEN_CAP = 1024
PATCH = 16                # SigLIP 2 ViT-B/16 patch size
# A 448px tile is 28x28 = 784 raw patch tokens, so a multi-tile scene cannot
# possibly fit a 1024 cap. Tiles are adaptively pooled before the budget is
# applied; these are the pooled token counts per tile class.
NEAR_TILE_TOKENS = 49     # 7x7, tiles that contain or border an entity
FAR_TILE_TOKENS = 4       # 2x2, background tiles


def expanded_box(x1, y1, x2, y2, W, H, ratio):
    """Context-expanded crop box, clamped to the image, preserving the shift
    that clamping would otherwise introduce."""
    bw, bh = x2 - x1, y2 - y1
    ex, ey = bw * ratio, bh * ratio
    nx1, ny1 = x1 - ex, y1 - ey
    nx2, ny2 = x2 + ex, y2 + ey
    # keep the box inside the image by borrowing unused slack from the other side
    if nx1 < 0:
        nx2 = min(W, nx2 - nx1); nx1 = 0
    if ny1 < 0:
        ny2 = min(H, ny2 - ny1); ny1 = 0
    if nx2 > W:
        nx1 = max(0, nx1 - (nx2 - W)); nx2 = W
    if ny2 > H:
        ny1 = max(0, ny1 - (ny2 - H)); ny2 = H
    return [nx1, ny1, nx2, ny2]


def _cover(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max(1.0, (b[2] - b[0]) * (b[3] - b[1]))


def protected_box(raw, others, W, H, ratio, cover_limit=0.15):
    """Context expansion, then pull an edge back where the crop would otherwise
    swallow a neighbouring aircraft. Without this, 12.8% of multi-entity regions
    contain >=50% of a different box (see index/region_check.json). Only the
    neighbour's far side on one axis can be cut, so overlapping aircraft stay
    merged and need the mask channel instead."""
    box = expanded_box(raw[0], raw[1], raw[2], raw[3], W, H, ratio)
    for _ in range(6):
        worst, wc = None, cover_limit
        for o in others:
            c = _cover(box, o)
            if c > wc:
                worst, wc = o, c
        if worst is None:
            break
        ox1, oy1, ox2, oy2 = worst
        x1, y1, x2, y2 = box
        raw = [v for v in raw]
        cands = []
        if ox2 + 1 <= x2 and ox2 + 1 <= raw[0]:
            cands.append([ox2 + 1, y1, x2, y2])
        if ox1 - 1 >= x1 and ox1 - 1 >= raw[2]:
            cands.append([x1, y1, ox1 - 1, y2])
        if oy2 + 1 <= y2 and oy2 + 1 <= raw[1]:
            cands.append([x1, oy2 + 1, x2, y2])
        if oy1 - 1 >= y1 and oy1 - 1 >= raw[3]:
            cands.append([x1, y1, x2, oy1 - 1])
        if not cands:
            break
        box = max(cands, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    return [round(v, 1) for v in box]


def tile_grid(W, H):
    """Tile origins for a 25%-overlapping grid; edge tiles are pulled inward so
    every tile is TILE x TILE and the border is covered exactly once."""
    def axis(n):
        if n <= TILE:
            return [0]
        step = (n - TILE) / math.ceil((n - TILE) / STRIDE)
        return [round(i * step) for i in range(int((n - TILE) / step) + 1)]
    return axis(W), axis(H)


def plan_scene(W, H, boxes):
    """Tiling plan for one image: which tiles are entity-bearing, and whether the
    resulting token count fits the cap."""
    long_side = max(W, H)
    if long_side <= SCENE_TRIGGER:
        return {"mode": "whole", "tiles": [], "n_tiles": 1,
                "near_tokens_per_tile": (TILE // PATCH) ** 2, "far_tokens_per_tile": 0,
                "tokens": (TILE // PATCH) ** 2, "fits": (TILE // PATCH) ** 2 <= SCENE_TOKEN_CAP,
                "degraded": []}

    xs, ys = tile_grid(W, H)
    tiles = []
    for ty in range(len(ys)):
        for tx in range(len(xs)):
            x, y = xs[tx], ys[ty]
            rect = (x, y, min(x + TILE, W), min(y + TILE, H))
            near = any(rect[0] < b[2] and rect[2] > b[0] and rect[1] < b[3] and rect[3] > b[1]
                       for b in boxes)
            tiles.append({"x": rect[0], "y": rect[1], "w": rect[2] - rect[0],
                          "h": rect[3] - rect[1], "near": near, "gi": tx, "gj": ty})
    n_near = sum(1 for t in tiles if t["near"])
    n_far = len(tiles) - n_near

    near_t, far_t = NEAR_TILE_TOKENS, FAR_TILE_TOKENS
    tokens = n_near * near_t + n_far * far_t
    degraded = []
    while tokens > SCENE_TOKEN_CAP:
        if far_t > 1:
            far_t = 1
            degraded.append("far->1tok")
        elif near_t > 9:
            near_t = 9
            degraded.append("near->3x3")
        elif near_t > 4:
            near_t = 4
            degraded.append("near->2x2")
        else:
            break
        tokens = n_near * near_t + n_far * far_t
    return {"mode": "tiled", "n_tiles": len(tiles), "n_near": n_near, "n_far": n_far,
            "tiles": tiles, "raw_tiles_at_cap": tokens <= SCENE_TOKEN_CAP,
            "near_tokens_per_tile": near_t, "far_tokens_per_tile": far_t,
            "tokens": tokens, "fits": tokens <= SCENE_TOKEN_CAP, "degraded": degraded}


def sampling_weights(counts):
    """Three class-balancing schemes over training-set instance counts."""
    tot = sum(counts.values())
    n = len(counts)
    inv = {c: tot / (n * k) for c, k in counts.items()}
    beta = 0.999
    eff = {c: (1 - beta) / (1 - beta ** k) for c, k in counts.items()}
    esc = max(eff.values())
    eff = {c: v / esc for c, v in eff.items()}
    mean = tot / n
    sqrt_ = {c: math.sqrt(mean / k) for c, k in counts.items()}
    rep = {c: max(1, round(tot / (n * k))) for c, k in counts.items()}
    return {"inverse_freq": inv, "effective_number_beta999": eff,
            "sqrt_inverse": sqrt_, "resample_repeats": rep}


def main():
    os.makedirs(IDX, exist_ok=True)
    domain = json.load(open(DOMAIN_PATH, encoding="utf-8"))["domain"]

    rows = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    classes = sorted({r["class"] for r in rows})
    cidx = {c: i for i, c in enumerate(classes)}

    order = defaultdict(list)
    for r in rows:
        order[r["filename"]].append(r)

    inst_records = []
    expand_clip = Counter()
    fit_ratio = []           # scale applied when fitting the crop into 384x384
    distortion = []          # longer/shorter side ratio of the crop

    imgs = defaultdict(lambda: {"boxes": [], "instances": []})
    for key, rws in order.items():
        raw_boxes = []
        for pos, r in enumerate(rws):
            W, H = int(r["width"]), int(r["height"])
            raw_boxes.append([max(0, int(r["xmin"])), max(0, int(r["ymin"])),
                              min(W, int(r["xmax"])), min(H, int(r["ymax"]))])
        for pos, r in enumerate(rws):
            W, H = int(r["width"]), int(r["height"])
            raw = raw_boxes[pos]
            others = [b for j, b in enumerate(raw_boxes) if j != pos]
            naive = expanded_box(raw[0], raw[1], raw[2], raw[3], W, H, REGION_EXPAND)
            exp = protected_box(raw, others, W, H, REGION_EXPAND)
            if exp[2] - exp[0] < (raw[2] - raw[0]) * (1 + 2 * REGION_EXPAND) - 1:
                expand_clip["shrunk_to_exclude_neighbour"] += 1
            if (naive[2] - naive[0]) < (raw[2] - raw[0]) * (1 + 2 * REGION_EXPAND) - 1:
                expand_clip["clamped_by_border"] += 1
            bw, bh = max(1, raw[2] - raw[0]), max(1, raw[3] - raw[1])
            ew, eh = max(1, exp[2] - exp[0]), max(1, exp[3] - exp[1])
            fit_ratio.append(min(384 / ew, 384 / eh))
            distortion.append(max(ew, eh) / max(1, min(ew, eh)))
            imgs[key]["boxes"].append(exp)
            imgs[key]["instances"].append(pos)
            inst_records.append({
                "image": key, "instance_index": pos, "class": r["class"],
                "class_index": cidx[r["class"]], "domain": domain[r["class"]],
                "split": r["split"], "image_size": [W, H],
                "bbox": raw, "bbox_expanded": exp,
                "bbox_expanded_naive": [round(v, 1) for v in naive],
                "region_file": "crop/%s/%s_%d.jpg" % (r["class"], key, pos),
                "scene_file": "dataset/%s.jpg" % key,
                "bbox_area_frac": round((bw * bh) / (W * H), 6),
                "long_side": max(bw, bh),
            })

    with open(os.path.join(IDX, "instances.jsonl"), "w", encoding="utf-8") as fh:
        for rec in inst_records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_tiled = n_whole = n_overflow = 0
    tile_counts, near_counts, token_counts = [], [], []
    size_of = {rec["image"]: rec["image_size"] for rec in inst_records}
    with open(os.path.join(IDX, "images.jsonl"), "w", encoding="utf-8") as fh:
        for key in sorted(imgs):
            W, H = size_of[key]
            plan = plan_scene(W, H, imgs[key]["boxes"])
            tiles = plan.pop("tiles", [])
            if plan["mode"] == "tiled":
                n_tiled += 1
                tile_counts.append(plan["n_tiles"])
                near_counts.append(plan["n_near"])
                token_counts.append(plan["tokens"])
                if not plan["fits"]:
                    n_overflow += 1
            else:
                n_whole += 1
                tile_counts.append(1)
                near_counts.append(1)
                token_counts.append(plan["tokens"])
            fh.write(json.dumps({"image": key, "size": [W, H],
                                 "n_instances": len(imgs[key]["boxes"]),
                                 "scene": plan, "tile_rects": tiles},
                                ensure_ascii=False) + "\n")

    train_counts = Counter(r["class"] for r in inst_records if r["split"] == "train")
    split_counts = Counter((r["class"], r["split"]) for r in inst_records)
    w = sampling_weights(dict(train_counts))
    meta = {"classes": classes,
            "class_index": cidx,
            "domain": domain,
            "counts": {c: {"train": split_counts[(c, "train")],
                           "val": split_counts[(c, "validation")],
                           "test": split_counts[(c, "test")]} for c in classes},
            "weights": w,
            "params": {"region_expand": REGION_EXPAND, "region_input": 384,
                       "region_neighbour_protection": True,
                       "scene_trigger_long_side": SCENE_TRIGGER, "tile": TILE,
                       "stride": STRIDE, "patch": PATCH,
                       "near_tokens_per_tile": NEAR_TILE_TOKENS,
                       "far_tokens_per_tile": FAR_TILE_TOKENS,
                       "scene_token_cap": SCENE_TOKEN_CAP}}
    with open(os.path.join(IDX, "class_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    def pct(vals, qs=(1, 5, 10, 25, 50, 75, 90, 95, 99)):
        v = sorted(vals)
        return {"n": len(v), "min": round(v[0], 3),
                **{"p%d" % q: round(v[min(len(v) - 1, int(q / 100 * len(v)))], 3) for q in qs},
                "max": round(v[-1], 3)}

    n_tiles = [p for p in tile_counts]
    rep = {
        "instances": len(inst_records), "images": len(imgs),
        "splits": dict(Counter(r["split"] for r in inst_records)),
        "region_expand": {"ratio": REGION_EXPAND,
                          "clamped_by_image_border": expand_clip["clamped_by_border"],
                          "clamped_pct": round(100 * expand_clip["clamped_by_border"] / len(inst_records), 1),
                          "shrunk_to_exclude_neighbour": expand_clip["shrunk_to_exclude_neighbour"],
                          "shrunk_pct": round(100 * expand_clip["shrunk_to_exclude_neighbour"] / len(inst_records), 1)},
        "region_fit_scale_into_384": pct(fit_ratio),
        "region_upsample_gt": {
            ">1x": round(100 * sum(1 for f in fit_ratio if f > 1) / len(fit_ratio), 1),
            ">2x": round(100 * sum(1 for f in fit_ratio if f > 2) / len(fit_ratio), 1),
            ">4x": round(100 * sum(1 for f in fit_ratio if f > 4) / len(fit_ratio), 1),
            ">8x": round(100 * sum(1 for f in fit_ratio if f > 8) / len(fit_ratio), 1)},
        "region_aspect_distortion": pct(distortion),
        "scene_tiling": {"whole_image": n_whole, "tiled": n_tiled,
                         "tiled_pct": round(100 * n_tiled / len(imgs), 1),
                         "tiles_per_image": pct(n_tiles),
                         "near_tiles_per_image": pct(near_counts),
                         "tokens_per_image": pct(token_counts),
                         "tokens_over_cap": n_overflow,
                         "cap": SCENE_TOKEN_CAP,
                         "raw_patch_tokens_per_tile": (TILE // PATCH) ** 2,
                         "tokens_p50_if_tiles_unpooled": pct(
                             [tc * (TILE // PATCH) ** 2 for tc in tile_counts])["p50"]},
        "class_balance": {"min_train": min(train_counts.values()),
                          "max_train": max(train_counts.values()),
                          "ratio": round(max(train_counts.values()) / min(train_counts.values()), 1),
                          "resample_repeats_max": max(w["resample_repeats"].values())},
    }
    with open(os.path.join(IDX, "build_report.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
