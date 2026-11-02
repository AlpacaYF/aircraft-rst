#!/usr/bin/env python
"""Integrated single-image inference: YOLO11 detector + the three trained RST
experts fused by DEDF (decision-level equal-weight logit average).

  .venv/bin/python -m tools.infer_image --image /path/to/photo.jpg
  # optional: --visualize out.png --conf 0.5 --no-detect (boxes via --boxes-json)

Pipeline (all on the *training* venv side except the detector):
  1. subprocess: .venv-det/bin/python tools/detect_one.py  -> box json
  2. per box: context-expanded crop (protected_box, neighbours = other boxes),
     384 letterbox + GT-style mask channel  -> Region input
  3. per image: tiling plan via scripts/build_dataset_index.plan_scene (the SAME
     geometry the index was built with) -> Scene tiles, encoded by the shared tower
  4. three experts (R+S / R+T adaptive / R+S+T) forward -> class/domain logits,
     averaged in the logits domain, argmax per head
Output: a JSON with every entity's box, class, domain, confidence + console table
+ optional annotated PNG.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rst.config import Config                                    # noqa: E402
from rst.entity_extract import DetectionSet, load_rst_expert      # noqa: E402
from rst.eval_checkpoint import cfg_from_ckpt                     # noqa: E402
from rst.train import load_semantic                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_bdi", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "build_dataset_index.py"))
_bdi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bdi)

DEFAULT_EXPERTS = "runs/v2_rs_s0,runs/v2_rt_adapt_s0,runs/v2_rst_f_s0"
DET_PYTHON = ".venv-det/bin/python"


def build_items_for_image(root, image_path, boxes, cfg):
    """One record per detected box, same geometry the training index used."""
    key = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path)
    W, H = img.size
    img.close()
    raws = [list(map(float, b[:4])) for b in boxes]
    items, order = [], []
    for j, box in enumerate(raws):
        others = [o for k, o in enumerate(raws) if k != j]
        items.append({
            "rec": {"image": key, "scene_file": image_path, "image_size": [W, H],
                    "bbox": box,
                    "bbox_expanded": _bdi.protected_box(box, others, W, H,
                                                        cfg.region_expand),
                    "instance_index": j, "split": "infer"},
            "flip": False, "class": 0, "domain": 0,
            "instance_id": "%s#det%d" % (key, j), "score": float(boxes[j][4]),
        })
        order.append((key, j))
    return items, order, (W, H), key


class ImageStore:
    """Minimal stand-in for EntityStore covering exactly what the collator and
    the region builder touch, with a dynamically planned scene for this image."""

    def __init__(self, meta, key, scene_file, W, H, boxes):
        self.classes = meta["classes"]
        self.class_index = meta["class_index"]
        self.domain_of = meta["domain"]
        self.train = False
        plan = _bdi.plan_scene(W, H, [list(map(float, b[:4])) for b in boxes])
        rects = plan.get("tiles", [])
        if plan.get("mode") != "tiled":
            rects = [{"x": 0, "y": 0, "w": W, "h": H, "near": True, "gi": 0, "gj": 0}]
        plan = dict(plan)
        plan["grid_nx"] = max((t["gi"] for t in rects), default=0) + 1
        plan["grid_ny"] = max((t["gj"] for t in rects), default=0) + 1
        self.scenes = {key: {"plan": plan, "rects": rects}}
        self.by_image = {key: [{"image": key, "scene_file": scene_file,
                                "image_size": [W, H]}]}
        self.instances = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--image", required=True)
    ap.add_argument("--experts", default=DEFAULT_EXPERTS)
    ap.add_argument("--yolo-weights",
                    default="runs/detect/det_runs/yolo11x_640_b16/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--boxes-json", default="",
                    help="skip detection and read boxes from this file instead")
    ap.add_argument("--out", default="", help="JSON output path")
    ap.add_argument("--visualize", default="", help="annotated PNG path")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise SystemExit("no such image: %s" % image_path)

    # ---------- 1. detection ----------
    if args.boxes_json:
        boxes = json.load(open(args.boxes_json))["boxes"]
        print("boxes from %s: %d" % (args.boxes_json, len(boxes)))
    else:
        tmp = os.path.join(root, "runs", "_infer_boxes.json")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        subprocess.run([DET_PYTHON, "-m", "tools.detect_one", "--image", image_path,
                        "--weights", args.yolo_weights, "--conf", str(args.conf),
                        "--out", tmp], check=True, cwd=root)
        boxes = json.load(open(tmp))["boxes"]
    if not boxes:
        print("no aircraft detected (conf >= %.2f) -- nothing to recognise" % args.conf)
        return

    # ---------- 2. experts ----------
    device = torch.device("cuda:%s" % args.gpu if torch.cuda.is_available() else "cpu")
    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    cfg = Config()
    cfg.root = root
    cfg.use_mask_channel = True
    cfg.use_scene_cache = False          # live scene encoding for arbitrary images
    cfg.num_workers = 0
    pairs = []
    for mdir in args.experts.split(","):
        md = mdir if os.path.isabs(mdir) else os.path.join(root, mdir)
        p = torch.load(os.path.join(md, "best.pt"), map_location=device,
                       weights_only=False)
        c = cfg_from_ckpt(p, root)
        c.use_scene_cache = False
        pr, _, _ = load_semantic(c)
        pairs.append((load_rst_expert(p, c, pr, device, tag=os.path.basename(md)), c))
        print("  expert loaded: %s (scene=%s text=%s)" % (md, c.use_scene, c.use_text))

    # ---------- 3. per-entity inputs (same geometry as training) ----------
    items, order, (W, H), key = build_items_for_image(root, image_path, boxes, cfg)
    store = ImageStore(meta, key, image_path, W, H, boxes)
    ds = DetectionSet(cfg, store, items)

    # ---------- 4. DEDF forward ----------
    with torch.no_grad():
        h = 8                                   # small batch: single image
        from torch.utils.data import DataLoader
        from rst.data import RSTCollator
        loader = DataLoader(ds, batch_size=h, shuffle=False, num_workers=0,
                            collate_fn=RSTCollator(ds))
        # reuse classify_dedf's aggregation by collecting logits per expert
        all_logits = []
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            per = []
            for model, c in pairs:
                out = model(b)
                lc = out["logits_class"]
                if c.use_text and "logits_semantic" in out:
                    lc = lc + out["logits_semantic"]
                per.append(lc.float())
            all_logits.append(torch.stack(per, 0).cpu().numpy())   # [E, B, C]
        LOG = np.concatenate(all_logits, 1)                        # [E, N, C]
    fused = LOG.mean(0)                                            # [N, C]
    pred = fused.argmax(1)
    conf = torch.softmax(torch.from_numpy(fused), -1).numpy().max(1)
    expert_top1 = [LOG[e].argmax(1) for e in range(LOG.shape[0])]
    agree = np.stack([(t == pred) for t in expert_top1]).mean(0)

    # domain: reuse each expert's visual domain head through a second quick pass
    domains = []
    with torch.no_grad():
        from torch.utils.data import DataLoader as _DL
        from rst.data import RSTCollator as _RC
        loader = _DL(ds, batch_size=h, shuffle=False, num_workers=0, collate_fn=_RC(ds))
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            ds_logits = []
            for model, c in pairs:
                out = model(b)
                ds_logits.append(out["logits_domain"].float())
            domains.append(torch.stack(ds_logits, 0).mean(0).argmax(1).cpu().numpy())
    dom = np.concatenate(domains).astype(int)

    classes = meta["classes"]
    domain_names = ["military", "civilian"]
    entities = []
    print("\n%-4s %-11s %-10s %8s %6s %7s" % ("#", "class", "domain", "conf", "conf_raw",
                                              "agreement"))
    for i, (box, c_i, d_i, cf, ag) in enumerate(zip(boxes, pred, dom, conf, agree)):
        e = {"bbox": box[:4], "det_score": box[4], "class": classes[int(c_i)],
             "class_id": int(c_i), "domain": domain_names[int(d_i)],
             "confidence": round(float(cf), 4), "expert_agreement": round(float(ag), 2),
             "expert_top1": [classes[int(t[i])] for t in expert_top1]}
        entities.append(e)
        print("%-4d %-11s %-10s %8.4f %6.2f %7.2f"
              % (i, e["class"], e["domain"], e["confidence"], box[4], ag))

    out = {
        "image": image_path,
        "image_size": [W, H],
        "n_entities": len(entities),
        "fused_by": "DEDF: equal-weight logits average of " + args.experts,
        "entities": entities,
    }
    out_path = args.out or os.path.join(
        "det_runs", "infer_%s.json" % os.path.splitext(os.path.basename(image_path))[0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nwrote %s" % out_path)

    if args.visualize:
        im = Image.open(image_path).convert("RGB")
        dr = ImageDraw.Draw(im)
        col = {"military": (255, 60, 60), "civilian": (60, 140, 255)}
        for e in entities:
            x1, y1, x2, y2 = [float(v) for v in e["bbox"]]
            dr.rectangle([x1, y1, x2, y2], outline=col[e["domain"]], width=3)
            tag = "%s %s %.2f" % (e["class"], e["domain"][:3].upper(), e["confidence"])
            tw = dr.textlength(tag)
            dr.rectangle([x1, max(0, y1 - 16), x1 + tw + 6, y1], fill=col[e["domain"]])
            dr.text((x1 + 3, max(0, y1 - 15)), tag, fill=(255, 255, 255))
        im.save(args.visualize)
        print("visualised -> %s" % args.visualize)


if __name__ == "__main__":
    main()
