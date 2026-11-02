#!/usr/bin/env python
"""End-to-end aircraft entity extraction from ONE image.

    .venv/bin/python deploy/demo.py --image /path/to/photo.jpg --outdir deploy/output/demo

Pipeline:
  1. YOLO11x (single-class "aircraft") runs in .venv-det via subprocess -> boxes.
  2. Per box: protected context-expanded crop + 4-channel input (RGB + mask).
  3. Per image: the same scene-tiling geometry the model was trained with
     (build_dataset_index.plan_scene), encoded live by the shared tower.
  4. Three RST experts -> equal-weight logit average (DEDF) -> class + domain.
  5. Writes <outdir>/result.json + <outdir>/result.png and prints a table.

Recipes (the paper's final method is `uniform`):
  uniform : runs/v2_rs_s0 runs/v2_rt_nosup_s0 runs/v2_rst_nosup_s0
  full    : runs/v2_rs_s0 runs/v2_rt_adapt_s0 runs/v2_rst_f_s0
  single  : runs/v2_rst_nosup_s0 (no DEDF)

Only writes inside the project directory and uses the project venvs; no
environment changes are made.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rst.config import Config                                    # noqa: E402
from rst.entity_extract import DetectionSet, load_rst_expert      # noqa: E402
from rst.eval_checkpoint import cfg_from_ckpt                     # noqa: E402
from rst.train import load_semantic                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_bdi", os.path.join(ROOT, "scripts", "build_dataset_index.py"))
_bdi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bdi)

DET_PYTHON = os.path.join(ROOT, ".venv-det", "bin", "python")
DET_WEIGHTS = "runs/detect/det_runs/yolo11x_640_b16/weights/best.pt"
RECIPES = {
    "uniform": "runs/v2_rs_s0,runs/v2_rt_nosup_s0,runs/v2_rst_nosup_s0",
    "full": "runs/v2_rs_s0,runs/v2_rt_adapt_s0,runs/v2_rst_f_s0",
    "single": "runs/v2_rst_nosup_s0",
}


def pick_gpu():
    """Prefer a card with no compute process, then the most free memory."""
    def run(cmd):
        return subprocess.run(cmd, capture_output=True, text=True).stdout
    gpus = []
    for line in run(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
                     "--format=csv,noheader,nounits"]).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 3:
            gpus.append({"idx": int(p[0]), "uuid": p[1], "free": int(float(p[2])), "apps": 0})
    by_uuid = {g["uuid"]: g for g in gpus}
    for line in run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                     "--format=csv,noheader"]).splitlines():
        p = [x.strip() for x in line.split(",")]
        if p and p[0] in by_uuid:
            by_uuid[p[0]]["apps"] += 1
    idle = [g for g in gpus if g["apps"] == 0]
    pool = idle or gpus
    g = max(pool, key=lambda x: x["free"])
    if not idle:
        print("WARNING: no fully idle GPU; using cuda:%d (%d MiB free, %d compute app(s))"
              % (g["idx"], g["free"], g["apps"]))
    return g["idx"]


def build_items_for_image(image_path, boxes, cfg):
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
                    "instance_index": 1000 + j, "split": "infer"},
            "flip": False, "class": 0, "domain": 0,
            "instance_id": "%s#det%d" % (key, j), "score": float(boxes[j][4]),
        })
        order.append((key, j))
    return items, order, (W, H), key


class ImageStore:
    """The minimum EntityStore surface the collator/scene plan need, for one image."""

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
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--image", required=True)
    ap.add_argument("--outdir", default="deploy/output/demo")
    ap.add_argument("--recipe", default="uniform", choices=list(RECIPES))
    ap.add_argument("--experts", default="",
                    help="comma-separated run dirs; overrides --recipe")
    ap.add_argument("--yolo-weights", default=DET_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--gpu", default="auto", help="auto | 0 | 1 | 2 | 3")
    ap.add_argument("--boxes-json", default="", help="skip detection, reuse boxes")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-visualize", action="store_true")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    image_path = os.path.abspath(args.image)
    if not os.path.exists(image_path):
        raise SystemExit("no such image: %s" % image_path)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    if args.boxes_json:
        boxes = json.load(open(args.boxes_json))["boxes"]
        print("boxes from %s: %d" % (args.boxes_json, len(boxes)))
    else:
        tmp = os.path.join(outdir, "boxes.json")
        subprocess.run([DET_PYTHON, "-m", "tools.detect_one", "--image", image_path,
                        "--weights", args.yolo_weights, "--conf", str(args.conf),
                        "--imgsz", str(args.imgsz), "--out", tmp],
                       check=True, cwd=root)
        boxes = json.load(open(tmp))["boxes"]
    t1 = time.time()
    if not boxes:
        print("no aircraft detected (conf >= %.2f) -- nothing to recognise" % args.conf)
        return

    if args.gpu == "auto":
        gpu = pick_gpu()
    else:
        gpu = int(args.gpu)
    device = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    cfg = Config()
    cfg.root = root
    cfg.use_mask_channel = True
    cfg.use_scene_cache = False
    cfg.num_workers = 0

    expert_dirs = (args.experts.split(",") if args.experts
                   else RECIPES[args.recipe].split(","))
    pairs = []
    for mdir in expert_dirs:
        md = mdir if os.path.isabs(mdir) else os.path.join(root, mdir)
        p = torch.load(os.path.join(md, "best.pt"), map_location=device,
                       weights_only=False)
        c = cfg_from_ckpt(p, root)
        c.use_scene_cache = False
        pr, _, _ = load_semantic(c)
        pairs.append((load_rst_expert(p, c, pr, device, tag=os.path.basename(md)), c))
        print("  expert: %-28s scene=%s text=%s" % (md, c.use_scene, c.use_text))
    t2 = time.time()

    items, order, (W, H), key = build_items_for_image(image_path, boxes, cfg)
    store = ImageStore(meta, key, image_path, W, H, boxes)
    ds = DetectionSet(cfg, store, items)

    from torch.utils.data import DataLoader
    from rst.data import RSTCollator
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0,
                        collate_fn=RSTCollator(ds))
    cls_batches, dom_batches = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            per_cls, per_dom = [], []
            for model, c in pairs:
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    out = model(b)
                lc = out["logits_class"]
                if c.use_text and "logits_semantic" in out:
                    lc = lc + out["logits_semantic"]
                per_cls.append(lc.float())
                per_dom.append(out["logits_domain"].float())
            cls_batches.append(torch.stack(per_cls, 0).cpu().numpy())   # [E,B,C]
            dom_batches.append(torch.stack(per_dom, 0).cpu().numpy())   # [E,B,2]
    LOG = np.concatenate(cls_batches, 1)                                # [E,N,C]
    DOM = np.concatenate(dom_batches, 1)                                # [E,N,2]
    fused = LOG.mean(0)
    pred = fused.argmax(1)
    conf = torch.softmax(torch.from_numpy(fused), -1).numpy().max(1)
    dom = DOM.mean(0).argmax(1)
    expert_top1 = [LOG[e].argmax(1) for e in range(LOG.shape[0])]
    agree = np.stack([(t == pred) for t in expert_top1]).mean(0)
    t3 = time.time()

    classes = meta["classes"]
    domain_names = ["military", "civilian"]
    entities = []
    print("\n%-4s %-14s %-9s %8s %7s %6s  %s" %
          ("#", "class", "domain", "conf", "det", "agree", "expert top-1"))
    for i, (box, ci, di, cf, ag) in enumerate(zip(boxes, pred, dom, conf, agree)):
        e = {"bbox": box[:4], "det_score": box[4],
             "class": classes[int(ci)], "class_id": int(ci),
             "domain": domain_names[int(di)],
             "domain_id": int(di),
             "confidence": round(float(cf), 4),
             "expert_agreement": round(float(ag), 3),
             "expert_top1": [classes[int(t[i])] for t in expert_top1]}
        entities.append(e)
        print("%-4d %-14s %-9s %8.4f %7.3f %6.2f  %s" %
              (i, e["class"], e["domain"], e["confidence"], box[4], ag,
               ",".join(e["expert_top1"])))

    result = {
        "system": "RST-DEDF",
        "recipe": args.recipe,
        "experts": [os.path.basename(d) for d in expert_dirs],
        "fused_by": "equal-weight logits average of the experts",
        "detector": {"weights": args.yolo_weights, "conf": args.conf,
                     "imgsz": args.imgsz},
        "image": image_path,
        "image_size": [W, H],
        "n_entities": len(entities),
        "timings_s": {"detect": round(t1 - t0, 3),
                      "load_experts": round(t2 - t1, 3),
                      "recognize": round(t3 - t2, 3),
                      "total": round(t3 - t0, 3)},
        "entities": entities,
    }
    jpath = os.path.join(outdir, "result.json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("\njson  -> %s" % jpath)

    if not args.no_visualize:
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
        ppath = os.path.join(outdir, "result.png")
        im.save(ppath)
        print("image -> %s" % ppath)

    print("time  -> detect %.3fs | load %.1fs | recognize %.3fs | total %.3fs"
          % (t1 - t0, t2 - t1, t3 - t2, t3 - t0))


if __name__ == "__main__":
    main()
