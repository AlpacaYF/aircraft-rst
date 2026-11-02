#!/usr/bin/env python
"""表4-8: entity-level extraction metrics for any detector + the RST classifier.

4.2.2 defines the task: no ground-truth boxes are given, a detector localises
aircraft, and the RST module decides the型号. An entity counts as correct only when
IoU >= 0.5 against a ground-truth box AND the predicted class matches, with one-to-one
greedy matching by detector confidence.

This is the shared second half of every row in that table. The detector runs in
.venv-det and writes a boxes JSON; this runs in the training venv because it needs the
RST checkpoint, and it reuses the *identical* crop geometry the model was trained with
-- `protected_box` for the 25% context expansion with neighbour protection, and
`RSTCollator` for region + scene assembly -- so a detected box becomes a region exactly
the way an annotated box would.

Three numbers are reported because a single Entity F1 hides where the loss is:
  * entity P/R/F1 -- the table's columns
  * detection-only P/R/F1 at the same IoU with the class ignored -- pure localisation
  * Top-1 over correctly localised entities -- pure recognition given localisation
With the GT-box Top-1 from 表4-3 as the ceiling, the last two bracket what dropping the
annotator's boxes actually costs.

  python -m rst.entity_extract --boxes det_runs/yolo11x_640/test_boxes.json \
      --run runs/align_full_kb --gpu 0
"""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import EntityStore, RSTCollator
from .eval import macro_f1
from .eval_checkpoint import cfg_from_ckpt
from .model import RSTModel
from .train import load_semantic

# imported rather than reimplemented: these define the crop geometry the whole pipeline
# was built with, and a second copy here would eventually drift from the first
_spec = importlib.util.spec_from_file_location(
    "_build_dataset_index",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "scripts", "build_dataset_index.py"))
_bdi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bdi)
protected_box = _bdi.protected_box


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


class DetectionSet:
    """Shim exposing the attributes RSTCollator reads off an EntityDataset, so
    detected boxes travel through exactly the same collation path as training data."""

    def __init__(self, cfg, store, items):
        self.cfg = cfg
        self.store = store
        self.items = items
        self.records = items
        self.train = False

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def build_items(det, gt, store, cfg, score_floor, max_per_image):
    """Detected boxes -> classifier inputs, with the same context expansion the index
    used. `others` for neighbour protection is the image's other *detections*, since
    the detector has no access to the annotations."""
    items, order, kept, dropped = [], [], 0, 0
    for image in gt:
        boxes = [b for b in det.get(image, []) if b[4] >= score_floor]
        dropped += len(det.get(image, [])) - len(boxes)
        boxes.sort(key=lambda b: -b[4])
        if max_per_image:
            boxes = boxes[:max_per_image]
        kept += len(boxes)
        if not boxes:
            continue
        rec0 = store.by_image[image][0]
        W, H = rec0["image_size"]
        raws = [list(map(float, b[:4])) for b in boxes]
        for j, box in enumerate(raws):
            others = [o for k, o in enumerate(raws) if k != j]
            items.append({
                "rec": {"image": image, "scene_file": rec0["scene_file"],
                        "image_size": [W, H], "bbox": box,
                        "bbox_expanded": protected_box(box, others, W, H,
                                                       cfg.region_expand),
                        "instance_index": 1000 + j, "split": "test"},
                "flip": False, "class": 0, "domain": 0,
                "instance_id": "%s#det%d" % (image, j), "score": float(boxes[j][4])})
            order.append((image, j))
    return items, order, kept, dropped


@torch.no_grad()
def classify(model, ds, cfg, device, batch, workers, collate=None):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=(collate or RSTCollator)(ds),
                        pin_memory=(device.type == "cuda"))
    preds = []
    for bi, b in enumerate(loader):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda" and cfg.amp)):
            out = model(b)
        lc = out["logits_class"]
        if cfg.use_text and "logits_semantic" in out:
            lc = lc + out["logits_semantic"]
        preds.append(lc.argmax(1).cpu())
        if bi % 50 == 0:
            print("  classified %d/%d" % (min((bi + 1) * batch, len(ds)), len(ds)),
                  flush=True)
    return torch.cat(preds).numpy()


def load_rst_expert(ck, cfg, protos, device, tag=""):
    """Load a checkpoint into the current RSTModel tolerating parameters that
    simply did not exist when the run was trained (sem_gate came with the
    adaptive-fusion release, proto_delta with the learnable-prototype one).
    Anything else missing, or any unexpected key, is a real mismatch and raises.
    """
    sd = ck["model"]
    attr_dim = int(sd["head_attr.weight"].shape[0]) if "head_attr.weight" in sd else 0
    m = RSTModel(cfg, protos, attr_dim=attr_dim)
    res = m.load_state_dict(sd, strict=False)
    fresh = ("sem_gate", "proto_delta")
    bad_missing = [k for k in res.missing_keys
                   if not k.startswith(fresh)]
    if bad_missing or res.unexpected_keys:
        raise RuntimeError("checkpoint/model mismatch%s: missing=%s unexpected=%s"
                           % (" " + tag if tag else "", bad_missing,
                              list(res.unexpected_keys)))
    if res.missing_keys:
        print("  note: %d keys created after this checkpoint, fresh-initialised%s"
              % (len(res.missing_keys), " (" + tag + ")" if tag else ""))
    m.to(device).eval()
    return m


@torch.no_grad()
def classify_dedf(models, ds, cfg, device, batch, workers):
    """Decision-level fusion of several RST experts over detector boxes.

    Every expert sees the same collated batch (region + cached scene tokens);
    each keeps its own cfg, so a scene-off or text-off expert simply ignores the
    fields it does not use. Class scores are averaged before argmax -- the same
    parameter-free aggregation the DEDF ablation uses on GT boxes.
    """
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=RSTCollator(ds), pin_memory=(device.type == "cuda"))
    preds = []
    for bi, b in enumerate(loader):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        total = None
        for model, c in models:
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda" and c.amp)):
                out = model(b)
            lc = out["logits_class"]
            if c.use_text and "logits_semantic" in out:
                lc = lc + out["logits_semantic"]
            lc = lc.float()
            total = lc if total is None else total + lc
        preds.append((total / len(models)).argmax(1).cpu())
        if bi % 50 == 0:
            print("  classified %d/%d" % (min((bi + 1) * batch, len(ds)), len(ds)),
                  flush=True)
    return torch.cat(preds).numpy()


def match(items, order, pred, gt, iou_th):
    """Greedy one-to-one matching, highest confidence first.

    Localisation and entity matching keep separate claim flags: a box that lands on a
    GT but names the wrong type has consumed that GT for detection purposes, yet the
    GT was still not extracted as a correct entity, so it must stay available to
    another detection and count as a miss if none claims it.
    """
    by_image = {}
    for i, (image, _) in enumerate(order):
        by_image.setdefault(image, []).append(i)

    tp_e = fp_e = fn_e = tp_d = fp_d = fn_d = 0
    m_pred, m_gt = [], []
    for image, idxs in by_image.items():
        idxs.sort(key=lambda i: -items[i]["score"])
        gts = gt[image]
        used_det = [False] * len(gts)
        used_ent = [False] * len(gts)
        for i in idxs:
            box = items[i]["rec"]["bbox"]
            best, bj, best_e, be = 0.0, -1, 0.0, -1
            for g, (gbox, gcls) in enumerate(gts):
                v = iou(box, gbox)
                if not used_det[g] and v > best:
                    best, bj = v, g
                if not used_ent[g] and v > best_e:
                    best_e, be = v, g
            if bj >= 0 and best >= iou_th:
                used_det[bj] = True
                tp_d += 1
                # every IoU-matched pair, right or wrong: the macro-F1 over these is
                # recognition quality given correct localisation. Collecting only the
                # correct ones would make it trivially 1.0.
                m_pred.append(int(pred[i]))
                m_gt.append(int(gts[bj][1]))
            else:
                fp_d += 1
            if be >= 0 and best_e >= iou_th and pred[i] == gts[be][1]:
                used_ent[be] = True
                tp_e += 1
            else:
                fp_e += 1
        fn_d += sum(1 for u in used_det if not u)
        fn_e += sum(1 for u in used_ent if not u)
    return dict(tp_e=tp_e, fp_e=fp_e, fn_e=fn_e, tp_d=tp_d, fp_d=fp_d, fn_d=fn_d,
                m_pred=m_pred, m_gt=m_gt)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boxes", required=True,
                    help="JSON {image: [[x1,y1,x2,y2,score], ...]} in original pixels")
    ap.add_argument("--run", default="runs/align_full_kb")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--arch", default="rst", choices=["rst", "swin_fair", "dedf"],
                    help="recognition module: the RST fusion model, the fair-384 "
                         "Swin-B baseline, or a decision-level fusion (dedf) of "
                         "several RST experts given in --models")
    ap.add_argument("--models", default="",
                    help="comma-separated run dirs for --arch dedf, e.g. "
                         "runs/v2_rs_s0,runs/v2_rt_adapt_s0,runs/v2_rst_f_s0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--score", type=float, default=0.5,
                    help="detector confidence floor; recorded in the output so the row "
                         "is reproducible")
    ap.add_argument("--max-per-image", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--pred-json", default="",
                    help="skip the RST classifier and take per-box predicted class "
                         "indices from this file instead -- used for the rows whose "
                         "recognition module is a VLM (Grounding DINO + VLM, Qwen3-VL). "
                         "Either {image: [[x1,y1,x2,y2,score,class_idx], ...]} or the "
                         "wrapper written by tools/vlm_classify_boxes.py.")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    # The RST checkpoint is only needed when this script does the classification
    # itself; with --pred-json the recognizer is a VLM and there is no checkpoint.
    ck = None
    if args.pred_json:
        cfg = Config()
        cfg.root = root
        cfg.num_workers = args.workers
        run_dir = args.run or "(external predictions)"
        print("classifier: external predictions, crop geometry from defaults "
              "(region_expand %.2f)" % cfg.region_expand)
    else:
        if args.arch == "dedf":
            if not args.models:
                raise SystemExit("--arch dedf needs --models dir1,dir2,dir3")
            # collate geometry is shared by every expert and identical to training
            cfg = Config()
            cfg.root = root
            cfg.num_workers = args.workers
            cfg.use_mask_channel = True
            cfg.use_scene_cache = True
            run_dir = os.path.join(root, "det_runs")
            ck = None
            print("classifier: dedf over %s" % args.models)
        else:
            run_dir = args.run if os.path.isabs(args.run) else os.path.join(root, args.run)
            ck = torch.load(os.path.join(run_dir, args.ckpt), map_location=device,
                            weights_only=False)
            cfg = cfg_from_ckpt(ck, root) if args.arch == "rst" else Config()
            cfg.root = root
            cfg.num_workers = args.workers
            if args.arch == "swin_fair":
                print("classifier: swin_fair %s/%s (baseline path, RegionCollator)"
                      % (run_dir, args.ckpt))
            else:
                print("classifier: %s/%s  scene=%s text=%s" % (run_dir, args.ckpt,
                                                               cfg.use_scene, cfg.use_text))

    with open(args.boxes, encoding="utf-8") as fh:
        det = json.load(fh)
    external = None
    miss = 0
    if args.pred_json:
        with open(args.pred_json, encoding="utf-8") as fh:
            blob = json.load(fh)
        ext = blob.get("predictions", blob)
        # key on the rounded box so the alignment cannot depend on iteration order
        external = {}
        for image, rows in ext.items():
            for r in rows:
                external[(image,) + tuple(round(float(v)) for v in r[:4])] = int(r[5])
        print("using external predictions from %s (%d distinct boxes)"
              % (args.pred_json, len(external)))
    store = EntityStore(os.path.join(root, cfg.index_dir), args.split)
    gt = {img: [(r["bbox"], store.class_index[r["class"]]) for r in recs]
          for img, recs in store.by_image.items()}

    items, order, kept, dropped = build_items(det, gt, store, cfg, args.score,
                                             args.max_per_image)
    n_gt = sum(len(v) for v in gt.values())
    print("detections: %d kept at score >= %.2f (%d dropped) | %d images | %d GT entities"
          % (kept, args.score, dropped, len(gt), n_gt))
    if not items:
        raise SystemExit("no detection survived the score floor -- check --boxes/--score")

    if external is not None:
        pred = np.array([external.get((im,) + tuple(round(float(v))
                                                    for v in it["rec"]["bbox"]), -1)
                         for it, (im, _) in zip(items, order)], dtype=np.int64)
        miss = int((pred < 0).sum())
        # A VLM that answers with something outside the 114 candidates leaves no entry
        # here. Counted as wrong (class -1 matches nothing), which is the same
        # conservative policy the 表4-2 Qwen3-VL row already publishes for its 265
        # unparsed outputs -- aborting on it would silently bias the row upward by
        # dropping the model's failures instead of scoring them.
        if miss > max(2, 0.01 * len(pred)):
            raise SystemExit("%d of %d boxes have no external prediction -- that is a "
                             "boxes/predictions file mismatch, not parse noise"
                             % (miss, len(pred)))
        print("external predictions: %d/%d boxes matched, %d unparsed (counted wrong)"
              % (len(pred) - miss, len(pred), miss))
    else:
        if args.arch == "dedf":
            pairs = []
            for mdir in args.models.split(","):
                md = mdir if os.path.isabs(mdir) else os.path.join(root, mdir)
                p = torch.load(os.path.join(md, args.ckpt), map_location=device,
                               weights_only=False)
                c = cfg_from_ckpt(p, root)
                pr, _, _ = load_semantic(c)
                mm = load_rst_expert(p, c, pr, device, tag=os.path.basename(md))
                pairs.append((mm, c))
                print("  expert: %s (scene=%s text=%s)" % (md, c.use_scene, c.use_text))
            pred = classify_dedf(pairs, DetectionSet(cfg, store, items), cfg, device,
                                 args.batch, args.workers)
        elif args.arch == "swin_fair":
            # fair-384 Swin-B classifier from rst.baselines: same data, same crop
            # geometry, same 4-channel input, no scene/text branches. The ckpt
            # carries its own native_input/in_channels, so the config follows it.
            from .baselines import RegionCollator, VisionBaseline
            cfg.use_scene = cfg.use_text = False
            cfg.use_hard_negative = cfg.use_diff_loss = False
            cfg.use_mask_channel = True
            n_cls = len(store.classes)
            model = VisionBaseline("swin_b", n_cls, dropout=cfg.dropout,
                                   in_channels=int(ck.get("in_channels", 4)),
                                   input_size=int(ck.get("native_input", 384)))
            model.load_state_dict(ck["model"])
            model.to(device).eval()
            print("classifier detail: swin_fair input %d, %dch, %.1fM params"
                  % (model.native, model.in_channels,
                     sum(p.numel() for p in model.parameters()) / 1e6))
            pred = classify(model, DetectionSet(cfg, store, items), cfg, device,
                            args.batch, args.workers,
                            collate=lambda ds: RegionCollator(ds, mask_channel=True))
        else:
            protos, attr_matrix, _ = load_semantic(cfg)
            model = load_rst_expert(ck, cfg, protos, device, tag="single")
            pred = classify(model, DetectionSet(cfg, store, items), cfg, device,
                            args.batch, args.workers)
        if len(pred) != len(items):
            raise SystemExit("collator changed the sample count: %d -> %d"
                             % (len(items), len(pred)))

    m = match(items, order, pred, gt, args.iou)
    ep, er, ef = prf(m["tp_e"], m["fp_e"], m["fn_e"])
    dp, dr, df = prf(m["tp_d"], m["fp_d"], m["fn_d"])
    res = {
        "boxes": args.boxes,
        "classifier": (os.path.basename(args.pred_json) if args.pred_json
                       else os.path.basename(run_dir)),
        "ckpt": args.ckpt, "top_k": args.max_per_image,
        "split": args.split, "iou_threshold": args.iou, "score_threshold": args.score,
        "max_per_image": args.max_per_image,
        "n_images": len(gt), "n_gt_entities": n_gt, "n_detections": len(items),
        "missing_predictions": miss,
        "entity_precision": ep, "entity_recall": er, "entity_f1": ef,
        "detection_precision": dp, "detection_recall": dr, "detection_f1": df,
        "recognition_top1_given_match": (m["tp_e"] / m["tp_d"]) if m["tp_d"] else 0.0,
        "matched_macro_f1": macro_f1(np.array(m["m_gt"]), np.array(m["m_pred"]),
                                     len(store.classes))[0] if m["m_gt"] else 0.0,
    }
    print("\nEntity  P %.4f  R %.4f  F1 %.4f   (TP %d FP %d FN %d)"
          % (ep, er, ef, m["tp_e"], m["fp_e"], m["fn_e"]))
    print("Detect  P %.4f  R %.4f  F1 %.4f   (class ignored, IoU %.2f)"
          % (dp, dr, df, args.iou))
    print("Recognition Top-1 on correctly localised entities: %.2f%%"
          % (100 * res["recognition_top1_given_match"]))
    print("  -> compare with the GT-box Top-1 in 表4-3; the difference is the price of"
          " dropping the annotator's boxes")

    stem = "entity_%s.json" % os.path.basename(args.boxes).replace(".json", "")
    # with external predictions there is no run directory to file this under
    out = args.out or os.path.join("det_runs" if args.pred_json else run_dir, stem)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
