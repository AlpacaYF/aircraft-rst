#!/usr/bin/env python
"""Whole-test-split recognition + matching + timing + error taxonomy.

Loads the three DEDF experts ONCE and processes every test image:
  per image: build region/scene inputs (training geometry) -> 3 experts ->
  equal-weight logits average -> argmax; then greedy IoU>=0.5 one-to-one
  matching against the ground truth to label every prediction:

    correct       matched GT, right class
    class_error   matched GT (IoU>=0.5), wrong class
    false_pos     box matched no GT
    missed        GT that no prediction claimed

Outputs (under --out-dir):
  predictions_all.json   per image: boxes, class, confidence, agreement, timing
  correct_entities.json  every correct prediction
  error_entities.json    every error, grouped by type (with GT class)
  summary.json           Entity P/R/F1 + timing stats (det / recog / e2e)
  errors_visualized/     annotated PNGs of the images that contain errors
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rst.config import Config                                     # noqa: E402
from rst.data import RSTCollator                                  # noqa: E402
from rst.entity_extract import DetectionSet, load_rst_expert      # noqa: E402
from rst.eval import macro_f1                                     # noqa: E402
from rst.eval_checkpoint import cfg_from_ckpt                     # noqa: E402
from rst.train import load_semantic                               # noqa: E402
from tools.infer_image import build_items_for_image, ImageStore   # noqa: E402

DEFAULT_EXPERTS = "runs/v2_rs_s0,runs/v2_rt_adapt_s0,runs/v2_rst_f_s0"


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boxes", default="det_runs/testset_boxes.json")
    ap.add_argument("--experts", default=DEFAULT_EXPERTS)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--out-dir", default="infer_testset")
    ap.add_argument("--limit-images", type=int, default=0,
                    help="smoke: only the first N images")
    ap.add_argument("--visualize-errors", action="store_true", default=True)
    ap.add_argument("--no-visualize-errors", dest="visualize_errors",
                    action="store_false")
    ap.add_argument("--max-vis-side", type=int, default=1280)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda:%s" % args.gpu if torch.cuda.is_available() else "cpu")
    meta = json.load(open("index/class_meta.json", encoding="utf-8"))
    classes = meta["classes"]
    domain_names = ["military", "civilian"]

    det = json.load(open(args.boxes))["images"]

    # ground truth per test image
    gt = {}
    for line in open("index/instances.jsonl", encoding="utf-8"):
        d = json.loads(line)
        if d["split"] == "test":
            gt.setdefault(d["image"], []).append(
                ([float(v) for v in d["bbox"]], meta["class_index"][d["class"]]))
    if args.limit_images:
        keep = sorted(gt)[:args.limit_images]
        gt = {k: gt[k] for k in keep}

    # experts, loaded once
    cfg = Config()
    cfg.root = root
    cfg.use_mask_channel = True
    cfg.use_scene_cache = False
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
    print("experts ready:", len(pairs))

    predictions, correct, errors = {}, [], []
    tp = fp = fn = 0
    t_det = t_rec = 0.0
    n_imgs = len(gt)
    st = time.time()

    for ii, (key, gt_list) in enumerate(sorted(gt.items())):
        boxes = det.get(key, {}).get("boxes", [])
        t_det += det.get(key, {}).get("t_det", 0.0)
        rec_img = {"n_boxes": len(boxes), "entities": [], "missed": [], "t_recog": 0.0}
        if not boxes:
            for (gb, gc) in gt_list:
                fn += 1
                rec_img["missed"].append({"gt_class": classes[gc], "gt_bbox": gb})
                errors.append({"image": key, "type": "missed", "gt_class": classes[gc],
                               "gt_bbox": gb})
            predictions[key] = rec_img
            continue

        path = det[key]["file"]
        t0 = time.time()
        items, order, (W, H), kk = build_items_for_image(root, path, boxes, cfg)
        store = ImageStore(meta, kk, path, W, H, boxes)
        ds = DetectionSet(cfg, store, items)
        with torch.no_grad():
            loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0,
                                collate_fn=RSTCollator(ds))
            all_logits, all_dom = [], []
            for b in loader:
                b = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in b.items()}
                per, per_d = [], []
                for model, c in pairs:
                    out = model(b)
                    lc = out["logits_class"]
                    if c.use_text and "logits_semantic" in out:
                        lc = lc + out["logits_semantic"]
                    per.append(lc.float())
                    per_d.append(out["logits_domain"].float())
                all_logits.append(torch.stack(per, 0).cpu().numpy())
                all_dom.append(torch.stack(per_d, 0).mean(0))
        LOG = np.concatenate(all_logits, 1)                 # [E, N, C]
        fused = LOG.mean(0)
        pred = fused.argmax(1)
        conf = torch.softmax(torch.from_numpy(fused), -1).numpy().max(1)
        agree = np.stack([LOG[e].argmax(1) == pred for e in range(LOG.shape[0])]).mean(0)
        dom = torch.cat(all_dom).argmax(1).cpu().numpy()
        dt_rec = time.time() - t0
        t_rec += dt_rec
        rec_img["t_recog"] = round(dt_rec, 4)

        # greedy one-to-one matching by detector confidence
        idx = sorted(range(len(boxes)), key=lambda i: -boxes[i][4])
        used_gt = [False] * len(gt_list)
        for i in idx:
            box = [float(v) for v in boxes[i][:4]]
            best_v, best_j = 0.0, -1
            for j, (gb, gc) in enumerate(gt_list):
                if used_gt[j]:
                    continue
                v = iou(box, gb)
                if v > best_v:
                    best_v, best_j = v, j
            ent = {"bbox": box, "det_score": boxes[i][4],
                   "class": classes[int(pred[i])], "class_id": int(pred[i]),
                   "domain": domain_names[int(dom[i])],
                   "confidence": round(float(conf[i]), 4),
                   "expert_agreement": round(float(agree[i]), 2)}
            if best_j >= 0 and best_v >= args.iou:
                used_gt[best_j] = True
                gt_cls = classes[gt_list[best_j][1]]
                ent["iou"] = round(best_v, 3)
                ent["gt_class"] = gt_cls
                if pred[i] == gt_list[best_j][1]:
                    ent["match"] = "correct"
                    tp += 1
                    correct.append({"image": key, **{k: ent[k] for k in
                                    ("bbox", "class", "domain", "confidence",
                                     "expert_agreement")}})
                else:
                    ent["match"] = "class_error"
                    fp += 1
                    errors.append({"image": key, "type": "class_error",
                                   "gt_class": gt_cls, "pred_class": ent["class"],
                                   "iou": ent["iou"], "det_score": ent["det_score"],
                                   "confidence": ent["confidence"],
                                   "bbox": ent["bbox"]})
            else:
                ent["match"] = "false_pos"
                fp += 1
                errors.append({"image": key, "type": "false_pos",
                               "pred_class": ent["class"],
                               "det_score": ent["det_score"],
                               "confidence": ent["confidence"],
                               "bbox": ent["bbox"]})
            rec_img["entities"].append(ent)
        for j, (gb, gc) in enumerate(gt_list):
            if not used_gt[j]:
                fn += 1
                rec_img["missed"].append({"gt_class": classes[gc], "gt_bbox": gb})
                errors.append({"image": key, "type": "missed",
                               "gt_class": classes[gc], "gt_bbox": gb})
        predictions[key] = rec_img
        if ii % 100 == 0:
            print("  %d/%d  recog %.3fs" % (ii, n_imgs, dt_rec), flush=True)

    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    rec_t = [v["t_recog"] for v in predictions.values()]
    det_t = [det[k]["t_det"] for k in gt if k in det]
    e2e = [predictions[k]["t_recog"] + det.get(k, {}).get("t_det", 0.0) for k in gt]

    def stat(a):
        a = sorted(a)
        n = len(a)
        return {"mean": round(sum(a) / n, 4), "median": round(a[n // 2], 4),
                "p90": round(a[int(n * 0.9)], 4), "min": round(a[0], 4),
                "max": round(a[-1], 4)}

    err_by_type = {}
    for e in errors:
        err_by_type[e["type"]] = err_by_type.get(e["type"], 0) + 1
    summary = {
        "split": "test", "n_images": n_imgs, "n_gt_entities": tp + fn,
        "n_detections": len(correct) + err_by_type.get("class_error", 0)
                        + err_by_type.get("false_pos", 0),
        "entity": {"TP": tp, "FP": fp, "FN": fn, "precision": round(pr, 4),
                   "recall": round(rc, 4), "f1": round(f1, 4)},
        "errors_by_type": err_by_type,
        "correct_count": len(correct),
        "timing_sec": {"detection": stat(det_t), "recognition_per_image": stat(rec_t),
                       "end_to_end_per_image": stat(e2e),
                       "total_wall_min": round((time.time() - st) / 60, 1)},
        "experts": args.experts,
    }

    json.dump(predictions, open(os.path.join(args.out_dir, "predictions_all.json"),
                                "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(correct, open(os.path.join(args.out_dir, "correct_entities.json"),
                            "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(errors, open(os.path.join(args.out_dir, "error_entities.json"),
                           "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\nEntity  P %.4f  R %.4f  F1 %.4f  (TP %d FP %d FN %d)"
          % (pr, rc, f1, tp, fp, fn))
    print("errors:", err_by_type)
    print("timing: det %.3fs | recog %.3fs | e2e %.3fs (mean per image)"
          % (summary["timing_sec"]["detection"]["mean"],
             summary["timing_sec"]["recognition_per_image"]["mean"],
             summary["timing_sec"]["end_to_end_per_image"]["mean"]))

    # visualize images that contain errors
    if args.visualize_errors and errors:
        err_imgs = sorted({e["image"] for e in errors})
        vis_dir = os.path.join(args.out_dir, "errors_visualized")
        os.makedirs(vis_dir, exist_ok=True)
        print("visualizing %d error images -> %s" % (len(err_imgs), vis_dir))
        for key in err_imgs:
            im = Image.open(det[key]["file"]).convert("RGB")
            s = args.max_vis_side / max(im.size)
            if s < 1:
                im = im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR)
            dr = ImageDraw.Draw(im)
            for ent in predictions[key]["entities"]:
                x1, y1, x2, y2 = [v * s for v in ent["bbox"]]
                if ent["match"] == "correct":
                    col = (0, 170, 0)
                elif ent["match"] == "class_error":
                    col = (255, 140, 0)
                else:
                    col = (230, 0, 0)
                dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
                tag = "%s(%.2f)" % (ent["class"], ent["confidence"])
                if ent["match"] == "class_error":
                    tag += " != %s" % ent["gt_class"]
                dr.rectangle([x1, max(0, y1 - 15), x1 + dr.textlength(tag) + 6, y1],
                             fill=col)
                dr.text((x1 + 3, max(0, y1 - 14)), tag, fill=(255, 255, 255))
            for m in predictions[key]["missed"]:
                x1, y1, x2, y2 = [v * s for v in m["gt_bbox"]]
                dr.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=3)
                dr.text((x1 + 3, max(0, y1 + 2)), "MISSED %s" % m["gt_class"],
                        fill=(255, 215, 0))
            im.save(os.path.join(vis_dir, key + ".png"))
    print("done in %.1f min" % ((time.time() - st) / 60))


if __name__ == "__main__":
    main()
