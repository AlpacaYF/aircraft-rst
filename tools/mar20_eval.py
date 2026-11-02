#!/usr/bin/env python
"""MAR20 cross-dataset evaluation for the deployed RST-DEDF (GT boxes).

MAR20 has 20 classes; 16 map to the 114-class model, 4 are absent
(Su-35 / E-3 / E-8 / KC-10). This script classifies every GT crop with the
three uniform-supervision experts (or a single expert) and reports:
  * Top-1 / Top-5 / macro-F1 over the 16 mapped classes
  * per-class accuracy and the dominant confusions
  * what the model predicts for the 4 absent classes (nearest-class behaviour)
  * a per-instance CSV and a markdown report

  .venv/bin/python tools/mar20_eval.py --mar20 mar20 --split test \
      --arch dedf --models runs/v2_rs_s0,runs/v2_rt_nosup_s0,runs/v2_rst_nosup_s0
"""

import argparse
import collections
import csv
import importlib.util
import json
import os
import random
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rst.config import Config                                    # noqa: E402
from rst.data import RSTCollator                                  # noqa: E402
from rst.entity_extract import DetectionSet, load_rst_expert      # noqa: E402
from rst.eval_checkpoint import cfg_from_ckpt                     # noqa: E402
from rst.train import load_semantic                               # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_bdi", os.path.join(ROOT, "scripts", "build_dataset_index.py"))
_bdi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bdi)

MAP = {
    "A2": "C130", "A3": "C17", "A4": "C5", "A5": "F16", "A6": "Tu160",
    "A8": "B52", "A9": "P3", "A10": "B1", "A12": "Tu22M", "A13": "F15",
    "A14": "KC135", "A15": "F22", "A16": "F18", "A17": "Tu95",
    "A19": "Su34", "A20": "Su24",
}
MISSING = {"A1": "Su-35", "A7": "E-3", "A11": "E-8", "A18": "KC-10"}


def find_dir(base, name, must_have_ext=""):
    for root, dirs, files in os.walk(base):
        if os.path.basename(os.path.normpath(root)) == name:
            if not must_have_ext or any(f.lower().endswith(must_have_ext) for f in files):
                return root
    return ""


def parse_xml(path):
    t = ET.parse(path).getroot()
    W = int(float(t.findtext("size/width")))
    H = int(float(t.findtext("size/height")))
    objs = []
    for o in t.findall("object"):
        nm = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if not nm or bb is None:
            continue
        box = [float(bb.findtext("xmin")), float(bb.findtext("ymin")),
               float(bb.findtext("xmax")), float(bb.findtext("ymax"))]
        objs.append((nm, box))
    return W, H, objs


class Mar20Store:
    def __init__(self, meta, records):
        self.classes = meta["classes"]
        self.class_index = meta["class_index"]
        self.domain_of = meta["domain"]
        self.train = False
        self.instances = []
        self.by_image = {}
        self.scenes = {}
        for key, rec in records.items():
            W, H = rec["size"]
            self.by_image[key] = [{"image": key, "scene_file": rec["path"],
                                   "image_size": [W, H]}]
            boxes = [b for _, b in rec["objs"]]
            plan = _bdi.plan_scene(W, H, boxes)
            rects = plan.get("tiles", [])
            if plan.get("mode") != "tiled":
                rects = [{"x": 0, "y": 0, "w": W, "h": H, "near": True, "gi": 0, "gj": 0}]
            plan = dict(plan)
            plan["grid_nx"] = max((t["gi"] for t in rects), default=0) + 1
            plan["grid_ny"] = max((t["gj"] for t in rects), default=0) + 1
            self.scenes[key] = {"plan": plan, "rects": rects}


def build_items(records, store, cfg):
    items = []
    for key, rec in records.items():
        W, H = rec["size"]
        boxes = [b for _, b in rec["objs"]]
        for j, (code, box) in enumerate(rec["objs"]):
            others = [o for k, o in enumerate(boxes) if k != j]
            name = MAP.get(code)
            cls = store.class_index[name] if name else -1
            dom = 0
            if name:
                d = store.domain_of[name]
                dom = d if isinstance(d, int) else (0 if str(d).startswith("mil") else 1)
            items.append({
                "rec": {"image": key, "scene_file": rec["path"], "image_size": [W, H],
                        "bbox": box,
                        "bbox_expanded": _bdi.protected_box(box, others, W, H,
                                                            cfg.region_expand),
                        "instance_index": j, "split": "mar20"},
                "flip": False, "class": cls, "domain": dom,
                "instance_id": "%s#%d" % (key, j), "score": 1.0,
                "code": code,
            })
    return items


def iou(a, b):
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def build_det_items(records, det, store, cfg):
    items = []
    for key, rec in records.items():
        boxes = sorted(det.get(key, {}).get("boxes", []), key=lambda b: -b[4])
        W, H = rec["size"]
        raws = [list(map(float, b[:4])) for b in boxes]
        for j, box in enumerate(raws):
            others = [o for k, o in enumerate(raws) if k != j]
            items.append({
                "rec": {"image": key, "scene_file": rec["path"], "image_size": [W, H],
                        "bbox": box,
                        "bbox_expanded": _bdi.protected_box(box, others, W, H,
                                                            cfg.region_expand),
                        "instance_index": 1000 + j, "split": "mar20det"},
                "flip": False, "class": -1, "domain": 0,
                "instance_id": "%s#det%d" % (key, j), "score": float(boxes[j][4]),
                "code": "DET",
            })
    return items


def match_det(items, pred, records, meta, iou_th=0.5):
    """Greedy one-to-one detection match, highest score first; entity correct
    only when the matched GT class is mapped AND the prediction equals it."""
    by_img = {}
    for i, it in enumerate(items):
        by_img.setdefault(it["rec"]["image"], []).append(i)
    n_gt = matched = fp = missed = tp_e = mapped_matched = 0
    per_cls, per_hit = collections.Counter(), collections.Counter()
    det_gt = [None] * len(items)
    for key, idxs in by_img.items():
        gts = records[key]["objs"]
        used = [False] * len(gts)
        idxs.sort(key=lambda i: -items[i]["score"])
        for i in idxs:
            best, bj = 0.0, -1
            for j, (code, gbox) in enumerate(gts):
                if not used[j]:
                    v = iou(items[i]["rec"]["bbox"], gbox)
                    if v > best:
                        best, bj = v, j
            if bj >= 0 and best >= iou_th:
                used[bj] = True
                matched += 1
                code = gts[bj][0]
                det_gt[i] = code
                name = MAP.get(code)
                if name:
                    mapped_matched += 1
                    per_cls[code] += 1
                    if pred[i] == meta["class_index"][name]:
                        tp_e += 1
                        per_hit[code] += 1
            else:
                fp += 1
        for j, (code, _) in enumerate(gts):
            n_gt += 1
            if not used[j]:
                missed += 1
    for key, rec in records.items():
        if key not in by_img:              # images with zero detections
            n_gt += len(rec["objs"])
            missed += len(rec["objs"])
    return {"n_det": len(items), "n_gt": n_gt, "matched": matched, "fp": fp,
            "missed": missed, "tp_e": tp_e, "mapped_matched": mapped_matched,
            "per_cls": per_cls, "per_hit": per_hit, "det_gt": det_gt}


def macro_f1(pred, gt, mapped_mask, n_cls):
    fs = []
    for c in range(n_cls):
        support = int(((gt == c) & mapped_mask).sum())
        if support == 0:
            continue
        tp = int(((pred == c) & (gt == c) & mapped_mask).sum())
        fp = int(((pred == c) & (gt != c) & mapped_mask).sum())
        fn = int(((gt == c) & (pred != c) & mapped_mask).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs)) if fs else 0.0


@torch.no_grad()
def classify(pairs, ds, cfg, device, batch):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0,
                        collate_fn=RSTCollator(ds))
    outs = []
    for bi, b in enumerate(loader):
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        per = []
        for model, c in pairs:
            with torch.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
                out = model(b)
            lc = out["logits_class"]
            if c.use_text and "logits_semantic" in out:
                lc = lc + out["logits_semantic"]
            per.append(lc.float())
        outs.append(torch.stack(per, 0).mean(0).cpu().numpy())
        if bi % 25 == 0:
            print("  classified %d batches" % (bi + 1), flush=True)
    return np.concatenate(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mar20", default="mar20")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--ids-file", default="",
                    help="optional file with one image id per line (e.g. a stratified subset)")
    ap.add_argument("--arch", default="dedf", choices=["dedf", "rst"])
    ap.add_argument("--models", default="runs/v2_rs_s0,runs/v2_rt_nosup_s0,runs/v2_rst_nosup_s0")
    ap.add_argument("--run", default="runs/v2_rst_nosup_s0")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--conf", type=float, default=0.5,
                    help="detector confidence recorded in the report (boxes already filtered)")
    ap.add_argument("--gpu", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="limit images (debug)")
    ap.add_argument("--det-json", default="",
                    help="evaluate detection boxes (mar20_detect.py output) instead of GT boxes")
    ap.add_argument("--out", default="mar20/results")
    args = ap.parse_args()

    root = ROOT
    os.chdir(root)
    base = os.path.abspath(args.mar20)
    img_dir = find_dir(base, "JPEGImages", ".jpg")
    ann_dir = find_dir(base, "Horizontal Bounding Boxes", ".xml")
    split_path = ""
    for r, _, fs in os.walk(base):
        for f in fs:
            if f == "%s.txt" % args.split and os.path.basename(r) == "Main":
                split_path = os.path.join(r, f)
    if args.ids_file:
        split_path = args.ids_file
    if not (img_dir and ann_dir and split_path):
        raise SystemExit("cannot locate JPEGImages / Horizontal Bounding Boxes / %s under %s"
                         % (args.ids_file or ("Main/%s.txt" % args.split), base))
    print("images:", img_dir)
    print("annotations:", ann_dir)
    print("split:", split_path)

    ids = [l.strip() for l in open(split_path, encoding="utf-8", errors="replace")
           if l.strip()]
    if args.limit:
        ids = ids[:args.limit]
    records = {}
    for i in ids:
        xml = os.path.join(ann_dir, i + ".xml")
        img = os.path.join(img_dir, i + ".jpg")
        if not (os.path.exists(xml) and os.path.exists(img)):
            continue
        W, H, objs = parse_xml(xml)
        if objs:
            records[i] = {"path": img, "size": (W, H), "objs": objs}
    print("images with boxes: %d | instances: %d"
          % (len(records), sum(len(r["objs"]) for r in records.values())))

    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    cfg = Config()
    cfg.root = root
    cfg.use_mask_channel = True
    cfg.use_scene_cache = False
    cfg.num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu != "auto":
        device = torch.device("cuda:%s" % args.gpu)
    print("device:", device)

    run_dirs = (args.models.split(",") if args.arch == "dedf" else [args.run])
    pairs = []
    for mdir in run_dirs:
        md = mdir if os.path.isabs(mdir) else os.path.join(root, mdir)
        p = torch.load(os.path.join(md, "best.pt"), map_location=device,
                       weights_only=False)
        c = cfg_from_ckpt(p, root)
        c.use_scene_cache = False
        pr, _, _ = load_semantic(c)
        pairs.append((load_rst_expert(p, c, pr, device, tag=os.path.basename(md)), c))
        print("  expert:", md)

    store = Mar20Store(meta, records)
    if args.det_json:
        det = json.load(open(args.det_json, encoding="utf-8"))["images"]
        items = build_det_items(records, det, store, cfg)
    else:
        items = build_items(records, store, cfg)
    ds = DetectionSet(cfg, store, items)
    t0 = time.time()
    LOG = classify(pairs, ds, cfg, device, args.batch)
    dt = time.time() - t0

    if args.det_json:
        names = meta["classes"]
        pred = LOG.argmax(1)
        conf = torch.softmax(torch.from_numpy(LOG), -1).numpy().max(1)
        m = match_det(items, pred, records, meta)
        n_det, n_gt = m["n_det"], m["n_gt"]
        matched, fp, missed, tp_e = m["matched"], m["fp"], m["missed"], m["tp_e"]
        det_p = matched / n_det if n_det else 0.0
        det_r = matched / n_gt if n_gt else 0.0
        det_f = 2 * det_p * det_r / (det_p + det_r) if det_p + det_r else 0.0
        ent_p = tp_e / n_det if n_det else 0.0
        ent_r = tp_e / n_gt if n_gt else 0.0
        ent_f = 2 * ent_p * ent_r / (ent_p + ent_r) if ent_p + ent_r else 0.0
        recog_all = tp_e / matched if matched else 0.0
        recog_mapped = tp_e / m["mapped_matched"] if m["mapped_matched"] else 0.0
        gt_cls = collections.Counter()
        for rec in records.values():
            for code, _ in rec["objs"]:
                gt_cls[code] += 1
        rows = []
        for code, n in sorted(gt_cls.items()):
            if code in MISSING:
                continue
            rows.append({"class": MAP[code], "gt": int(n),
                         "hit": int(m["per_hit"][code]),
                         "recall": round(m["per_hit"][code] / n, 4) if n else 0.0})
        outdir = os.path.abspath(args.out)
        os.makedirs(outdir, exist_ok=True)
        summary = {
            "mode": "end_to_end", "det_json": args.det_json,
            "n_images": len(records), "n_gt": n_gt, "n_det": n_det,
            "matched": matched, "fp": fp, "missed": missed,
            "detection_precision": round(det_p, 4), "detection_recall": round(det_r, 4),
            "detection_f1": round(det_f, 4),
            "entity_precision": round(ent_p, 4), "entity_recall": round(ent_r, 4),
            "entity_f1": round(ent_f, 4), "tp_e": tp_e,
            "recognition_on_matched_all": round(recog_all, 4),
            "recognition_on_matched_mapped16": round(recog_mapped, 4),
            "per_class_e2e": rows, "timing_s": round(dt, 1),
            "thresholds": {"det_conf": args.conf, "iou": 0.5},
        }
        with open(os.path.join(outdir, "e2e_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        with open(os.path.join(outdir, "e2e_detections.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "bbox", "score", "pred", "matched_gt", "correct"])
            for i, it in enumerate(items):
                code = m["det_gt"][i]
                name = MAP.get(code) if code else None
                ok = 1 if (name and pred[i] == meta["class_index"][name]) else 0
                w.writerow([it["rec"]["image"], " ".join("%.1f" % v for v in it["rec"]["bbox"]),
                            "%.3f" % it["score"], names[pred[i]], code or "",
                            ok])
        md = ["# MAR20 end-to-end（YOLO + DEDF）", "",
              "- 图像 %d，GT %d，检测 %d，匹配 %d，FP %d，漏检 %d"
              % (len(records), n_gt, n_det, matched, fp, missed),
              "- **检测 P %.2f%% / R %.2f%% / F1 %.2f%%**"
              % (100 * det_p, 100 * det_r, 100 * det_f),
              "- **实体 P %.2f%% / R %.2f%% / F1 %.2f%%**（缺失类必然错）"
              % (100 * ent_p, 100 * ent_r, 100 * ent_f),
              "- 匹配框上的型号识别率：全部 %.2f%%，16 个可映射类 %.2f%%"
              % (100 * recog_all, 100 * recog_mapped), "",
              "## 逐类端到端召回", "", "| 类 | GT | 正确 | 召回 |", "|---|---|---|---|"]
        for r in rows:
            md.append("| %s | %d | %d | %.2f%% |" % (r["class"], r["gt"], r["hit"],
                                                     100 * r["recall"]))
        open(os.path.join(outdir, "e2e_report.md"), "w", encoding="utf-8").write("\n".join(md))
        print("\n== MAR20 end-to-end ==")
        print("images %d | GT %d | det %d | matched %d | FP %d | missed %d"
              % (len(records), n_gt, n_det, matched, fp, missed))
        print("detection P %.4f R %.4f F1 %.4f" % (det_p, det_r, det_f))
        print("entity    P %.4f R %.4f F1 %.4f | recog(all/mapped) %.4f / %.4f"
              % (ent_p, ent_r, ent_f, recog_all, recog_mapped))
        print("wrote", outdir)
        return

    gt = np.array([it["class"] for it in items])
    codes = np.array([it["code"] for it in items])
    mapped = gt >= 0
    pred = LOG.argmax(1)
    top5 = np.argsort(-LOG, 1)[:, :5]
    conf = torch.softmax(torch.from_numpy(LOG), -1).numpy().max(1)
    names = meta["classes"]

    acc16 = float((pred[mapped] == gt[mapped]).mean())
    top5_16 = float((top5[mapped] == gt[mapped, None]).any(1).mean())
    # restricted-16: renormalise over the 16 mapped class logits only, the fair
    # "16-way" comparison against a task that has 20 classes (16 supported)
    idx16 = [meta["class_index"][n] for n in MAP.values()]
    sub = LOG[:, idx16]
    pred16 = np.array(idx16)[sub.argmax(1)]
    acc16r = float((pred16[mapped] == gt[mapped]).mean())
    acc20 = float((pred == gt).mean())  # missing classes can never be 'correct'
    mf = macro_f1(pred, gt, mapped, len(names))
    print("\n== MAR20 %s (GT boxes) ==" % args.split)
    print("instances %d | mapped16 %d (%.1f%%) | missing4 %d"
          % (len(items), int(mapped.sum()), 100 * mapped.mean(), int((~mapped).sum())))
    print("Top-1 (16 mapped, 114-class head) %.4f | Top-5 %.4f | restricted-16 %.4f"
          % (acc16, top5_16, acc16r))
    print("acc(all20, missing=wrong) %.4f | macroF1(16) %.4f" % (acc20, mf))
    print("time %.1f s (%.2f ms/instance)" % (dt, 1000 * dt / max(1, len(items))))

    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    # per-class
    rows = []
    for c in range(len(names)):
        m = (gt == c) & mapped
        if m.sum() == 0:
            continue
        rows.append({"class": names[c], "n": int(m.sum()),
                     "acc": float((pred[m] == c).mean())})
    rows.sort(key=lambda r: -r["n"])
    with open(os.path.join(outdir, "per_class.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["class", "n", "acc"])
        w.writeheader(); w.writerows(rows)

    # confusions (wrong only)
    confu = collections.Counter()
    for i in np.where(mapped & (pred != gt))[0]:
        confu[(names[gt[i]], names[pred[i]])] += 1

    # absent classes: prediction distribution
    absent = {}
    for code, label in MISSING.items():
        idx = np.where(codes == code)[0]
        if len(idx) == 0:
            continue
        cnt = collections.Counter(names[p] for p in pred[idx])
        absent[label] = {"n": int(len(idx)),
                         "top": [[k, int(v)] for k, v in cnt.most_common(10)]}

    # per-instance csv
    with open(os.path.join(outdir, "predictions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "gt_code", "gt_name", "pred", "confidence",
                    "correct", "top5"])
        for i, it in enumerate(items):
            name = MAP.get(it["code"])
            w.writerow([it["rec"]["image"], it["code"], name or MISSING.get(it["code"], ""),
                        names[pred[i]], "%.4f" % conf[i],
                        int(bool(name)) and int(pred[i] == gt[i]),
                        "|".join(names[t] for t in top5[i])])

    summary = {
        "split": args.split,
        "n_images": len(records),
        "n_instances": len(items),
        "n_mapped16": int(mapped.sum()),
        "n_missing4": int((~mapped).sum()),
        "top1_16": round(acc16, 4),
        "top5_16": round(top5_16, 4),
        "top1_16_restricted16": round(acc16r, 4),
        "acc_all20_missing_wrong": round(acc20, 4),
        "macro_f1_16": round(mf, 4),
        "timing_s": round(dt, 1),
        "ms_per_instance": round(1000 * dt / max(1, len(items)), 2),
        "per_class": rows,
        "top_confusions": [[a, b, int(c)] for (a, b), c in confu.most_common(20)],
        "absent_classes": absent,
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    # markdown
    md = ["# MAR20 %s 评测（GT 框，DEDF 统一监督配方）" % args.split, "",
          "- 图像 %d 张，实例 %d 个（16 个可映射类 %d 个，4 个缺失类 %d 个）"
          % (len(records), len(items), mapped.sum(), (~mapped).sum()),
          "- **Top-1（16 类，114 类头）%.2f%%**，限制在 16 个候选类时 %.2f%%，"
          "Top-5 %.2f%%，Macro-F1 %.2f%%"
          % (100 * acc16, 100 * acc16r, 100 * top5_16, 100 * mf),
          "- 把 4 个缺失类也计入（必然错）：%.2f%%" % (100 * acc20),
          "- 推理 %.1f s（%.2f ms/实例）" % (dt, 1000 * dt / max(1, len(items))), "",
          "## 逐类", "", "| 类 | n | Top-1 |", "|---|---|---|"]
    for r in rows:
        md.append("| %s | %d | %.2f%% |" % (r["class"], r["n"], 100 * r["acc"]))
    md += ["", "## 主要混淆", "", "| GT → 预测 | 次数 |", "|---|---|"]
    for (a, b), c in confu.most_common(15):
        md.append("| %s → %s | %d |" % (a, b, c))
    md += ["", "## 缺失 4 类的预测分布", ""]
    for k, v in absent.items():
        top = "、".join("%s×%d" % (a, b) for a, b in v["top"][:6])
        md.append("- **%s**（n=%d）：%s" % (k, v["n"], top))
    open(os.path.join(outdir, "report.md"), "w", encoding="utf-8").write("\n".join(md))

    # a few visualisations
    vis_dir = os.path.join(outdir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    by_img = collections.defaultdict(list)
    for i, it in enumerate(items):
        by_img[it["rec"]["image"]].append(i)
    picks = random.Random(0).sample(list(by_img), min(20, len(by_img)))
    for key in picks:
        rec = records[key]
        im = Image.open(rec["path"]).convert("RGB")
        dr = ImageDraw.Draw(im)
        for i in by_img[key]:
            x1, y1, x2, y2 = items[i]["rec"]["bbox"]
            ok = bool(MAP.get(items[i]["code"])) and pred[i] == gt[i]
            col = (0, 200, 0) if ok else (230, 30, 30)
            dr.rectangle([x1, y1, x2, y2], outline=col, width=2)
            tag = "%s→%s" % (items[i]["code"], names[pred[i]])
            dr.rectangle([x1, max(0, y1 - 13), x1 + 8 * len(tag), y1], fill=col)
            dr.text((x1 + 1, max(0, y1 - 12)), tag, fill=(255, 255, 255))
        im.save(os.path.join(vis_dir, "%s.jpg" % key))
    print("\nwrote", outdir)


if __name__ == "__main__":
    main()
