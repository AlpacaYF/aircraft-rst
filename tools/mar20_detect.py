#!/usr/bin/env python
"""YOLO11x detection over MAR20 images (runs in .venv-det).

  .venv-det/bin/python tools/mar20_detect.py --mar20 mar20 --split test \
      --out mar20/results/boxes.json --conf 0.5
"""

import argparse
import json
import os
import time

DEFAULT_WEIGHTS = "runs/detect/det_runs/yolo11x_640_b16/weights/best.pt"


def find_dir(base, name, ext):
    for root, dirs, files in os.walk(base):
        if os.path.basename(os.path.normpath(root)) == name:
            if any(f.lower().endswith(ext) for f in files):
                return root
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mar20", default="mar20")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ids-file", default="",
                    help="optional file with one image id per line (stratified subset)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="mar20/results/boxes.json")
    args = ap.parse_args()

    root = os.path.abspath(".")
    base = os.path.abspath(args.mar20)
    img_dir = find_dir(base, "JPEGImages", ".jpg")
    split = ""
    for r, _, fs in os.walk(base):
        if os.path.basename(r) == "Main" and ("%s.txt" % args.split) in fs:
            split = os.path.join(r, "%s.txt" % args.split)
    if args.ids_file:
        split = args.ids_file
    ids = [l.strip() for l in open(split, encoding="utf-8", errors="replace") if l.strip()]
    if args.limit:
        ids = ids[:args.limit]
    print("images:", len(ids), "| dir:", img_dir)

    from ultralytics import YOLO
    model = YOLO(args.weights)
    out = {}
    t_all = time.time()
    for i, key in enumerate(ids):
        p = os.path.join(img_dir, key + ".jpg")
        if not os.path.exists(p):
            continue
        t0 = time.time()
        r = model.predict(source=p, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                              round(float(b.conf[0]), 4)])
        out[key] = {"file": p, "boxes": boxes, "t_det": round(time.time() - t0, 4)}
        if i % 200 == 0:
            print("  %d/%d" % (i, len(ids)), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
                   "n_images": len(out), "images": out}, fh)
    ts = [v["t_det"] for v in out.values()]
    print("wrote %s | mean %.3fs | total %.1f min"
          % (args.out, sum(ts) / max(1, len(ts)), (time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
