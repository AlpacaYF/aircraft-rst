#!/usr/bin/env python
"""Batch detection over the whole test split with the YOLO11x detector.
Runs in .venv-det. Times every image individually.

  .venv-det/bin/python tools/detect_batch.py --conf 0.5 \
      --out det_runs/testset_boxes.json
"""

import argparse
import json
import os
import time

DEFAULT_WEIGHTS = "runs/detect/det_runs/yolo11x_640_b16/weights/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="det_runs/testset_boxes.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)

    images = {}
    with open("index/instances.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            if d["split"] == args.split:
                images.setdefault(d["image"], d["scene_file"])
    print("test images:", len(images))

    from ultralytics import YOLO
    model = YOLO(args.weights)

    out = {}
    t_all = time.time()
    for i, (key, path) in enumerate(sorted(images.items())):
        t0 = time.time()
        r = model.predict(source=path, conf=args.conf, imgsz=args.imgsz,
                          verbose=False)[0]
        dt = time.time() - t0
        boxes = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                              round(float(b.conf[0]), 4)])
        out[key] = {"file": path, "boxes": boxes, "t_det": round(dt, 4)}
        if i % 100 == 0:
            print("  %d/%d  last %.3fs" % (i, len(images), dt), flush=True)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
                   "n_images": len(out), "images": out}, fh)
    ts = [v["t_det"] for v in out.values()]
    print("wrote %s | det mean %.3fs  median %.3fs  total %.1f min"
          % (args.out, sum(ts) / len(ts), sorted(ts)[len(ts) // 2],
             (time.time() - t_all) / 60))


if __name__ == "__main__":
    main()
