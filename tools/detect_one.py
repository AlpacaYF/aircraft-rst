#!/usr/bin/env python
"""Single-image detection for the integrated pipeline. Runs in .venv-det
(ultralytics lives there and must not sit next to the training venv's pinned
transformers). Output: a small JSON with the boxes in original pixels.

  .venv-det/bin/python tools/detect_one.py --image <path> --out boxes.json
"""

import argparse
import json
import os

DEFAULT_WEIGHTS = "runs/detect/det_runs/yolo11x_640_b16/weights/best.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from ultralytics import RTDETR, YOLO
    model = (RTDETR(args.weights) if "rtdetr" in args.weights.lower()
             else YOLO(args.weights))
    r = model.predict(source=args.image, conf=args.conf, imgsz=args.imgsz,
                      verbose=False)[0]
    boxes = []
    if r.boxes is not None and len(r.boxes):
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            boxes.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                          round(float(b.conf[0]), 4)])
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"image": os.path.abspath(args.image), "weights": args.weights,
                   "conf": args.conf, "boxes": boxes}, fh)
    print("detector: %d boxes (conf>=%.2f) -> %s" % (len(boxes), args.conf, args.out))


if __name__ == "__main__":
    main()
