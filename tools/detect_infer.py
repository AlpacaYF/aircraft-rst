#!/usr/bin/env python
"""Run a trained detector over the test images: 表4-8's AP columns + the boxes file.

Two outputs, because the table needs both and they come from different passes:
  * AP50 / mAP@50:95 from ultralytics' own COCO-style validator on the test split
  * <out>.json = {image: [[x1, y1, x2, y2, score], ...]} in ORIGINAL pixel coords,
    which `rst.entity_extract` feeds to the RST classifier to get Entity P/R/F1

Boxes are written at a deliberately low confidence floor (--conf, default 0.05) so the
entity metrics can be recomputed at any threshold later without running the detector
again. The threshold actually used for the paper's numbers is chosen by
`rst.entity_extract --score` and recorded in its output.

Runs in .venv-det, not the training venv: ultralytics lives there and must not be
installed next to transformers 4.49.0, which the frozen SigLIP 2 text features depend
on.

  .venv-det/bin/python -m tools.detect_infer \
      --weights runs/detect/det_runs/yolo11x_640/weights/best.pt --split test
"""

import argparse
import json
import os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="det_dataset/data.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.05,
                    help="floor for the boxes file; the paper's threshold is applied "
                         "later by rst.entity_extract --score")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="", help="output tag; defaults to the weights dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import RTDETR, YOLO
    # ultralytics dispatches on the checkpoint, but asking for the right class keeps
    # the RT-DETR head/decoder path explicit instead of relying on inference
    model = RTDETR(args.weights) if "rtdetr" in os.path.basename(args.weights).lower() \
        or "rtdetr" in args.weights.lower() else YOLO(args.weights)
    tag = args.name or os.path.basename(os.path.dirname(os.path.dirname(args.weights)))

    print("=== AP on the %s split (%s) ===" % (args.split, args.weights))
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                  device=args.device, project="det_runs", name=tag + "_val",
                  exist_ok=True, verbose=False)
    ap50, ap5095 = float(m.box.map50), float(m.box.map)
    print("AP50 %.4f  mAP@50:95 %.4f  (precision %.4f recall %.4f)"
          % (ap50, ap5095, float(m.box.mp), float(m.box.mr)))

    src = os.path.join(os.path.dirname(args.data), "images", args.split)
    names = sorted(os.listdir(src))
    print("=== boxes for %d %s images at conf >= %.2f ===" % (len(names), args.split,
                                                              args.conf))
    boxes, n_box = {}, 0
    for i, r in enumerate(model.predict(source=src, stream=True, conf=args.conf,
                                        imgsz=args.imgsz, device=args.device,
                                        verbose=False)):
        key = os.path.splitext(os.path.basename(r.path))[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                out.append([round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                            round(float(b.conf[0]), 4)])
        boxes[key] = out
        n_box += len(out)
        if (i + 1) % 200 == 0:
            print("  %d/%d images, %d boxes" % (i + 1, len(names), n_box), flush=True)

    out_path = args.out or os.path.join("det_runs", "%s_%s_boxes.json" % (tag, args.split))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(boxes, fh)
    summary = {"weights": args.weights, "split": args.split, "imgsz": args.imgsz,
               "conf_floor": args.conf, "AP50": ap50, "mAP50_95": ap5095,
               "n_images": len(boxes), "n_boxes": n_box,
               "boxes_per_image": n_box / max(1, len(boxes)), "boxes_file": out_path}
    with open(out_path.replace(".json", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\nwrote %s (%d images, %d boxes, %.2f per image)"
          % (out_path, len(boxes), n_box, summary["boxes_per_image"]))
    print("AP50 %.4f | mAP@50:95 %.4f" % (ap50, ap5095))


if __name__ == "__main__":
    main()
