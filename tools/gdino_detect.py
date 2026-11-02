#!/usr/bin/env python
"""表4-3 "Grounding DINO + VLM": the detection half.

Grounding DINO is an open-vocabulary detector, so it fits 4.4.2's requirement of a
class-agnostic aircraft locator without any fine-tuning on this dataset -- it is given
a text prompt and returns boxes. That is also the honest way to run it: fine-tuning it
on our boxes would make it a second supervised detector and the row would no longer
represent "Grounding DINO + VLM".

Boxes come out in ORIGINAL pixel coordinates (target_sizes is passed to the
post-processor), so no coordinate mapping is needed and the output is directly
comparable with the ultralytics detectors' boxes files.

The VLM half is tools/vlm_classify_boxes.py; Entity P/R/F1 then comes from
rst/entity_extract.py --pred-json, which uses the same tested matcher as every other
row.

  .venv-det/bin/python -m tools.gdino_detect --split test --gpu 1
"""

import argparse
import json
import os
import time

import torch
from PIL import Image

from tools.coco_ap import coco_ap

# Grounding DINO wants lowercase noun phrases separated by " . " and a trailing period.
DEFAULT_PROMPT = "airplane . aircraft . jet . airliner . military aircraft ."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default="", help="defaults to det_dataset/images/<split>")
    ap.add_argument("--gt", default="det_dataset/gt_classes.json")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="box score floor for the saved boxes; AP integrates over all "
                         "thresholds, so a low floor is right here. The Entity metrics "
                         "apply their own floor later.")
    ap.add_argument("--text-threshold", type=float, default=0.10)
    ap.add_argument("--max-per-image", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N images")
    ap.add_argument("--tag", default="gdino_tiny")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    device = "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"

    from transformers import AutoProcessor, GroundingDinoForObjectDetection
    t0 = time.time()
    model = GroundingDinoForObjectDetection.from_pretrained(args.model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print("loaded %s on %s in %.0f s | prompt %r"
          % (args.model, device, time.time() - t0, args.prompt))

    img_dir = args.images or os.path.join("det_dataset", "images", args.split)
    names = sorted(os.listdir(img_dir))
    if args.limit:
        names = names[:args.limit]
    print("%d images from %s" % (len(names), img_dir))

    boxes, t0 = {}, time.time()
    with torch.no_grad():
        for start in range(0, len(names), args.batch):
            chunk = names[start:start + args.batch]
            imgs, sizes = [], []
            for n in chunk:
                im = Image.open(os.path.join(img_dir, n)).convert("RGB")
                imgs.append(im)
                sizes.append(im.size[::-1])          # (H, W) as target_sizes wants
            inputs = processor(images=imgs, text=[args.prompt] * len(imgs),
                               return_tensors="pt").to(device)
            out = model(**inputs)
            res = processor.post_process_grounded_object_detection(
                out, inputs["input_ids"], threshold=args.threshold,
                text_threshold=args.text_threshold,
                target_sizes=torch.tensor(sizes, device=device))
            for n, r in zip(chunk, res):
                b = r["boxes"].cpu().tolist()
                s = r["scores"].cpu().tolist()
                order = sorted(range(len(s)), key=lambda i: -s[i])[:args.max_per_image]
                boxes[os.path.splitext(n)[0]] = [
                    [round(b[i][0], 1), round(b[i][1], 1), round(b[i][2], 1),
                     round(b[i][3], 1), round(float(s[i]), 4)] for i in order]
            for im in imgs:
                im.close()
            done = start + len(chunk)
            if done % (args.batch * 20) < args.batch or done == len(names):
                el = time.time() - t0
                print("  %d/%d images  %.2f s/img  eta %.0f min  boxes so far %d"
                      % (done, len(names), el / done, (len(names) - done) * el / done / 60,
                         sum(len(v) for v in boxes.values())), flush=True)

    os.makedirs("det_runs", exist_ok=True)
    out_boxes = os.path.join("det_runs", "%s_%s_boxes.json" % (args.tag, args.split))
    with open(out_boxes, "w", encoding="utf-8") as fh:
        json.dump(boxes, fh)
    print("wrote %s (%d images, %d boxes)"
          % (out_boxes, len(boxes), sum(len(v) for v in boxes.values())))

    if args.limit:
        print("smoke test (--limit %d): skipping AP over the full split" % args.limit)
        return
    print("\n=== AP against %s (class-agnostic, single category) ===" % args.gt)
    ap_res = coco_ap(args.gt, boxes, split=args.split)
    summary = {"model": args.model, "prompt": args.prompt, "split": args.split,
               "box_threshold": args.threshold, "text_threshold": args.text_threshold,
               "max_per_image": args.max_per_image, "boxes_file": out_boxes,
               "AP50": ap_res["AP50"], "mAP50_95": ap_res["mAP50_95"],
               "n_det": ap_res["n_det"], "n_gt": ap_res["n_gt"],
               "sec_per_image": (time.time() - t0) / max(1, len(names))}
    with open(out_boxes.replace(".json", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("wrote", out_boxes.replace(".json", "_summary.json"))


if __name__ == "__main__":
    main()
