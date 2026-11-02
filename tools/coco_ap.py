#!/usr/bin/env python
"""COCO-style AP for a class-agnostic single-class detector, from this project's JSONs.

表4-3 has two rows whose detector is not an ultralytics model -- Grounding DINO and
Qwen3-VL grounding -- so ultralytics' validator cannot score them. pycocotools is
already installed in .venv-det, so this wraps it around the two formats the project
already uses:

  ground truth : det_dataset/gt_classes.json
                 {image: {"split":..., "size":[W,H], "boxes":[[x1,y1,x2,y2,class_idx],...]}}
  detections   : {image: [[x1,y1,x2,y2,score], ...]}   (original pixel coordinates)

The detector is class-agnostic by design (4.4.2: it only localises aircraft), so every
ground-truth box is mapped to a single category and AP measures localisation only.
Entity Precision/Recall/F1, which additionally require the predicted type to be right,
come from rst/entity_extract.py instead.

  from tools.coco_ap import coco_ap
  print(coco_ap("det_dataset/gt_classes.json", boxes))
"""

import json


def coco_ap(gt_path, dt, split="test", verbose=True, rank_score=False):
    """`rank_score`: use the emission order inside each image as the confidence.

    Qwen3-VL grounding returns boxes with no score at all, and the project's writer
    fills 1.0. A flat score makes the COCO confidence sweep degenerate to a single
    operating point, so AP would measure nothing about ranking ability. The model does
    emit its boxes most-confident-first, so turning that order into a strictly
    decreasing score keeps AP well-defined without inventing a number; the table note
    says so.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)

    images, anns, img_ids = [], [], {}
    aid = 1
    for i, (image, info) in enumerate(sorted(gt.items())):
        if split and info.get("split") != split:
            continue
        W, H = info["size"]
        img_ids[image] = i + 1
        images.append({"id": i + 1, "width": int(W), "height": int(H),
                       "file_name": image})
        for box in info["boxes"]:
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            anns.append({"id": aid, "image_id": i + 1, "category_id": 1,
                         "bbox": [x1, y1, x2 - x1, y2 - y1],
                         "area": max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)),
                         "iscrowd": 0})
            aid += 1
    if not anns:
        raise SystemExit("no ground-truth boxes for split=%r in %s" % (split, gt_path))

    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns,
                    "categories": [{"id": 1, "name": "aircraft"}]}
    coco.createIndex()

    dts = []
    n_dt = 0
    for image, boxes in dt.items():
        iid = img_ids.get(image)
        if iid is None:
            continue
        k = len(boxes)
        for j, b in enumerate(boxes):
            x1, y1, x2, y2, sc = [float(v) for v in b[:5]]
            if rank_score:
                sc = (k - j) / float(k)
            dts.append({"image_id": iid, "category_id": 1, "score": sc,
                        "bbox": [x1, y1, x2 - x1, y2 - y1]})
            n_dt += 1
    if not dts:
        raise SystemExit("no detections matched any ground-truth image -- check the "
                         "image keys in the boxes file against gt_classes.json")

    res = {
        "n_images": len(images), "n_gt": len(anns), "n_det": n_dt,
        "det_per_image": n_dt / max(1, len(images)),
        "AP50": 0.0, "mAP50_95": 0.0,
    }
    for iou_type, key in (("bbox", "mAP50_95"),):
        coco_dt = coco.loadRes(dts)
        ev = COCOeval(coco, coco_dt, iou_type)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        res[key] = float(ev.stats[0])
        res["AP50"] = float(ev.stats[1])
        res["AP75"] = float(ev.stats[2])
        res["AP_small"] = float(ev.stats[3])
        res["AP_medium"] = float(ev.stats[4])
        res["AP_large"] = float(ev.stats[5])
    if verbose:
        print("AP50 %.4f | mAP@50:95 %.4f | AP75 %.4f | small/med/large %.4f/%.4f/%.4f"
              % (res["AP50"], res["mAP50_95"], res["AP75"],
                 res["AP_small"], res["AP_medium"], res["AP_large"]))
        print("images %d | GT boxes %d | detections %d (%.2f per image)"
              % (res["n_images"], res["n_gt"], res["n_det"], res["det_per_image"]))
    return res


def main():
    import argparse
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="det_dataset/gt_classes.json")
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--rank-score", action="store_true")
    ap.add_argument("--out", default="",
                    help="defaults to the boxes path with _summary appended, which is "
                         "the name tools/collect_paper_results.py looks for")
    args = ap.parse_args()

    with open(args.boxes, encoding="utf-8") as fh:
        dt = json.load(fh)
    res = coco_ap(args.gt, dt, args.split, rank_score=args.rank_score)
    res["boxes"] = args.boxes
    res["rank_score"] = bool(args.rank_score)
    out = args.out or (args.boxes[:-5] + "_summary.json"
                       if args.boxes.endswith(".json") else args.boxes + "_summary.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
