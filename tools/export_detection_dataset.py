#!/usr/bin/env python
"""Export the entity index as a single-class aircraft detection dataset.

表4-8 evaluates entity extraction: a detector localises aircraft and the RST module
decides the型号, so every box is labelled class 0 here no matter what its fine-grained
class is. The fine-grained labels are written alongside in gt_classes.json, because
Entity Precision/Recall/F1 count a match only when IoU >= 0.5 AND the predicted class
is correct -- the detector alone cannot be scored against that.

Images are symlinked rather than copied: the split is 8.1 GB and duplicating it inside
the project directory would buy nothing.

  python -m tools.export_detection_dataset --out det_dataset
"""

import argparse
import json
import os

from rst.config import Config
from rst.data import EntityStore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="det_dataset")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    out = os.path.join(root, args.out)

    store = EntityStore(os.path.join(root, "index"), None)
    gt_classes = {}
    counts = {}
    for split in ("train", "val", "test"):
        img_dir = os.path.join(out, "images", split)
        lbl_dir = os.path.join(out, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n_img = n_box = 0
        # the index uses "validation" internally, the detector split is called "val"
        key = "validation" if split == "val" else split
        for image, recs in store.by_image.items():
            if recs[0]["split"] != key:
                continue
            src = os.path.join(root, recs[0]["scene_file"])
            if not os.path.exists(src):
                print("MISSING source image %s -- skipped" % src)
                continue
            W, H = recs[0]["image_size"]
            link = os.path.join(img_dir, image + ".jpg")
            if not os.path.exists(link):
                os.symlink(src, link)
            lines, boxes = [], []
            for r in recs:
                x1, y1, x2, y2 = [float(v) for v in r["bbox"]]
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                w, h = (x2 - x1) / W, (y2 - y1) / H
                if w <= 0 or h <= 0:
                    continue
                lines.append("0 %.6f %.6f %.6f %.6f" % (cx, cy, w, h))
                boxes.append([x1, y1, x2, y2, store.class_index[r["class"]]])
            with open(os.path.join(lbl_dir, image + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            gt_classes[image] = {"split": split, "size": [W, H], "boxes": boxes}
            n_img += 1
            n_box += len(lines)
        counts[split] = (n_img, n_box)
        print("%-6s %5d images %6d boxes -> %s" % (split, n_img, n_box, img_dir))

    with open(os.path.join(out, "gt_classes.json"), "w", encoding="utf-8") as fh:
        json.dump(gt_classes, fh)
    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("path: %s\n" % out)
        for split in ("train", "val", "test"):
            fh.write("%s: images/%s\n" % (split, split))
        fh.write("names:\n  0: aircraft\n")
    print("wrote %s and gt_classes.json (%d images)" % (yaml_path, len(gt_classes)))

    # a detector trained on this must be checked against the same boxes the RST
    # module is scored on, so make the mapping between the two explicit
    tiny = sum(1 for v in gt_classes.values() for b in v["boxes"]
               if max(b[2] - b[0], b[3] - b[1]) < 32)
    total = sum(len(v["boxes"]) for v in gt_classes.values())
    print("boxes under 32 px long side: %d / %d (%.1f%%) -- these dominate Entity Recall"
          % (tiny, total, 100 * tiny / max(1, total)))


if __name__ == "__main__":
    main()
