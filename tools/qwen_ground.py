#!/usr/bin/env python
"""表4-3 "Qwen3-VL" row: the MLLM does detection AND recognition in one pass.

No detector is involved -- the model is shown the whole image and asked for every
aircraft with a box and a type, which is what "直接使用 MLLM 的方法" means in 4.4.2.
Both AP (localisation) and Entity P/R/F1 (localisation + correct type) come out of the
same JSON.

The one thing that silently ruins this experiment is the coordinate space. Qwen-VL
models emit absolute coordinates in the image they were actually shown, which is the
processor's smart-resized version, not the original -- and these test images run up to
8508x7360. So this script resizes each image itself to a size that smart_resize will
leave alone (max side <= --max-side and both sides multiples of 28), records the scale,
and maps the boxes back. `--calibrate` verifies the mapping against ground truth on a
few images before committing to a full run, because a wrong mapping does not error --
it just yields AP near zero.

  .venv-det/bin/python -m tools.qwen_ground --calibrate --limit 12 --gpu 3
  .venv-det/bin/python -m tools.qwen_ground --gpu 3
"""

import argparse
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image

PROMPT = """请找出图中所有的飞机，并给出每架飞机的边界框与具体型号。

候选型号（label 只能从中选择，必须原样输出，不要翻译、不要加空格或连字符）：
{classes}

输出要求：
1. 严格输出一个 JSON 数组，不要输出任何解释、markdown 代码块标记或其他文字。
2. 每个元素格式为 {{"bbox_2d": [x1, y1, x2, y2], "label": "型号名"}}，坐标为左上角原点的像素坐标。
3. 图中有几架飞机就输出几个元素；没有飞机则输出 []。"""


def fit_size(w, h, max_side, multiple=28):
    """Largest (w', h') <= max_side with both sides a multiple of `multiple`.

    A size the processor's smart_resize will not touch, so the coordinates the model
    returns are in exactly this space and one scale factor maps them back.
    """
    s = min(1.0, max_side / float(max(w, h)))
    nw = max(multiple, int(round(w * s / multiple)) * multiple)
    nh = max(multiple, int(round(h * s / multiple)) * multiple)
    return nw, nh


def parse_boxes(text, classes, norm):
    """Extract boxes from a free-form answer. JSON first, then a bracket fallback,
    because models often wrap the array in prose or a code fence."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    start, end = t.find("["), t.rfind("]")
    if start >= 0 and end > start:
        t = t[start:end + 1]
    out = []
    try:
        data = json.loads(t)
    except Exception:
        data = None
    if isinstance(data, list):
        for d in data:
            if not isinstance(d, dict):
                continue
            box = d.get("bbox_2d") or d.get("bbox") or d.get("box")
            lab = d.get("label") or d.get("label_1") or d.get("class") or d.get("name")
            if not box or len(box) < 4:
                continue
            try:
                b = [float(v) for v in box[:4]]
            except (TypeError, ValueError):
                continue
            c = -1
            if isinstance(lab, str):
                lab = lab.strip()
                if lab in classes:
                    c = classes.index(lab)
                elif norm.get(lab.lower()) is not None:
                    c = classes.index(norm[lab.lower()])
            out.append((b, c, lab if isinstance(lab, str) else ""))
    return out


def _dump(path, boxes_out, stats=None):
    """Publish the shard atomically. `tools/watch_and_score_mllm.sh` polls for this file
    and a checkpoint write landing mid-scores would otherwise hand it half a JSON."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(boxes_out, fh)
    os.replace(tmp, path)
    if stats is not None:
        with open(path.replace(".json", "_stats.json"), "w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default="/home/wangyangfan/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--images", default="")
    ap.add_argument("--gt", default="det_dataset/gt_classes.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1568)
    ap.add_argument("--coord-scale", type=float, default=1000.0,
                    help="the model reports coordinates on a 0..coord_scale grid over "
                         "the image; Qwen3-VL uses 1000. Set to the fed width only if a "
                         "future model reports absolute pixels instead.")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-per-image", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split the image list this many ways; run one shard per free GPU "
                         "to halve the wall clock of a 5 s/image pass")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="keep whatever is already in the output file and skip those "
                         "images, so a crash or a preemption costs the unfinished tail "
                         "instead of the whole run")
    ap.add_argument("--merge", default="",
                    help="glob of shard box files to concatenate into --out, then exit. "
                         "Needs no GPU and no model.")
    ap.add_argument("--calibrate", action="store_true",
                    help="also report the mean IoU under both coordinate hypotheses, so a "
                         "wrong mapping is caught on 12 images instead of after 3 hours")
    ap.add_argument("--verbose-n", type=int, default=0,
                    help="print the raw model coordinates for the first N images next to "
                         "the original size, the size we fed it and the ground truth, "
                         "which is how the coordinate convention gets identified")
    ap.add_argument("--tag", default="qwen3vl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)

    def base_out():
        return args.out or "det_runs/%s_%s_boxes.json" % (args.tag, args.split)

    if args.merge:
        import glob
        # the per-shard summary and stats files share the shard's filename prefix, so a
        # shell-style glob picks them up too and their "n_images" key then looks like an
        # image that is in two shards at once. Filter by suffix, not just by prefix.
        all_files = sorted(glob.glob(args.merge))
        files = [f for f in all_files
                 if not (f.endswith("_summary.json") or f.endswith("_stats.json"))]
        if not files:
            raise SystemExit("--merge %r matched no shard box files (saw %d paths, all "
                             "summaries/stats)" % (args.merge, len(all_files)))
        merged = {}
        for f in files:
            with open(f, encoding="utf-8") as fh:
                part = json.load(fh)
            for k, v in part.items():
                if k in merged:
                    raise SystemExit("image %r is in more than one shard (%s) -- the "
                                     "shard split overlaps" % (k, f))
                merged[k] = v
        _dump(base_out(), merged)
        print("merged %d shards | %d images | %d boxes -> %s"
              % (len(files), len(merged), sum(len(v) for v in merged.values()), base_out()))
        return

    device = "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"

    from rst.config import Config
    from rst.data import EntityStore
    cfg = Config()
    cfg.root = root
    store = EntityStore(os.path.join(root, cfg.index_dir), args.split)
    classes = [str(c) for c in store.classes]
    norm = {}
    for c in classes:
        norm[c.lower()] = c
        norm[c.lower().replace("-", "").replace("_", "")] = c
    prompt = PROMPT.format(classes="、".join(classes))
    gt = json.load(open(args.gt, encoding="utf-8"))

    img_dir = args.images or os.path.join("det_dataset", "images", args.split)
    names = sorted(os.listdir(img_dir))
    if args.limit:
        names = names[:args.limit]
    if args.num_shards > 1:
        names = names[args.shard::args.num_shards]
    out = (base_out().replace(".json", "_shard%d.json" % args.shard)
           if args.num_shards > 1 and not args.out else base_out())
    if args.calibrate:
        # a 12-image calibration pass must never land on the file the table reads
        out = out.replace(".json", "_calib.json")
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    boxes_out = {}
    if args.resume and os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            boxes_out = json.load(fh)
        # a key is a receipt that the image was processed -- an aircraft-free image is
        # recorded as [] and must not be redone
        names = [n for n in names if os.path.splitext(n)[0] not in boxes_out]
        print("resume: %d images already in %s, %d left"
              % (len(boxes_out), out, len(names)))
    print("%d images | prompt %d chars | max side %d | shard %d/%d -> %s"
          % (len(names), len(prompt), args.max_side, args.shard, args.num_shards, out))
    if not names and boxes_out:
        _dump(out, boxes_out)
        print("nothing left to do, output republished")
        return

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print("loaded %s in %.0f s" % (args.model, time.time() - t0))

    def iou(a, b):
        ix = min(a[2], b[2]) - max(a[0], b[0])
        iy = min(a[3], b[3]) - max(a[1], b[1])
        if ix <= 0 or iy <= 0:
            return 0.0
        inter = ix * iy
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / u if u > 0 else 0.0

    stats = {"n_images": 0, "n_boxes": 0, "unparsed": 0, "empty": 0,
             "iou_scaled": [], "iou_asis": []}
    # A --resume must not restart the counters at zero or the summary understates the
    # run; all four are recoverable from the boxes already on disk.
    for rows in boxes_out.values():
        stats["n_images"] += 1
        stats["n_boxes"] += len(rows)
        stats["empty"] += int(not rows)
        stats["unparsed"] += sum(1 for r in rows if int(r[5]) < 0)
    t0 = time.time()
    for i, name in enumerate(names):
        key = os.path.splitext(name)[0]
        im = Image.open(os.path.join(img_dir, name)).convert("RGB")
        W, H = im.size
        nw, nh = fit_size(W, H, args.max_side)
        small = im.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (W, H) else im
        # Do NOT close anything here. When the image already sits on the 28-px grid
        # `small IS im`, and transformers 4.57's fast image processor only reads the
        # pixels inside apply_chat_template -- closing early raises
        # "ValueError: Operation on closed image" on the first such image, which is how
        # a 1,669-image run died at 79% with nothing written.
        msgs = [{"role": "user", "content": [{"type": "image", "image": small},
                                             {"type": "text", "text": prompt}]}]
        inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
        if small is not im:
            small.close()
        im.close()
        parsed = parse_boxes(text, classes, norm)
        if args.verbose_n and i < args.verbose_n:
            thw = inp.get("image_grid_thw")
            fed = None
            if thw is not None:
                _, gh, gw = [int(x) for x in thw[0].tolist()]
                ps = int(getattr(processor.image_processor, "patch_size", 16) or 16)
                fed = "%dx%d (grid %dx%d, patch %d)" % (gw * ps, gh * ps, gw, gh, ps)
            print("  [%s] original %dx%d | we fed %dx%d | processor saw %s"
                  % (key, W, H, nw, nh, fed))
            for b, c, lab in parsed[:3]:
                print("      model box %s label %r -> class %s"
                      % ([round(v, 1) for v in b], lab, c))
            for gbx in gt.get(key, {}).get("boxes", [])[:3]:
                print("      GT    box %s class %d"
                      % ([round(float(v), 1) for v in gbx[:4]], gbx[4]))
        if not parsed:
            stats["empty"] += 1
        stats["n_images"] += 1
        # Qwen3-VL emits coordinates normalised to a 0-1000 grid over the image it was
        # shown, NOT pixels of it. Verified against ground truth: a 550x390 image whose
        # aircraft box is [121,22,439,315] comes back as [220,54,798,807], which is
        # box/size*1000 to within one unit. Treating them as pixels of the resized image
        # (the Qwen2-VL convention) gives a mean best IoU of 0.22 instead of 0.85.
        sx, sy = W / float(args.coord_scale), H / float(args.coord_scale)
        rows = []
        for b, c, lab in parsed[:args.max_per_image]:
            if not c >= 0:
                stats["unparsed"] += 1
            rows.append([round(b[0] * sx, 1), round(b[1] * sy, 1),
                         round(b[2] * sx, 1), round(b[3] * sy, 1), 1.0, int(c)])
            if args.calibrate and key in gt:
                g = [x[:4] for x in gt[key]["boxes"]]
                if g:
                    best_s = max(iou([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], x) for x in g)
                    best_a = max(iou(b, x) for x in g)
                    stats["iou_scaled"].append(best_s)
                    stats["iou_asis"].append(best_a)
        boxes_out[key] = rows
        stats["n_boxes"] += len(rows)
        if (i + 1) % 25 == 0 or i + 1 == len(names):
            el = time.time() - t0
            print("  %d/%d images  %.2f s/img  eta %.0f min  boxes %d  empty %d  bad-label %d"
                  % (i + 1, len(names), el / (i + 1), (len(names) - i - 1) * el / (i + 1) / 60,
                     stats["n_boxes"], stats["empty"], stats["unparsed"]), flush=True)
            # checkpoint first, report second: a pass that dies at image 1,325 must still
            # have 1,325 images on disk to resume from
            _dump(out, boxes_out, stats)
            if args.calibrate and stats["iou_scaled"]:
                print("     calibration: mean best IoU  scaled-back %.3f   as-is %.3f"
                      % (float(np.mean(stats["iou_scaled"])),
                         float(np.mean(stats["iou_asis"]))), flush=True)

    _dump(out, boxes_out, stats)
    summary = {"model": args.model, "split": args.split, "max_side": args.max_side,
               "coord_scale": args.coord_scale,
               "boxes_file": out, "sec_per_image": (time.time() - t0) / max(1, len(names)),
               **{k: v for k, v in stats.items() if not isinstance(v, list)}}
    if stats["iou_scaled"]:
        summary["calib_mean_iou_scaled_back"] = round(float(np.mean(stats["iou_scaled"])), 4)
        summary["calib_mean_iou_as_is"] = round(float(np.mean(stats["iou_asis"])), 4)
    with open(out.replace(".json", "_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print("\nwrote %s | %d images, %d boxes (%.2f/img), %d empty answers, %d bad labels"
          % (out, stats["n_images"], stats["n_boxes"],
             stats["n_boxes"] / max(1, stats["n_images"]), stats["empty"], stats["unparsed"]))
    if stats["iou_scaled"]:
        print("coordinate check: mean best IoU against GT  scaled-back %.3f  as-is %.3f"
              % (float(np.mean(stats["iou_scaled"])), float(np.mean(stats["iou_asis"]))))


if __name__ == "__main__":
    main()
