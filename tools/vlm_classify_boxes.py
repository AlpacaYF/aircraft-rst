#!/usr/bin/env python
"""表4-3 "Grounding DINO + VLM": the VLM half.

Takes a detector's boxes and asks the MLLM for the type of each one, producing the
per-box class predictions that `rst/entity_extract.py --pred-json` turns into Entity
Precision / Recall / F1 with the same one-to-one greedy matcher every other row uses.

Crops are built with `rst.data.build_region` and the same 25% context expansion with
neighbour protection (`protected_box`) that the index builder used, so the VLM sees a
region constructed exactly like the one our own model and the 表4-2 Qwen3-VL row see.
The only difference is where the box came from.

The prompt and the answer parser are imported from tools.mllm_direct_eval so this row
and the 表4-2 rows are scored with literally the same instruction text -- 4.4.1
promises "各模型采用统一的结构化 Prompt".

  .venv-det/bin/python -m tools.vlm_classify_boxes \
      --boxes det_runs/gdino_tiny_test_boxes.json --top-k 3 --gpu 2
"""

import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from rst.config import Config
from rst.data import EntityStore, build_region
from tools.mllm_direct_eval import PROMPT, SCENE_NOTE, parse_top5

_spec = importlib.util.spec_from_file_location(
    "_bdi", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "scripts", "build_dataset_index.py"))
_bdi = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bdi)
protected_box = _bdi.protected_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--boxes", required=True,
                    help="{image: [[x1,y1,x2,y2,score], ...]} in original pixels")
    ap.add_argument("--model", default="/home/wangyangfan/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--top-k", type=int, default=3,
                    help="boxes per image to classify. A uniform K is used for every "
                         "detector row because score calibrations are not comparable "
                         "across detectors (Grounding DINO's maximum score on this set "
                         "is 0.58, so a 0.5 floor would leave it with almost nothing). "
                         "Ground truth averages 1.8 entities per image.")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N images")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    cfg = Config()
    cfg.root = root
    device = "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"

    with open(args.boxes, encoding="utf-8") as fh:
        det = json.load(fh)
    store = EntityStore(os.path.join(root, cfg.index_dir), args.split)
    classes = [str(c) for c in store.classes]
    cidx = {c: i for i, c in enumerate(classes)}
    norm = {}
    for c in classes:
        norm[c.lower()] = c
        norm[c.lower().replace("-", "").replace("_", "")] = c
    prompt = PROMPT.format(scene_note=SCENE_NOTE["region"], classes="、".join(classes))

    # flatten to (image, box, score) keeping only the top-K per image
    tasks = []
    for image, recs in store.by_image.items():
        boxes = sorted(det.get(image, []), key=lambda b: -float(b[4]))[:args.top_k]
        if not boxes:
            continue
        W, H = recs[0]["image_size"]
        raws = [list(map(float, b[:4])) for b in boxes]
        for j, box in enumerate(raws):
            others = [o for k, o in enumerate(raws) if k != j]
            tasks.append((image, recs[0], (W, H), box,
                          protected_box(box, others, W, H, cfg.region_expand),
                          float(boxes[j][4])))
    if args.limit:
        kept, seen = [], set()
        for t in tasks:
            seen.add(t[0])
            if len(seen) > args.limit:
                break
            kept.append(t)
        tasks = kept
    print("%d boxes over %d images (top-%d per image)"
          % (len(tasks), len({t[0] for t in tasks}), args.top_k))
    if not tasks:
        raise SystemExit("no boxes to classify")

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print("loaded %s in %.0f s" % (args.model, time.time() - t0))

    preds, raws_out, unparsed = [], [], 0
    t0 = time.time()
    for start in range(0, len(tasks), args.batch):
        chunk = tasks[start:start + args.batch]
        convs, imgs = [], []
        for image, rec0, WH, box, exp, sc in chunk:
            img = Image.open(os.path.join(root, rec0["scene_file"])).convert("RGB")
            region, _ = build_region(img, WH[0], WH[1],
                                     {"bbox": box, "bbox_expanded": exp}, False, cfg,
                                     jitter=False)
            img.close()
            arr = ((region[:3].permute(1, 2, 0).numpy() + 1.0) * 127.5
                   ).clip(0, 255).astype(np.uint8)
            imgs.append(Image.fromarray(arr))
            convs.append([{"role": "user", "content": [
                {"type": "image", "image": imgs[-1]}, {"type": "text", "text": prompt}]}])
        try:
            inputs = processor.apply_chat_template(
                convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            outs = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        except Exception as exc:
            print("batched call failed (%s); one at a time" % type(exc).__name__)
            outs = []
            for c in convs:
                i1 = processor.apply_chat_template(
                    [c], tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt").to(device)
                with torch.no_grad():
                    g = model.generate(**i1, max_new_tokens=args.max_new_tokens,
                                       do_sample=False)
                outs.append(processor.batch_decode(g[:, i1["input_ids"].shape[1]:],
                                                   skip_special_tokens=True)[0])
        for text in outs:
            top = parse_top5(text, set(classes), norm)
            if not top:
                unparsed += 1
                preds.append(-1)
            else:
                preds.append(cidx[top[0]])
            raws_out.append(text.strip()[:160])
        done = start + len(chunk)
        if done % (args.batch * 25) < args.batch or done == len(tasks):
            el = time.time() - t0
            print("  %d/%d boxes  %.2f s/box  eta %.0f min  unparsed %d"
                  % (done, len(tasks), el / done, (len(tasks) - done) * el / done / 60,
                     unparsed), flush=True)

    # group predictions back per image, keeping the detector's box and score
    out = {}
    for (image, rec0, WH, box, exp, sc), pi in zip(tasks, preds):
        out.setdefault(image, []).append([round(box[0], 1), round(box[1], 1),
                                          round(box[2], 1), round(box[3], 1),
                                          round(sc, 4), int(pi)])
    path = args.out or args.boxes.replace(".json", "_vlm.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "boxes": args.boxes, "top_k": args.top_k,
                   "n_boxes": len(tasks), "unparsed": unparsed,
                   "prompt_chars": len(prompt), "sec_per_box": (time.time() - t0) / max(1, len(tasks)),
                   "predictions": out}, fh, ensure_ascii=False)
    print("\nwrote %s | %d boxes, %d unparsed (%.1f%%), %.2f s/box"
          % (path, len(tasks), unparsed, 100 * unparsed / max(1, len(tasks)),
             (time.time() - t0) / max(1, len(tasks))))


if __name__ == "__main__":
    main()
