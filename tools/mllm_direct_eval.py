#!/usr/bin/env python
"""表4-3 / 表4-8 的 MLLM 直接识别行：给图 + 114 个候选类别，让模型直接报型号。

Runs in .venv-det (transformers 5.x), NOT in the training venv: the training
environment is pinned at transformers 4.49.0 because that is the version the frozen
SigLIP 2 text tower was called through when knowledge/text_features.npz was built,
and Qwen3-VL needs 5.x.

The prompt is a module-level constant because 4.4.1 promises "各模型采用统一的结构化
Prompt" -- every MLLM row has to be scored with exactly this text, and the thesis has
to be able to quote it. Top-5 is asked for explicitly: the table has a Top-5 column,
and a generative model has no logits to take a top-k from.

  .venv-det/bin/python -m tools.mllm_direct_eval --model ~/models/Qwen3-VL-8B-Instruct \
      --mode region --limit 8            # smoke test first
  .venv-det/bin/python -m tools.mllm_direct_eval --model ~/models/Qwen3-VL-8B-Instruct \
      --mode region --gpu 1
"""

import argparse
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image

from rst.config import Config
from rst.data import EntityDataset, EntityStore, build_region

PROMPT = """你是航空装备识别专家。下面是{scene_note}一架飞机的图像，请从给定的 114 个候选类别中判断它的具体型号。

候选类别（只能从中选择，必须原样输出，不要翻译、不要加空格或连字符）：
{classes}

要求：
1. 输出最可能的 5 个类别，按置信度从高到低排列。
2. 严格只输出一行，格式为：1. 类别名 2. 类别名 3. 类别名 4. 类别名 5. 类别名
3. 不要输出任何解释、标点或其他文字。"""

SCENE_NOTE = {"region": "", "region_scene": "（第一张是飞机局部区域，第二张是它所在的完整场景）"}


def parse_top5(text, classes, norm):
    """Map a free-form answer onto the candidate list. Exact match first, then a
    normalised match; anything else is an explicit miss rather than a silent guess."""
    out = []
    for tok in re.split(r"[,，;；\n\r\t]+|\s*\d+[.)、]\s*", text):
        t = tok.strip().strip(".。:：")
        if not t:
            continue
        if t in classes:
            out.append(t)
        elif norm.get(t.lower()):
            out.append(norm[t.lower()])
        if len(out) == 5:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", required=True, help="local dir or HF id of the MLLM")
    ap.add_argument("--mode", default="region", choices=["region", "region_scene"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N entities")
    ap.add_argument("--scene-long-side", type=int, default=1024)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    cfg = Config()
    cfg.root = root
    device = "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    print("loaded %s on %s in %.0f s" % (args.model, device, time.time() - t0))

    store = EntityStore(os.path.join(root, cfg.index_dir), args.split)
    classes = [str(c) for c in store.classes]
    norm = {}
    for c in classes:
        norm[c.lower()] = c
        norm[c.lower().replace("-", "").replace("_", "")] = c
    prompt = PROMPT.format(scene_note=SCENE_NOTE[args.mode], classes="、".join(classes))

    ds = EntityDataset(cfg, args.split, train=False)
    recs = ds.records[:args.limit] if args.limit else ds.records
    print("%s split %s: %d entities, mode %s" % (args.split, args.split, len(recs), args.mode))

    def region_of(rec):
        img = Image.open(os.path.join(root, rec["scene_file"])).convert("RGB")
        W, H = rec["image_size"]
        region, _ = build_region(img, W, H, rec, False, cfg, jitter=False)
        # build_region returns a [-1,1] tensor with an optional mask channel; the MLLM
        # wants a plain PIL image, so undo the normalisation and drop the mask
        arr = ((region[:3].permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
        if args.mode == "region_scene":
            sc = img.copy()
            sc.thumbnail((args.scene_long_side, args.scene_long_side))
            img.close()
            return out, sc
        img.close()
        return out, None

    preds, ys, ids, raws = [], [], [], []
    t0 = time.time()
    for start in range(0, len(recs), args.batch):
        chunk = recs[start:start + args.batch]
        convs, meta = [], []
        for rec in chunk:
            region, scene = region_of(rec)
            content = [{"type": "image", "image": region}]
            if scene is not None:
                content.append({"type": "image", "image": scene})
            content.append({"type": "text", "text": prompt})
            convs.append([{"role": "user", "content": content}])
            meta.append((rec, store.class_index[rec["class"]]))
        try:
            inputs = processor.apply_chat_template(
                convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", padding=True).to(device)
        except Exception as exc:
            # batching support varies across processor versions; fall back to one at a
            # time rather than losing the whole run
            print("batched call failed (%s); retrying one by one" % type(exc).__name__)
            inputs = None
        if inputs is None:
            outs = []
            for c in convs:
                i1 = processor.apply_chat_template(
                    [c], tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt").to(device)
                with torch.no_grad():
                    g = model.generate(**i1, max_new_tokens=args.max_new_tokens,
                                       do_sample=False)
                outs.append(processor.batch_decode(
                    g[:, i1["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        else:
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     do_sample=False)
            outs = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
        for (rec, y), text in zip(meta, outs):
            top = parse_top5(text, set(classes), norm)
            preds.append(top)
            ys.append(y)
            ids.append("%s#%d" % (rec["image"], rec["instance_index"]))
            raws.append(text.strip()[:200])
        done = start + len(chunk)
        if done % (args.batch * 25) < args.batch or done == len(recs):
            el = time.time() - t0
            print("  %d/%d  %.2f s/entity  eta %.0f min"
                  % (done, len(recs), el / done, (len(recs) - done) * el / done / 60),
                  flush=True)

    ys = np.asarray(ys)
    top1 = np.asarray([p[0] == classes[y] if p else False for p, y in zip(preds, ys)])
    top5 = np.asarray([classes[y] in p for p, y in zip(preds, ys)])
    unparsed = sum(1 for p in preds if not p)
    mil = np.isin(ys, list(store.military_index))
    cidx = {c: i for i, c in enumerate(classes)}
    pred_idx = np.asarray([cidx[p[0]] if p else -1 for p in preds])

    f1, present = [], []
    for c in range(len(classes)):
        m = ys == c
        if not m.any():
            continue
        tp = int(((pred_idx == c) & m).sum())
        fp = int(((pred_idx == c) & ~m).sum())
        fn = int(((pred_idx != c) & m).sum())
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        present.append(c)

    res = {"model": args.model, "mode": args.mode, "split": args.split, "n": int(len(ys)),
           "top1": float(top1.mean()), "top5": float(top5.mean()),
           "macro_f1": float(np.mean(f1)) if f1 else 0.0,
           "military_top1": float(top1[mil].mean()) if mil.any() else 0.0,
           "military_macro_f1": float(np.mean(
               [f for f, c in zip(f1, present) if c in set(store.military_index)]))
           if f1 else 0.0,
           "unparsed": unparsed, "sec_per_entity": (time.time() - t0) / max(1, len(ys)),
           "prompt_chars": len(prompt)}
    print("\n%s [%s] Top-1 %.2f  Top-5 %.2f  MacroF1 %.2f | mil Top-1 %.2f mil F1 %.2f "
          "| unparsed %d"
          % (os.path.basename(args.model), args.mode, 100 * res["top1"], 100 * res["top5"],
             100 * res["macro_f1"], 100 * res["military_top1"],
             100 * res["military_macro_f1"], unparsed))

    out = args.out or os.path.join(
        root, cfg.out_dir,
        "mllm_%s_%s.json" % (os.path.basename(args.model.rstrip("/")).lower(), args.mode))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"summary": res, "ids": ids, "y": ys.tolist(),
                   "pred": preds, "raw": raws}, fh, ensure_ascii=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
