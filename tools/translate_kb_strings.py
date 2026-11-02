#!/usr/bin/env python
"""Translate the knowledge base sentences to English with the local MLLM.

Why this exists: the MLLM-generated knowledge is 1891/1892 Chinese sentences of ~20
characters each. CLIP-ViT-L/14 -- the only text tower measured here that can separate
these 114 types at all (35.68% zero-shot from class names) -- uses an English BPE
tokenizer, so Chinese sentences are shredded into byte pieces and its prototypes
retrieve at 8.00% instead. SigLIP 2 is multilingual but cannot separate the types in
either language (names 1.23%, knowledge 0.33%).

So before spending ~8 GPU-hours retraining the Text branch on a different tower, the
knowledge has to be in a language that tower can actually read. This produces that
text; `rst.build_text_features --strings` then encodes it.

Text-only generation, no images: the MLLM is used as a translator here, which is a
different role from the offline knowledge generation of 3.3.1.

  .venv-det/bin/python -m tools.translate_kb_strings --gpu 2 --batch 20
"""

import argparse
import json
import os
import re

import numpy as np
import torch

PROMPT = """把下列飞机外观描述逐条翻译成英文，用于图文检索模型的文本输入。
要求：
1. 严格保持条数与编号，每行一条，格式为「编号. 英文」。
2. 只输出翻译结果，不要解释、不要额外文字。
3. 术语用航空领域通用英文（如 后掠翼 swept wing、鸭翼 canard、双发 twin-engine、
   背负式 dorsal/over-fuselage、翼下挂架 underwing pylon）。
4. 型号名保持原样（如 J20、Su34、EF2000、Boeing787 不要展开或改写）。
5. 输出必须是纯英文，任何中文字符都不允许出现在译文里；没有把握时用意思最接近的
   英文描述代替，不要保留原文片段。

{lines}"""


def parse_block(text, n, start_idx):
    """Map numbered output lines back onto the input order; fall back to position."""
    out = [None] * n
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)\s*[.)、]\s*(.+?)\s*$", line)
        if not m:
            continue
        i = int(m.group(1)) - start_idx
        if 0 <= i < n and out[i] is None:
            out[i] = m.group(2).strip()
    filled = 0
    loose = [l.strip(" 。.;；") for l in text.splitlines() if l.strip()]
    for i in range(n):
        if out[i] is None:
            if filled < len(loose):
                out[i] = re.sub(r"^\d+\s*[.)、]\s*", "", loose[filled])
            filled += 1
    return [o if o else "" for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--src", default="knowledge/text_features_clipL.npz",
                    help="npz whose `strings`/`kinds`/`rows` define what to translate")
    ap.add_argument("--model", default="/home/wangyangfan/models/Qwen3-VL-8B-Instruct")
    ap.add_argument("--out", default="knowledge/kb_strings_en.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    os.chdir(root)
    device = "cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu"

    z = np.load(args.src, allow_pickle=True)
    strings = [str(s) for s in z["strings"]]
    kinds = [str(k) for k in z["kinds"]]
    rows = [int(r) for r in z["rows"]]
    classes = [str(c) for c in z["classes"]]
    if args.limit:
        strings, kinds, rows = strings[:args.limit], kinds[:args.limit], rows[:args.limit]
    print("%d sentences to translate from %s" % (len(strings), args.src))

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)

    translated = []
    for start in range(0, len(strings), args.batch):
        chunk = strings[start:start + args.batch]
        lines = "\n".join("%d. %s" % (start + i + 1, s) for i, s in enumerate(chunk))
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(lines=lines)}]}]
        inp = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                            return_dict=True, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0]
        got = parse_block(text, len(chunk), start + 1)
        translated.extend(got)
        empty = sum(1 for g in got if not g)
        print("  %d/%d translated (%d empty in this block)"
              % (len(translated), len(strings), empty), flush=True)

    # One retry pass over whatever still contains CJK, then strip any survivors: a
    # leftover Chinese fragment is worse than a slightly vaguer English phrase, because
    # an English BPE tokenizer turns it into byte noise.
    cjk = re.compile(r"[\u4e00-\u9fff]")
    bad = [i for i, t in enumerate(translated) if cjk.search(t)]
    if bad:
        print("retrying %d sentences that kept Chinese fragments" % len(bad), flush=True)
        for i in bad:
            msgs = [{"role": "user", "content": [{"type": "text", "text":
                     "Translate to English. Output English only, no Chinese characters, "
                     "no explanation:\n%s" % strings[i]}]}]
            inp = processor.apply_chat_template(msgs, tokenize=True,
                                                add_generation_prompt=True,
                                                return_dict=True,
                                                return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=120, do_sample=False)
            out = processor.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                                         skip_special_tokens=True)[0].strip()
            if out and not cjk.search(out):
                translated[i] = out
        translated = [cjk.sub(" ", t).replace("  ", " ").strip() for t in translated]

    n_empty = sum(1 for t in translated if not t)
    # leftover CJK characters would be shredded by an English BPE tokenizer, which is
    # the exact failure this translation is meant to remove -- so count them
    n_cjk = sum(1 for t in translated if re.search(r"[一-鿿]", t))
    out = {"source": args.src, "translator": args.model, "classes": classes,
           "items": [[r, k, t] for r, k, t in zip(rows, kinds, translated)],
           "n_empty": n_empty, "n_with_cjk_leftover": n_cjk}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("wrote %s | %d sentences, %d empty, %d still contain CJK (%.1f%%)"
          % (args.out, len(translated), n_empty, n_cjk, 100 * n_cjk / max(1, len(translated))))
    for i in range(min(4, len(translated))):
        print("  %s\n    -> %s" % (strings[i], translated[i]))


if __name__ == "__main__":
    main()
