#!/usr/bin/env python
"""Encode the 114 class descriptions (text modality) with Qwen3-VL-Embedding.

One vector per class: "ClassName: description-sentence". Run with qwen3vl_env.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_vl_embedding import Qwen3VLEmbedder  # noqa: E402

INSTRUCTION = ("Represent this aircraft type description for fine-grained "
               "classification and discrimination from similar types.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--model", default=os.path.expanduser("~/models/Qwen3-VL-Embedding-8B"))
    ap.add_argument("--strings", default="knowledge/kb_strings_en.json")
    ap.add_argument("--out", default="index/emb_qwen3vle_text_classes.npz")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    kb = json.load(open(os.path.join(root, args.strings), encoding="utf-8"))
    classes = [str(c) for c in kb["classes"]]
    # first desc sentence per class, else first item of any kind, else the name
    best = {}
    for r, k, t in kb["items"]:
        r = int(r)
        if r not in best or (k == "desc" and best[r][0] != "desc"):
            best[r] = (k, str(t))
    texts = []
    for i, c in enumerate(classes):
        kind_txt = best.get(i)
        texts.append("%s: %s" % (c, kind_txt[1]) if kind_txt else c)
    print("texts:", len(texts), "| e.g.", texts[0][:90])

    model = Qwen3VLEmbedder(model_name_or_path=args.model, torch_dtype="bfloat16")
    emb = model.process([{"text": t, "instruction": INSTRUCTION} for t in texts],
                        normalize=True)
    X = emb.float().cpu().numpy().astype(np.float32)
    np.savez_compressed(os.path.join(root, args.out),
                        features=X, classes=np.array(classes, dtype=object),
                        texts=np.array(texts, dtype=object), instruction=INSTRUCTION)
    print("wrote %s  shape %s" % (args.out, X.shape))


if __name__ == "__main__":
    main()
