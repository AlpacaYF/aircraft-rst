#!/usr/bin/env python
"""Frozen-feature classifiers on top of the Qwen3-VL-Embedding probe features.

Three protocols, all trained ONLY on the train-split embeddings and evaluated on
the full test split (3002 entities), same metrics as the main pipeline:
  1. class-mean prototype   (nearest cosine prototype, no training)
  2. 1-NN                   (nearest training embedding)
  3. linear / MLP heads     (torch, inverse-frequency CE)

  .venv/bin/python -m tools.emb_probe_classify --npz index/emb_probe_qwen3vle.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rst.eval import macro_f1  # noqa: E402


def class_index(root):
    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    return meta["classes"], meta["class_index"]


def topk_acc(sim, y, k):
    pred = np.argsort(-sim, axis=1)[:, :k]
    return float((pred == y[:, None]).any(1).mean())


def report(tag, sim, y, n_classes):
    pred = sim.argmax(1)
    f1 = macro_f1(y, pred, n_classes)[0]
    print("%-34s Top-1 %6.2f  Top-5 %6.2f  MacroF1 %6.2f"
          % (tag, 100 * (pred == y).mean(), 100 * topk_acc(sim, y, 5), 100 * f1))
    return dict(top1=float((pred == y).mean()), top5=topk_acc(sim, y, 5), f1=f1)


def train_head(Xtr, ytr, Xte, n_classes, hidden=0, epochs=60, lr=1e-3, wd=1e-4,
               batch=256, class_weight=None, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Xtr.shape[1]
    net = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.2),
                         nn.Linear(hidden, n_classes))
           if hidden else nn.Linear(d, n_classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    cw = None
    if class_weight is not None:
        cw = torch.tensor(class_weight, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=cw)
    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    n = len(ytr_t)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = crit(net(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        sim = net(torch.tensor(Xte, device=dev)).float().cpu().numpy()
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--npz", default="index/emb_probe_qwen3vle.npz")
    ap.add_argument("--out", default="runs/emb_probe_report.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    classes, cidx = class_index(root)
    n_classes = len(classes)

    z = np.load(os.path.join(root, args.npz), allow_pickle=True)
    X, y_lab, splits = z["features"], z["labels"], z["splits"]
    y = np.array([cidx[str(c)] for c in y_lab])
    tr, te = splits == "train", splits == "test"
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    print("probe: model=%s | dim %d | train %d | test %d"
          % (z["model"], X.shape[1] if False else Xte.shape[1], len(ytr), len(yte)))

    results = {}

    # 1) class-mean prototype
    proto = np.zeros((n_classes, Xtr.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = ytr == c
        if m.any():
            v = Xtr[m].mean(0)
            proto[c] = v / max(1e-8, np.linalg.norm(v))
    results["prototype"] = report("class-mean prototype (no training)",
                                  Xte @ proto.T, yte, n_classes)

    # 2) 1-NN
    sim = Xte @ Xtr.T
    nn_pred = ytr[sim.argmax(1)]
    f1 = macro_f1(yte, nn_pred, n_classes)[0]
    print("%-34s Top-1 %6.2f  Top-5 %6.2f  MacroF1 %6.2f"
          % ("1-NN (cosine)", 100 * (nn_pred == yte).mean(), float("nan"), 100 * f1))
    results["1nn"] = dict(top1=float((nn_pred == yte).mean()), f1=f1)

    # 3) linear + MLP with inverse-frequency weights
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.clip(counts, 1, None), 0.0)
    w = w / w[w > 0].mean()
    results["linear"] = report("linear head (frozen feats)",
                               train_head(Xtr, ytr, Xte, n_classes, hidden=0,
                                          class_weight=w),
                               yte, n_classes)
    results["mlp"] = report("MLP 1024 (frozen feats)",
                            train_head(Xtr, ytr, Xte, n_classes, hidden=1024,
                                       class_weight=w),
                            yte, n_classes)

    with open(os.path.join(root, args.out), "w", encoding="utf-8") as fh:
        json.dump({"npz": args.npz, "results": results}, fh, indent=2)
    print("wrote %s" % args.out)
    print()
    print("references (same test split): SigLIP2 frozen prototype 38.61 / 1-NN 44.77;")
    print("SigLIP2 fine-tuned Region-only 90.64; CLIP zero-shot 35.68")


if __name__ == "__main__":
    main()
