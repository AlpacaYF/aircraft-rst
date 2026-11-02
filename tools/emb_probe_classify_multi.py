#!/usr/bin/env python
"""Multi-modal frozen-feature classifiers on Qwen3-VL-Embedding features.

Feature ladders (all trained on the FULL train split, scored on the full test split):
  A. region                     4096
  B. region + scene             8192
  C. region + scene + text      + 114-d image->class-text similarity (from both
                                image vectors) = 8420   (text enters the only way it
                                can at test time: as a class-prototype match score)

Classifiers: nearest class-mean prototype / 1-NN (A only), linear, MLP.
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


def topk_acc(sim, y, k):
    pred = np.argsort(-sim, axis=1)[:, :k]
    return float((pred == y[:, None]).any(1).mean())


def report(tag, logits, y, n_classes):
    pred = logits.argmax(1)
    f1 = macro_f1(y, pred, n_classes)[0]
    r = dict(top1=float((pred == y).mean()), top5=topk_acc(logits, y, 5), f1=f1)
    print("%-40s Top-1 %6.2f  Top-5 %6.2f  MacroF1 %6.2f"
          % (tag, 100 * r["top1"], 100 * r["top5"], 100 * r["f1"]))
    return r


def train_head(Xtr, ytr, Xte, n_classes, hidden=1024, epochs=60, lr=1e-3,
               wd=1e-4, batch=256, class_weight=None, seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = Xtr.shape[1]
    net = (nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(0.2),
                         nn.Linear(hidden, n_classes))
           if hidden else nn.Linear(d, n_classes)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    cw = (torch.tensor(class_weight, dtype=torch.float32, device=dev)
          if class_weight is not None else None)
    crit = nn.CrossEntropyLoss(weight=cw)
    Xtr_t = torch.tensor(Xtr, device=dev)
    ytr_t = torch.tensor(ytr, device=dev)
    n = len(ytr_t)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            crit(net(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, device=dev)).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--region", default="index/emb_qwen3vle_region_all.npz")
    ap.add_argument("--scene", default="index/emb_qwen3vle_scene_all.npz")
    ap.add_argument("--text", default="index/emb_qwen3vle_text_classes.npz")
    ap.add_argument("--out", default="runs/emb_probe_multimodal_report.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    cidx = meta["class_index"]
    n_classes = len(meta["classes"])

    R = np.load(os.path.join(root, args.region), allow_pickle=True)
    Xr, ylr, spr, idsr = R["features"], R["labels"], R["splits"], R["ids"]
    yr = np.array([cidx[str(c)] for c in ylr])
    img_of = np.array([i.split("#")[0] for i in idsr])

    S = np.load(os.path.join(root, args.scene), allow_pickle=True)
    Xs, ids_s = S["features"], S["ids"]
    scene_of = {i.split("#")[0]: v for i, v in zip(ids_s, Xs)}

    T = np.load(os.path.join(root, args.text), allow_pickle=True)
    Xt = T["features"]                      # [114, 4096], row = class index order

    miss = sum(1 for im in img_of if im not in scene_of)
    print("region %d | scene %d | text %d | missing scene for %d regions"
          % (len(Xr), len(Xs), len(Xt), miss))
    keep = np.array([im in scene_of for im in img_of])
    Xr, yr, img_of = Xr[keep], yr[keep], img_of[keep]
    spr = spr[keep]
    Xs_al = np.stack([scene_of[im] for im in img_of])

    tr, te = spr == "train", spr == "test"
    sim_r = Xr @ Xt.T                        # image->class-text match scores
    sim_s = Xs_al @ Xt.T

    feats = {
        "A region": Xr,
        "B region+scene": np.concatenate([Xr, Xs_al], 1),
        "C +textmatch": np.concatenate([Xr, Xs_al, sim_r, sim_s], 1),
    }

    counts = np.bincount(yr[tr], minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.clip(counts, 1, None), 0.0)
    w = w / w[w > 0].mean()

    results = {}
    # A: prototype + 1-NN as well
    Xa_tr, Xa_te = feats["A region"][tr], feats["A region"][te]
    proto = np.zeros((n_classes, Xa_tr.shape[1]), dtype=np.float32)
    for c in range(n_classes):
        m = yr[tr] == c
        if m.any():
            v = Xa_tr[m].mean(0)
            proto[c] = v / max(1e-8, np.linalg.norm(v))
    results["A_prototype"] = report("A region: class-mean prototype",
                                    Xa_te @ proto.T, yr[te], n_classes)
    sim1 = Xa_te @ Xa_tr.T
    nn_pred = yr[tr][sim1.argmax(1)]
    results["A_1nn"] = report("A region: 1-NN",
                              np.eye(n_classes)[nn_pred], yr[te], n_classes)

    for name, X in feats.items():
        Xtr, Xte = X[tr], X[te]
        results[name + " linear"] = report(
            name + ": linear",
            train_head(Xtr, yr[tr], Xte, n_classes, hidden=0, class_weight=w),
            yr[te], n_classes)
        results[name + " MLP"] = report(
            name + ": MLP 1024",
            train_head(Xtr, yr[tr], Xte, n_classes, hidden=1024, class_weight=w),
            yr[te], n_classes)

    with open(os.path.join(root, args.out), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("wrote %s" % args.out)
    print()
    print("references (same test split): SigLIP2 frozen 1-NN 44.77 | Qwen3VLE probe"
          " (20/class) MLP 52.53 | SigLIP2 fine-tuned 90.64")


if __name__ == "__main__":
    main()
