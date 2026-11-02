#!/usr/bin/env python
"""Fusion-mode x modality matrix on FINE-TUNED SigLIP2 features.

Same protocol as the earlier frozen-embedding matrix, but all visual features
come from the trained v2_rs_s0 checkpoint (fine-tuned SigLIP2 visual tower):
  region : mean-pooled encoder tokens of the expanded crop  [768]
  scene  : masked mean of pooled scene tokens               [768]
  text   : class prototypes (paper's text_features)         [114,768]

EARLY: concat + trained head (R / RS / RT / RST)
LATE : per-modality experts + logit average (R / R+S / R+T / R+S+T)
Reference mid-fusion numbers are printed at the end.
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


def topk(logits, y, k):
    pred = np.argsort(-logits, axis=1)[:, :k]
    return float((pred == y[:, None]).any(1).mean())


def report(tag, logits, y, n_classes, out):
    pred = logits.argmax(1)
    r = dict(top1=float((pred == y).mean()), top5=topk(logits, y, 5),
             f1=macro_f1(y, pred, n_classes)[0])
    out[tag] = r
    print("%-40s Top-1 %6.2f  Top-5 %6.2f  MacroF1 %6.2f"
          % (tag, 100 * r["top1"], 100 * r["top5"], 100 * r["f1"]))


def mlp_head(Xtr, ytr, Xte, n_classes, hidden=1024, epochs=60, class_weight=None,
             seed=0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layers = ([nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(0.2),
               nn.Linear(hidden, n_classes)] if hidden
              else [nn.Linear(Xtr.shape[1], n_classes)])
    net = nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    cw = (torch.tensor(class_weight, dtype=torch.float32, device=dev)
          if class_weight is not None else None)
    crit = nn.CrossEntropyLoss(weight=cw)
    Xt = torch.tensor(Xtr, device=dev)
    yt = torch.tensor(ytr, device=dev)
    for _ in range(epochs):
        perm = torch.randperm(len(yt), device=dev)
        for i in range(0, len(yt), 256):
            idx = perm[i:i + 256]
            opt.zero_grad()
            crit(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(Xte, device=dev)).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--prefix", default="index/siglipft_rs")
    ap.add_argument("--out", default="runs/fusion_matrix_siglip_report.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    cidx, n_classes = meta["class_index"], len(meta["classes"])

    R = np.load(os.path.join(root, args.prefix + "_region.npz"), allow_pickle=True)
    Xr = R["features"]
    lab = R["labels"]
    yr = (np.array([cidx[str(c)] for c in lab]) if lab.dtype == object
          else lab.astype(np.int64))
    spr, ids = R["splits"], R["ids"]
    img_of = np.array([i.split("#")[0] for i in ids])
    S = np.load(os.path.join(root, args.prefix + "_scene.npz"), allow_pickle=True)
    scene_of = {im: v for im, v in zip(S["images"], S["features"])}
    T = np.load(os.path.join(root, args.prefix + "_text.npz"), allow_pickle=True)
    Xt = T["features"]
    order = [cidx[str(c)] for c in T["classes"]]
    Xt = Xt[np.argsort(order)]                 # rows back into class-index order

    keep = np.array([im in scene_of for im in img_of])
    Xr, yr, img_of, spr = Xr[keep], yr[keep], img_of[keep], spr[keep]
    Xs = np.stack([scene_of[im] for im in img_of])
    # L2-normalise everything so cosine-style sims and linear heads see the same scale
    def l2(a):
        return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    Xr, Xs, Xt = l2(Xr), l2(Xs), l2(Xt)
    print("aligned %d samples (train %d / test %d)"
          % (len(yr), int((spr == 'train').sum()), int((spr == 'test').sum())))

    tr, te = spr == "train", spr == "test"
    sim_r, sim_s = Xr @ Xt.T, Xs @ Xt.T
    counts = np.bincount(yr[tr], minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.clip(counts, 1, None), 0.0)
    w = w / w[w > 0].mean()

    out = {"early": {}, "late": {}}
    print("--- EARLY (concat + trained head), fine-tuned SigLIP2 features ---")
    early = {"R": Xr, "RS": np.concatenate([Xr, Xs], 1),
             "RT": np.concatenate([Xr, sim_r], 1),
             "RST": np.concatenate([Xr, Xs, sim_r, sim_s], 1)}
    for name, X in early.items():
        report("early %-4s MLP1024" % name,
               mlp_head(X[tr], yr[tr], X[te], n_classes, class_weight=w),
               yr[te], n_classes, out["early"])
        report("early %-4s linear" % name,
               mlp_head(X[tr], yr[tr], X[te], n_classes, hidden=0, class_weight=w),
               yr[te], n_classes, out["early"])

    print("--- LATE (per-modality experts, logit average) ---")
    e_r = mlp_head(Xr[tr], yr[tr], Xr[te], n_classes, class_weight=w)
    e_s = mlp_head(Xs[tr], yr[tr], Xs[te], n_classes, class_weight=w)
    sim_tr = np.concatenate([sim_r, sim_s], 1)
    e_t = mlp_head(sim_tr[tr], yr[tr], sim_tr[te], n_classes, hidden=512,
                   class_weight=w)
    e_t_ret = (sim_r[te] + sim_s[te]) * 10.0   # scale to be comparable with logits
    report("late  E_R alone", e_r, yr[te], n_classes, out["late"])
    report("late  E_S alone", e_s, yr[te], n_classes, out["late"])
    report("late  E_T alone (trained)", e_t, yr[te], n_classes, out["late"])
    for tag, lg in [("late  R+S (avg)", (e_r + e_s) / 2),
                    ("late  R+T (avg)", (e_r + e_t) / 2),
                    ("late  R+S+T (avg)", (e_r + e_s + e_t) / 3)]:
        report(tag, lg, yr[te], n_classes, out["late"])

    with open(os.path.join(root, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % args.out)
    print()
    print("MID-fusion reference (RST system, same test split):")
    print("  R 90.64/88.93 | RS 91.67/89.74 | RT 91.51/89.30 | RST 91.14/88.70")


if __name__ == "__main__":
    main()
