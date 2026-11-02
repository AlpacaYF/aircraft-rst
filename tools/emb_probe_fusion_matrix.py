#!/usr/bin/env python
"""Fusion-mode x modality ablation on the frozen Qwen3-VL-Embedding features.

Two fusion families, same four modality combinations {R, RS, RT, RST}:

EARLY  (feature concat + trained head)
    R     : region 4096
    RS    : region + scene 8192
    RT    : region + region->class-text sim 114
    RST   : region + scene + two sims 228

LATE   (one expert per modality, decision-level average)
    E_R   : MLP on region features
    E_S   : MLP on scene features
    E_T   : class-text retrieval scores (no training; sim of region+scene to the
            114 class vectors), reported alone and averaged in
    combos: R / R+S / R+T / R+S+T  (logits averaged, no fitting)

For reference the MID-fusion numbers from the RST system are printed alongside.
Text has no per-sample vector by construction (a sample's text IS its class), so
in both families the text modality enters through the 114-d class-prototype
match, which is the only form available at test time.
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


def report(tag, logits, y, n_classes, out_dict):
    pred = logits.argmax(1)
    r = dict(top1=float((pred == y).mean()), top5=topk(logits, y, 5),
             f1=macro_f1(y, pred, n_classes)[0])
    out_dict[tag] = r
    print("%-42s Top-1 %6.2f  Top-5 %6.2f  MacroF1 %6.2f"
          % (tag, 100 * r["top1"], 100 * r["top5"], 100 * r["f1"]))
    return logits


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
    for ep in range(epochs):
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
    ap.add_argument("--region", default="index/emb_qwen3vle_region_all.npz")
    ap.add_argument("--scene", default="index/emb_qwen3vle_scene_all.npz")
    ap.add_argument("--text", default="index/emb_qwen3vle_text_classes.npz")
    ap.add_argument("--out", default="runs/emb_fusion_matrix_report.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    meta = json.load(open(os.path.join(root, "index", "class_meta.json"),
                          encoding="utf-8"))
    cidx, n_classes = meta["class_index"], len(meta["classes"])

    R = np.load(os.path.join(root, args.region), allow_pickle=True)
    Xr, ylr, spr, idsr = R["features"], R["labels"], R["splits"], R["ids"]
    yr = np.array([cidx[str(c)] for c in ylr])
    img_of = np.array([i.split("#")[0] for i in idsr])
    S = np.load(os.path.join(root, args.scene), allow_pickle=True)
    scene_of = {i.split("#")[0]: v for i, v in zip(S["ids"], S["features"])}
    Xt = np.load(os.path.join(root, args.text), allow_pickle=True)["features"]

    keep = np.array([im in scene_of for im in img_of])
    Xr, yr, img_of, spr = Xr[keep], yr[keep], img_of[keep], spr[keep]
    Xs = np.stack([scene_of[im] for im in img_of])
    print("aligned: %d samples (train %d / test %d) | text %d classes"
          % (len(yr), int((spr == 'train').sum()), int((spr == 'test').sum()),
             len(Xt)))

    tr, te = spr == "train", spr == "test"
    sim_r, sim_s = Xr @ Xt.T, Xs @ Xt.T

    counts = np.bincount(yr[tr], minlength=n_classes).astype(np.float64)
    w = np.where(counts > 0, counts.sum() / np.clip(counts, 1, None), 0.0)
    w = w / w[w > 0].mean()

    out = {"early": {}, "late": {}}
    print("--- EARLY fusion (concat + trained head) ---")
    early = {
        "R": Xr,
        "RS": np.concatenate([Xr, Xs], 1),
        "RT": np.concatenate([Xr, sim_r], 1),
        "RST": np.concatenate([Xr, Xs, sim_r, sim_s], 1),
    }
    for name, X in early.items():
        lg = mlp_head(X[tr], yr[tr], X[te], n_classes, hidden=1024, class_weight=w)
        report("early %-4s MLP1024" % name, lg, yr[te], n_classes, out["early"])
        lg = mlp_head(X[tr], yr[tr], X[te], n_classes, hidden=0, class_weight=w)
        report("early %-4s linear " % name, lg, yr[te], n_classes, out["early"])

    print("--- LATE fusion (per-modality experts, logit average) ---")
    # experts
    e_r = mlp_head(Xr[tr], yr[tr], Xr[te], n_classes, class_weight=w)
    e_s = mlp_head(Xs[tr], yr[tr], Xs[te], n_classes, class_weight=w)
    e_t_ret = (sim_r[te] + sim_s[te])            # untrained retrieval score
    e_t_mlp = mlp_head(np.concatenate([sim_r, sim_s], 1)[tr], yr[tr],
                       np.concatenate([sim_r, sim_s], 1)[te], n_classes,
                       hidden=512, class_weight=w)
    report("late  E_R alone", e_r, yr[te], n_classes, out["late"])
    report("late  E_S alone", e_s, yr[te], n_classes, out["late"])
    report("late  E_T alone (retrieval)", e_t_ret, yr[te], n_classes, out["late"])
    report("late  E_T alone (trained)", e_t_mlp, yr[te], n_classes, out["late"])
    for tag, lg in [("late  R+S (avg)", (e_r + e_s) / 2),
                    ("late  R+T (avg, retrieval T)", (e_r + e_t_ret) / 2),
                    ("late  R+T (avg, trained T)", (e_r + e_t_mlp) / 2),
                    ("late  R+S+T (avg, retrieval T)", (e_r + e_s + e_t_ret) / 3),
                    ("late  R+S+T (avg, trained T)", (e_r + e_s + e_t_mlp) / 3)]:
        report(tag, lg, yr[te], n_classes, out["late"])

    with open(os.path.join(root, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % args.out)
    print()
    print("MID-fusion reference (RST system, same test split):")
    print("  R 90.64/88.93 | RS 91.67/89.74 | RT 91.51/89.30 | RST 91.14/88.70")


if __name__ == "__main__":
    main()
