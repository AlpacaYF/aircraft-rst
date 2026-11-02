#!/usr/bin/env python
"""Do the knowledge-base prototypes actually separate the 114 classes?

Cheap gate before spending ~8 GPU-hours retraining with a different text tower. It
pools each class's knowledge sentences into one prototype and uses it as a zero-shot
classifier over real region crops, so the number is directly comparable to the
class-name zero-shot rows of 表4-2.

Two things have to line up for the result to mean anything, and both are printed:
  * the tower that encoded the text must also encode the images (a CLIP prototype
    compared against a SigLIP 2 image feature lives in a different space and measures
    nothing), so --text-features and --encoder have to agree;
  * the prototypes' pairwise cosine says how much geometry a learned text_proj has to
    work with. Near 1.0 means every class looks the same and no projection can fix it.

  python -m tools.probe_text_prototypes --text-features knowledge/text_features_clipL.npz \
      --encoder clip --n 600 --gpu 3
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rst.baselines import RegionCollator
from rst.config import Config
from rst.data import EntityDataset
from rst.eval import macro_f1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--text-features", default="knowledge/text_features_clipL.npz")
    ap.add_argument("--encoder", default="clip", choices=["clip", "siglip2"])
    ap.add_argument("--model", default="", help="defaults to the tower's usual checkpoint")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kinds", default="",
                    help="comma-separated subset of sentence kinds to score separately "
                         "(desc/attr/feat/diff). Attributes describe features shared by "
                         "many types, so they are weak retrieval cues by construction; "
                         "the `diff` sentences explicitly contrast similar types and are "
                         "what 3.3.3 claims carries the discriminative signal. Scoring "
                         "per kind is what tests that claim.")
    ap.add_argument("--whiten", action="store_true",
                    help="subtract the mean sentence vector and renormalise before "
                         "scoring. Contrastive text towers are strongly anisotropic -- "
                         "measured mean pairwise cosine 0.92-0.97 for these prototypes -- "
                         "so every class shares one dominant direction that carries no "
                         "class information and swamps the differences. Removing it is "
                         "what turns 12 distinct predicted classes into 98.")
    args = ap.parse_args()

    cfg = Config()
    cfg.root = os.path.abspath(args.root)
    os.chdir(cfg.root)
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    z = np.load(args.text_features, allow_pickle=True)
    tokens, mask = z["tokens"], z["mask"]
    classes = [str(c) for c in z["classes"]]
    m = torch.from_numpy(mask).unsqueeze(-1)
    proto = (torch.from_numpy(tokens) * m).sum(1) / m.sum(1).clamp(min=1)
    proto = F.normalize(proto.float(), dim=-1)
    G = (proto @ proto.T).fill_diagonal_(0)
    n_sent = float(m.squeeze(-1).sum(1).float().mean())
    print("%s: %d classes, mean %.1f sentences each, dim %d"
          % (args.text_features, len(classes), n_sent, proto.shape[1]))
    print("prototype pairwise cosine: mean %.4f  p99 %.4f  max %.4f"
          % (float(G.mean()), float(G.quantile(0.99)), float(G.max())))

    if args.encoder == "clip":
        from rst.baselines import CLIPEncoder
        enc = CLIPEncoder(args.model or "openai/clip-vit-large-patch14")
    else:
        from rst.encoders import SigLIP2Encoder
        enc = SigLIP2Encoder(args.model or cfg.vision_model, freeze=True, load_text=False)
    enc = enc.to(device).eval()

    meta = json.load(open(os.path.join(cfg.index_dir, "class_meta.json"), encoding="utf-8"))
    if [str(c) for c in meta["classes"]] != classes:
        raise SystemExit("class order in %s differs from class_meta.json -- cannot score"
                         % args.text_features)

    ds = EntityDataset(cfg, "test", train=False)
    if args.n and args.n < len(ds):
        ds.records = ds.records[:args.n]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                        collate_fn=RegionCollator(ds), pin_memory=(device.type == "cuda"))
    # image features do not depend on which sentences are used, so encode once
    imgs, ys = [], []
    for b in loader:
        px = b["region"][:, :3].to(device)
        if px.shape[-1] != enc.native_image_size:
            px = F.interpolate(px, size=enc.native_image_size, mode="bilinear",
                               align_corners=False)
        # contrastive_feature is the embedding that shares space with text; the
        # mean-pooled image_feature does not, and would measure nothing here
        imgs.append(F.normalize(enc.contrastive_feature(px).float(), dim=-1).cpu())
        ys.append(b["class"])
    V = torch.cat(imgs)
    y = torch.cat(ys).numpy()
    mil = np.isin(y, list(ds.store.military_index))

    sent = F.normalize(torch.from_numpy(z["features"]).float(), dim=-1)
    kinds = np.asarray([str(k) for k in z["kinds"]])
    rows_np = z["rows"].astype(np.int64)

    def report(sel, label):
        """Score one selection of sentences, both ways of aggregating them."""
        if sel.sum() == 0:
            print("%-22s (no sentences)" % label)
            return
        s = sent[sel].to(device)
        if args.whiten:
            s = F.normalize(s - s.mean(0, keepdim=True), dim=-1)
        r = torch.from_numpy(rows_np[sel]).to(device)
        v = V.to(device)
        # (a) mean over each class's selected sentences -> one prototype per class
        acc = torch.zeros(len(classes), s.size(1), device=device)
        cnt = torch.zeros(len(classes), 1, device=device)
        acc.index_add_(0, r, s)
        cnt.index_add_(0, r, torch.ones(s.size(0), 1, device=device))
        pm = F.normalize(acc / cnt.clamp(min=1), dim=-1)
        G = (pm @ pm.T).fill_diagonal_(0)
        pred_mean = (v @ pm.T).argmax(1).cpu().numpy()
        # (b) max over the class's individual sentences -- what eq. 3-12 does
        sims = v @ s.T
        best = sims.new_full((sims.size(0), len(classes)), -1e4)
        best.scatter_reduce_(1, r.expand(sims.size(0), -1), sims, reduce="amax")
        pred_max = best.argmax(1).cpu().numpy()
        for lab, pred in (("class-mean", pred_mean), ("max-sentence", pred_max)):
            print("%-22s %-13s Top-1 %6.2f%%  MacroF1 %6.2f%%  mil %6.2f%%  "
                  "distinct %3d  proto cos %.4f  (n=%d sentences)"
                  % (label, lab, 100 * float((pred == y).mean()),
                     100 * macro_f1(y, pred, len(classes))[0],
                     100 * float((pred[mil] == y[mil]).mean()),
                     len(np.unique(pred)), float(G.mean()), int(sel.sum())))

    report(np.ones(len(sent), dtype=bool), "ALL")
    if args.kinds:
        for k in [x.strip() for x in args.kinds.split(",") if x.strip()]:
            report(kinds == k, k)
    print("scored %d crops (chance %.2f%%)" % (len(y), 100 / len(classes)))


if __name__ == "__main__":
    main()
