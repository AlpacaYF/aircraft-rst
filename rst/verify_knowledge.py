"""3.3.2 -- visual consistency verification of the knowledge base.

Run this AFTER the 3.5.2 contrastive alignment, not before. On these 114
fine-grained types a frozen SigLIP 2 gives no usable text->image signal
(own-class rank median 56 against a chance median of 57.5, top-1 0.7%), so
verifying against frozen features measures noise and would discard most of the
knowledge base. `rst.extract_embeddings --ckpt runs/<name>/best.pt` exports both
sides of the similarity from the trained model; this script then asks, per
sentence, whether it favours its own class over the specific types it could be
confused with.

The verdict is only actionable if it beats chance by a real margin -- see
`chance_keep_rate` in the output and the gate in run_pipeline.sh.

  python -m rst.verify_knowledge --embeddings index/region_embeddings_trained.npz
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--embeddings", default="index/region_embeddings.npz")
    ap.add_argument("--text", default="knowledge/text_features.npz")
    ap.add_argument("--out", default="knowledge/kb_verification.json")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="require the own-vs-hard-negative similarity gap to exceed this")
    ap.add_argument("--hard-table", default="index/hard_table.npy",
                    help="from rst.mine_hard_pairs; enables the contrast criterion")
    ap.add_argument("--n-negatives", type=int, default=3)
    ap.add_argument("--prototype-split", default="train")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    emb = np.load(os.path.join(root, args.embeddings), allow_pickle=True)
    txt = np.load(os.path.join(root, args.text), allow_pickle=True)
    classes = [str(c) for c in txt["classes"]]
    cidx = {c: i for i, c in enumerate(classes)}

    keep_mask = emb["splits"] == args.prototype_split
    feats = emb["features"][keep_mask].astype(np.float32)
    # map by class *name*, not by the index stored in the other file
    rows = np.array([cidx[str(c)] for c in emb["classes"][keep_mask]], dtype=np.int64)
    proto = np.zeros((len(classes), feats.shape[1]), dtype=np.float32)
    counts = np.zeros(len(classes))
    for c in range(len(classes)):
        m = rows == c
        counts[c] = m.sum()
        if m.any():
            proto[c] = feats[m].mean(0)
    proto /= np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8, None)

    # A class with no prototype regions has a zero vector, and a zero vector ties
    # every similarity at 0.0 -- leaving it in the candidate set makes argmax
    # arbitrary and collapses the keep rate under partial coverage.
    covered = counts > 0
    if not covered.all():
        print("NOTE %d/%d classes have no %s regions and are excluded as candidates: %s"
              % ((~covered).sum(), len(classes), args.prototype_split,
                 ", ".join(classes[i] for i in np.where(~covered)[0][:12])))

    strings = [str(s) for s in txt["strings"]]
    kinds = [str(k) for k in txt["kinds"]]
    # When the embeddings were exported with --ckpt they carry every knowledge
    # sentence pushed through the same trained text projection as the visual side,
    # so the two live in one aligned space. The frozen text tower is kept only for
    # the pre-training diagnostic, whose verdict is known to be at chance.
    if "text_projected" in emb.files:
        text_feats = emb["text_projected"].astype(np.float32)
        trows = emb["text_rows"].astype(np.int64)
        space = str(emb["space"][0]) if "space" in emb.files else "trained"
    else:
        text_feats = txt["features"].astype(np.float32)
        trows = txt["rows"].astype(np.int64)
        space = "frozen_siglip2 (diagnostic only -- see the module docstring)"
    if text_feats.shape[0] != len(strings):
        raise SystemExit("text side has %d rows but the knowledge base has %d sentences"
                         % (text_feats.shape[0], len(strings)))
    if text_feats.shape[1] != feats.shape[1]:
        raise SystemExit("text dim %d != visual dim %d -- the two sides are not in the "
                         "same space; re-run extract_embeddings"
                         % (text_feats.shape[1], feats.shape[1]))
    text_feats /= np.clip(np.linalg.norm(text_feats, axis=1, keepdims=True), 1e-8, None)
    print("similarity space: %s" % space)

    sims = text_feats @ proto.T                       # [n_strings, C]
    sims = np.where(covered[None, :], sims, -np.inf)  # drop empty prototypes
    own = sims[np.arange(len(sims)), trows]

    """Two candidate criteria.

    all-classes (argmax rank) turned out to be unusable here: on 114 fine-grained
    aircraft types a *frozen* SigLIP 2 retrieves the owning class at chance level
    (median rank ~57 of 107), so any rank threshold keeps a random ~9%.

    hard-negative contrast is what actually separates a good description from a
    bad one: does this sentence favour its own class over the specific type it
    could be confused with. Needs index/hard_table.npy from rst.mine_hard_pairs.
    """
    table_path = os.path.join(root, args.hard_table)
    mode = "all_classes_rank1"
    if os.path.exists(table_path):
        table = np.load(table_path)
        neg = np.array([[j for j in table[r] if j != r and covered[j]][:args.n_negatives]
                        for r in range(len(classes))], dtype=object)
        contrast = np.zeros(len(sims), dtype=np.float32)
        for i, r in enumerate(trows):
            cand = neg[r]
            if len(cand) == 0:
                contrast[i] = np.nan
                continue
            contrast[i] = own[i] - sims[i, np.asarray(cand, dtype=int)].max()
        margin = contrast
        correct = np.isfinite(margin) & (margin > 0)
        mode = "hard_negative_contrast"
    else:
        masked = sims.copy()
        masked[np.arange(len(sims)), trows] = -np.inf
        margin = own - masked.max(1)
        correct = sims.argmax(1) == trows
    print("criterion: %s" % mode)
    rank = (sims > own[:, None]).sum(1) + 1

    unverifiable = ~covered[trows]

    drop, keep, unknown = [], [], []
    per_kind = defaultdict(lambda: [0, 0])
    per_class = defaultdict(lambda: [0, 0])
    for i, (s, k, r) in enumerate(zip(strings, kinds, trows)):
        if unverifiable[i]:
            unknown.append({"text": s, "kind": k, "class": classes[r]})
            continue
        ok = bool(correct[i]) and margin[i] >= args.min_margin
        per_kind[k][0] += 1
        per_kind[k][1] += int(ok)
        per_class[classes[r]][0] += 1
        per_class[classes[r]][1] += int(ok)
        (keep if ok else drop).append(
            {"text": s, "kind": k, "class": classes[r], "margin": round(float(margin[i]), 4),
             "own_class_rank": int(rank[i])})

    out = {
        "meta": {"prototype_split": args.prototype_split,
                 "space": space,
                 "embeddings": args.embeddings,
                 "classes_with_prototype": int(covered.sum()),
                 "classes_without_prototype": int((~covered).sum()),
                 "min_margin": args.min_margin,
                 "n_strings": len(strings), "n_keep": len(keep), "n_drop": len(drop),
                 "n_unverifiable": len(unknown),
                 "criterion": mode, "n_negatives": args.n_negatives,
                 # with no real signal, "own beats the best of N negatives"
                 # happens 1/(N+1) of the time; compare the keep rate to this
                 "chance_keep_rate": round(1.0 / (args.n_negatives + 1), 3)
                 if mode == "hard_negative_contrast" else
                 round(1.0 / max(1, int(covered.sum())), 3),
                 "own_class_rank_median": int(np.median(rank)),
                 "own_class_rank_top1_pct": round(100 * float((rank == 1).mean()), 2),
                 "chance_median_rank": round((int(covered.sum()) + 1) / 2, 1),
                 "overall_keep_rate": round(len(keep) / max(1, len(keep) + len(drop)), 3)},
        "keep_rate_by_kind": {k: {"n": v[0], "kept": v[1],
                                  "rate": round(v[1] / max(1, v[0]), 3)}
                              for k, v in sorted(per_kind.items())},
        "keep_rate_by_class": {c: {"n": v[0], "kept": v[1],
                                   "rate": round(v[1] / max(1, v[0]), 3)}
                               for c, v in sorted(per_class.items())},
        "drop": [d["text"] for d in sorted(drop, key=lambda d: d["margin"])],
        "drop_detail": sorted(drop, key=lambda d: d["margin"])[:200],
        "keep": [d["text"] for d in keep],
        "unverifiable": unknown,
    }
    path = os.path.join(root, args.out)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print("prototypes from %s split, %d regions, %d/%d classes covered"
          % (args.prototype_split, int(keep_mask.sum()), int(covered.sum()), len(classes)))
    judged = len(keep) + len(drop)
    print("strings %d -> judged %d, keep %d (%.1f%%), drop %d, unverifiable %d"
          % (len(strings), judged, len(keep), 100 * len(keep) / max(1, judged),
             len(drop), len(unknown)))
    for k, v in sorted(per_kind.items()):
        print("  %-5s %4d/%4d kept (%.1f%%)" % (k, v[1], v[0], 100 * v[1] / max(1, v[0])))
    worst = sorted(per_class.items(), key=lambda kv: kv[1][1] / max(1, kv[1][0]))[:8]
    print("chance baseline for this criterion: %.1f%% -- a keep rate near it means the "
          "frozen encoder carries little usable signal and the filter is mostly noise"
          % (100 * out["meta"]["chance_keep_rate"]))
    print("worst classes:", ", ".join("%s %d/%d" % (c, v[1], v[0]) for c, v in worst))
    print("own-class rank: median %d (chance %.1f), top-1 %.1f%% -- if the median sits "
          "near chance, a frozen encoder gives no usable retrieval signal and the "
          "hard-negative contrast criterion is the only workable one"
          % (int(np.median(rank)), (int(covered.sum()) + 1) / 2, 100 * float((rank == 1).mean())))
    print("wrote", path)


if __name__ == "__main__":
    main()
