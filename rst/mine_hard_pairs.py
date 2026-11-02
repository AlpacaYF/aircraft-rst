"""3.3.3 -- mine hard negative pairs from real visual similarity.

The teacher model already names types it finds confusable, but that is a
language prior. Here pairs come from cosine similarity between class prototypes
built on training-split regions, so "these two look alike" is a claim about the
pixels. The two sources are reported separately and combined.

  python -m rst.mine_hard_pairs --topk 8
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--embeddings", default="index/region_embeddings.npz")
    ap.add_argument("--text", default="knowledge/text_features.npz")
    ap.add_argument("--mllm-graph", default="knowledge/confusable_pairs.json")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--out", default="knowledge/hard_pairs.json")
    ap.add_argument("--table", default="index/hard_table.npy")
    ap.add_argument("--prototype-split", default="train")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    emb = np.load(os.path.join(root, args.embeddings), allow_pickle=True)
    txt = np.load(os.path.join(root, args.text), allow_pickle=True)
    classes = [str(c) for c in txt["classes"]]
    cidx = {c: i for i, c in enumerate(classes)}
    C = len(classes)

    mask = emb["splits"] == args.prototype_split
    feats = emb["features"][mask].astype(np.float32)
    rows = np.array([cidx[str(c)] for c in emb["classes"][mask]], dtype=np.int64)
    proto = np.zeros((C, feats.shape[1]), dtype=np.float32)
    counts = np.zeros(C, dtype=np.int64)
    for c in range(C):
        m = rows == c
        counts[c] = int(m.sum())
        if m.any():
            proto[c] = feats[m].mean(0)
    proto /= np.clip(np.linalg.norm(proto, axis=1, keepdims=True), 1e-8, None)

    sim = proto @ proto.T
    np.fill_diagonal(sim, -np.inf)
    covered = counts > 0
    if not covered.all():
        print("NOTE %d/%d classes have no %s regions, excluded from pairing: %s"
              % ((~covered).sum(), C, args.prototype_split,
                 ", ".join(np.asarray(classes)[~covered][:12])))
        sim[:, ~covered] = -np.inf
    k = min(args.topk, C - 1)
    top = np.argsort(-sim, axis=1)[:, :k]

    mllm = {}
    gp = os.path.join(root, args.mllm_graph)
    if os.path.exists(gp):
        with open(gp, encoding="utf-8") as fh:
            mllm = json.load(fh).get("confusable_pairs", {})

    pairs = []
    for r in range(C):
        for j in range(k):
            c = int(top[r, j])
            if c <= r:
                continue
            a, b = classes[r], classes[c]
            # a's entry lists the types it is confusable *with*, so it never
            # contains a itself -- test membership of the other name, not of the
            # pair as a set.
            named = (b in mllm.get(a, ())) or (a in mllm.get(b, ()))
            pairs.append({
                "a": a, "b": b,
                "visual_similarity": round(float(sim[r, c]), 4),
                "named_by_mllm": named,
                "named_both_ways": (b in mllm.get(a, ())) and (a in mllm.get(b, ())),
            })
    pairs.sort(key=lambda p: -p["visual_similarity"])

    both = [p for p in pairs if p["named_by_mllm"]]
    out = {
        "meta": {"prototype_split": args.prototype_split, "topk": k,
                 "min_class_instances": int(counts.min()),
                 "n_visual_pairs": len(pairs), "n_agreeing_with_mllm": len(both),
                 "agreement_rate": round(len(both) / max(1, len(pairs)), 3)},
        "hardest_pairs": pairs[:60],
        "per_class": {classes[r]: [{"class": classes[int(c)],
                                    "sim": round(float(sim[r, c]), 4)}
                                   for c in top[r] if covered[c]]
                      for r in range(C) if covered[r]},
    }
    with open(os.path.join(root, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    table = np.zeros((C, k), dtype=np.int32)
    for r in range(C):
        row = top[r].copy()
        # an uncovered class has no real neighbours; point the row at itself so
        # the loss's cand==y guard skips it rather than acting on a garbage pick
        row[~covered[row]] = r
        table[r] = row
    np.save(os.path.join(root, args.table), table)

    print("%d classes, prototypes from %d %s regions"
          % (C, int(mask.sum()), args.prototype_split))
    print("visual hard pairs: %d, of which %d (%.1f%%) also named by the MLLM, "
          "%d mutually named both ways"
          % (len(pairs), len(both), 100 * len(both) / max(1, len(pairs)),
             sum(1 for x in pairs if x["named_both_ways"])))
    print("\ntop 12 visually closest pairs:")
    for p in pairs[:12]:
        print("  %-12s %-12s sim=%.3f %s"
              % (p["a"], p["b"], p["visual_similarity"],
                 "mllm-agreed" if p["named_by_mllm"] else ""))
    print("\nwrote knowledge/hard_pairs.json and index/hard_table.npy")


if __name__ == "__main__":
    main()
