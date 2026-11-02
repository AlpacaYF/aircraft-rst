#!/usr/bin/env python
"""Answer, from one re-scoring pass, whether the fusion has anything to gain from.

The ablation table says Region Only, Region+Scene and Region+Scene+Text land within
0.4 points of each other, which is inside the seed-to-seed spread. Overall accuracy
cannot say more than that. Per-entity predictions can, and they answer the only two
questions that decide whether to change the architecture or change the claim:

  complementarity   how many entities does the auxiliary branch get RIGHT where the
                    region branch is WRONG? If that is ~0, no gating, dropout or
                    attention structure can help -- there is no information to fuse,
                    and the honest move is to restate the paper's claim, not to train
                    more variants. If it is large but the overall number does not move,
                    the fusion is taking the information and failing to use it, and the
                    structural fixes are worth their GPU-hours.
  where it helps    the same counts broken out by entity size, box area ratio, objects
                    per image, region confidence and hard-pair membership, because the
                    design intent is conditional (3.4.3), not uniform.

Paired outcomes are compared with McNemar's exact test: the two configurations are
scored on the SAME 3,002 entities, so the paired test is the right one and a 1-point
difference on 3,002 samples is not automatically evidence of anything.

  .venv/bin/python -u -m tools.subset_report --root . \
      --pairs region_only:region_scene region_scene:align_full_kb region_only:align_full_kb
"""

import argparse
import json
import math
import os

import numpy as np

from rst.config import Config
from rst.data import EntityStore
from rst.eval import SIZE_BUCKETS, entity_long_sides, macro_f1


def wilson(k, n, z=1.96):
    """95% interval for a proportion; the normal approximation is misleading at n~3000
    only when p is extreme, which is exactly the regime of a small complementarity rate."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_p(b, c):
    """Exact two-sided binomial test on the discordant pairs (H0: b and c are equally
    likely). Chi-square without continuity correction overstates significance here."""
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


def subset_masks(store, ids, gt_path, hard_table, split):
    """Boolean masks keyed by subset name. Everything is derived from the index or from
    the reference run's own predictions, so no run has to be re-scored to redefine one."""
    recs = {"%s#%d" % (r["image"], r["instance_index"]): r for r in store.instances}
    got = [recs.get(i) for i in ids]
    missing = sum(1 for g in got if g is None)
    if missing:
        raise SystemExit("%d of %d dumped ids are not in the %s index -- the dump is "
                         "from a different index build, re-score instead of trusting it"
                         % (missing, len(ids), split))
    per_image = {}
    for r in store.instances:
        per_image[r["image"]] = per_image.get(r["image"], 0) + 1
    sizes = {}
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as fh:
            sizes = {k: v.get("size") for k, v in json.load(fh).items()}

    long_side = entity_long_sides(store, ids)
    n_obj = np.array([per_image[g["image"]] for g in got])
    area_ratio = np.zeros(len(got))
    for i, g in enumerate(got):
        x0, y0, x1, y1 = [float(v) for v in g["bbox"][:4]]
        wh = sizes.get(g["image"])
        ia = float(wh[0]) * float(wh[1]) if wh else max((x1 - x0) * (y1 - y0) * 40.0, 1.0)
        area_ratio[i] = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / ia

    m = {}
    for lo, hi, name in SIZE_BUCKETS:
        m["size:" + name] = (long_side >= lo) & (long_side < hi)
    for lo, hi, name in [(0, 0.005, "<0.5%"), (0.005, 0.02, "0.5-2%"),
                         (0.02, 0.10, "2-10%"), (0.10, 10, ">10%")]:
        m["area:" + name] = (area_ratio >= lo) & (area_ratio < hi)
    m["objects:single"] = n_obj <= 1
    m["objects:multi"] = n_obj >= 2
    if hard_table is not None:
        y = np.array([store.class_index[g["class"]] for g in got])
        hp = np.array([bool(set(int(v) for v in hard_table[c] if int(v) >= 0))
                       for c in y])
        m["hard:has-confusable"] = hp
        m["hard:none"] = ~hp
    return m, {"long_side": long_side.tolist(), "area_ratio": area_ratio.tolist(),
               "n_objects": n_obj.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="test")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="baseline:augmented run-name pairs to compare per entity")
    ap.add_argument("--out", default="runs/subset_report.json")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.chdir(root)
    cfg = Config()
    cfg.root = root

    store = EntityStore(os.path.join(root, cfg.index_dir), args.split)
    hard_path = os.path.join(root, cfg.index_dir, "hard_table.npy")
    hard_table = np.load(hard_path) if os.path.exists(hard_path) else None
    if hard_table is None:
        print("NOTE %s missing -- the hard-pair subsets are omitted" % hard_path)

    runs = {}
    for name in set(sum([p.split(":") for p in args.pairs], [])):
        path = "runs/%s/samples_%s.npz" % (name, args.split)
        if not os.path.exists(path):
            raise SystemExit("missing %s -- run tools/dump_samples.sh %s first" % (path, name))
        runs[name] = np.load(path, allow_pickle=True)
    ref = args.pairs[0].split(":")[0]
    ids = list(runs[ref]["ids"])
    y = runs[ref]["y"].astype(int)
    if any(not np.array_equal(runs[n]["ids"], runs[ref]["ids"]) for n in runs):
        raise SystemExit("the dumps are not aligned by entity id -- something reordered "
                         "the split; refusing to compare")

    ref_lp = runs[ref]["logits"]
    ref_pred_sort = np.sort(ref_lp, axis=1)
    margin_ref = ref_pred_sort[:, -1] - ref_pred_sort[:, -2]
    masks, extra = subset_masks(store, ids, os.path.join(root, "det_dataset",
                                                         "gt_classes.json"),
                                hard_table, args.split)
    # confidence terciles come from the region branch's own margin
    q = np.quantile(margin_ref, [1 / 3.0, 2 / 3.0])
    masks["conf:low"] = margin_ref <= q[0]
    masks["conf:mid"] = (margin_ref > q[0]) & (margin_ref <= q[1])
    masks["conf:high"] = margin_ref > q[1]

    out = {"split": args.split, "n": int(len(y)), "subsets": {}, "pairs": {},
           "gate": {}, "verdict": {}}
    for name, r in runs.items():
        pred = r["pred"].astype(int)
        out["subsets"][name] = {k: {"n": int(m.sum()),
                                    "top1": round(float((pred[m] == y[m]).mean()), 4)}
                               for k, m in masks.items() if m.any()}
        out["subsets"][name]["overall"] = {"n": len(y),
                                           "top1": round(float((pred == y).mean()), 4)}
        if bool(r["gate_present"]):
            g = r["gate"]
            out["gate"][name] = {
                "mean": [round(float(x), 4) for x in g.mean(0)],
                "std": [round(float(x), 4) for x in g.std(0)],
                "by_size": {k.split(":")[1]: [round(float(x), 4) for x in g[m].mean(0)]
                            for k, m in masks.items() if k.startswith("size:") and m.any()},
                "by_conf": {k.split(":")[1]: [round(float(x), 4) for x in g[m].mean(0)]
                            for k, m in masks.items() if k.startswith("conf:") and m.any()},
            }

    for spec in args.pairs:
        a, b = spec.split(":")
        pa, pb = runs[a]["pred"].astype(int), runs[b]["pred"].astype(int)
        ok_a, ok_b = pa == y, pb == y
        cc = int((ok_a & ok_b).sum())
        cw = int((ok_a & ~ok_b).sum())       # a right, b wrong  -> b lost these
        wc = int((~ok_a & ok_b).sum())       # a wrong, b right  -> b gained these
        ww = int((~ok_a & ~ok_b).sum())
        lo, hi = wilson(wc, len(y))
        entry = {"n": int(len(y)), "cc": cc, "wc": wc, "cw": cw, "ww": ww,
                 "net_top1_points": round(100.0 * (wc - cw) / len(y), 3),
                 "complementarity": round(100.0 * wc / len(y), 3),
                 "complementarity_ci95": [round(100 * lo, 3), round(100 * hi, 3)],
                 "regression_rate": round(100.0 * cw / len(y), 3),
                 "mcnemar_p": round(mcnemar_p(wc, cw), 6),
                 "by_subset": {}}
        for k, m in masks.items():
            if not m.any():
                continue
            s_w = int((~ok_a[m] & ok_b[m]).sum())
            s_c = int((ok_a[m] & ~ok_b[m]).sum())
            entry["by_subset"][k] = {"n": int(m.sum()), "wc": s_w, "cw": s_c,
                                     "delta_top1_points": round(100.0 * (s_w - s_c) / m.sum(), 2),
                                     "mcnemar_p": round(mcnemar_p(s_w, s_c), 5)}
        out["pairs"][spec] = entry

    first = out["pairs"][args.pairs[0]]
    out["verdict"] = {
        "pair": args.pairs[0],
        "complementarity_pct": first["complementarity"],
        "complementarity_ci95_pct": first["complementarity_ci95"],
        "mcnemar_p": first["mcnemar_p"],
        "reading": ("no information to fuse: restate the claim, do not add structure"
                    if first["complementarity"] < 1.0 else
                    "information exists; the fusion is what fails to exploit it"
                    if abs(first["net_top1_points"]) < 1.0 else
                    "the pair already differs beyond noise on the overall metric"),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    for spec, e in out["pairs"].items():
        print("%-34s gained %4d / lost %4d  net %+0.2f pts  complement %.2f%% "
              "[%.2f,%.2f]  McNemar p=%.4f"
              % (spec, e["wc"], e["cw"], e["net_top1_points"], e["complementarity"],
                 e["complementarity_ci95"][0], e["complementarity_ci95"][1], e["mcnemar_p"]))
        top = sorted(e["by_subset"].items(), key=lambda kv: -kv[1]["delta_top1_points"])[:3]
        bot = sorted(e["by_subset"].items(), key=lambda kv: kv[1]["delta_top1_points"])[:2]
        print("    best subsets :", ", ".join("%s %+0.1f(n=%d)" % (k, v["delta_top1_points"],
                                                                   v["n"]) for k, v in top))
        print("    worst subset :", ", ".join("%s %+0.1f(n=%d)" % (k, v["delta_top1_points"],
                                                                   v["n"]) for k, v in bot))
    print("\nverdict: %s" % json.dumps(out["verdict"], ensure_ascii=False))
    print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
