#!/usr/bin/env python
"""Where does the augmentation recipe's +3.0 come from? Decompose the gain of
v2_ro_s0 (with erase+jitter) over region_only (v1, plain) by entity-size bucket
and by class-frequency bucket. Both checkpoints' per-entity predictions exist.
"""
import json
import os
from collections import Counter

import numpy as np

os.chdir(os.path.expanduser("~/dmt/aircraft_rst"))

meta = json.load(open("index/class_meta.json", encoding="utf-8"))
classes = meta["classes"]

# training frequency per class
sup = Counter()
with open("index/instances.jsonl", encoding="utf-8") as fh:
    for line in fh:
        d = json.loads(line)
        if d.get("split") == "train":
            sup[d["class"]] += 1
sup_arr = np.array([sup.get(c, 0) for c in classes])

runs = {"plain (v1)": "region_only", "augmented (v2)": "v2_ro_s0"}
S = {}
for tag, r in runs.items():
    z = np.load("runs/%s/samples_test.npz" % r, allow_pickle=True)
    S[tag] = (z["y"].astype(int), z["pred"].astype(int))
y = S["plain (v1)"][0]
assert np.array_equal(y, S["augmented (v2)"][0]), "同一 test 集应逐行对齐"

# ---- by entity size (from each eval json) ----
ej = {tag: json.load(open("runs/%s/eval_test.json" % r))["by_entity_size"]
      for tag, r in runs.items()}
order = ["<64px", "64-128", "128-256", "256-384", "384-768", "768-1536", ">=1536px"]
print("=== gain by entity-size bucket (Top-1 / Macro-F1, pp) ===")
print("%-10s %6s %14s %14s %8s %8s" % ("bucket", "n", "plain", "augmented", "dT1", "dF1"))
tot_w = 0.0
for b in order:
    p, a = ej["plain (v1)"][b], ej["augmented (v2)"][b]
    dt = 100 * (a["top1"] - p["top1"])
    df = 100 * (a["macro_f1"] - p["macro_f1"])
    tot_w += dt * p["n"]
    print("%-10s %6d  %6.1f/%5.1f  %6.1f/%5.1f  %+7.1f %+7.1f"
          % (b, p["n"], 100 * p["top1"], 100 * p["macro_f1"],
             100 * a["top1"], 100 * a["macro_f1"], dt, df))
print("weighted average dTop1: %+.2f" % (tot_w / sum(ej["plain (v1)"][b]["n"] for b in order)))

# ---- by class frequency ----
print()
print("=== gain by train-frequency bucket ===")
buckets = [("tail (<50)", sup_arr[y] < 50), ("mid (50-300)", (sup_arr[y] >= 50) & (sup_arr[y] < 300)),
           ("head (>=300)", sup_arr[y] >= 300)]
print("%-14s %6s %10s %10s %8s" % ("bucket", "n", "plain", "augmented", "dT1"))
for name, m in buckets:
    pt = (S["plain (v1)"][1][m] == y[m]).mean()
    at = (S["augmented (v2)"][1][m] == y[m]).mean()
    print("%-14s %6d %9.1f%% %9.1f%%  %+6.1f" % (name, m.sum(), 100 * pt, 100 * at, 100 * (at - pt)))

# ---- which classes gain most (for the paper's case study) ----
print()
print("=== classes with the largest augmented-only recovery (>=3 flips) ===")
plain_ok = S["plain (v1)"][1] == y
aug_ok = S["augmented (v2)"][1] == y
gains = Counter()
for i in range(len(y)):
    if not plain_ok[i] and aug_ok[i]:
        gains[classes[y[i]]] += 1
for c, n in gains.most_common(10):
    print("  %-14s +%d (train n=%d)" % (c, n, sup.get(c, 0)))
losses = Counter()
for i in range(len(y)):
    if plain_ok[i] and not aug_ok[i]:
        losses[classes[y[i]]] += 1
print("net flips: +%d / -%d (total %d)" % (sum(gains.values()), sum(losses.values()), len(y)))

out = {
    "by_size": {b: {"n": ej["plain (v1)"][b]["n"],
                    "plain_top1": ej["plain (v1)"][b]["top1"],
                    "aug_top1": ej["augmented (v2)"][b]["top1"],
                    "d_top1_pp": round(100 * (ej["augmented (v2)"][b]["top1"]
                                              - ej["plain (v1)"][b]["top1"]), 2)}
                for b in order},
    "by_freq": {name: {"n": int(m.sum()),
                       "plain_top1": float((S["plain (v1)"][1][m] == y[m]).mean()),
                       "aug_top1": float((S["augmented (v2)"][1][m] == y[m]).mean())}
                for name, m in buckets},
    "flips": {"recovered": int(sum(gains.values())), "regressed": int(sum(losses.values()))},
}
json.dump(out, open("runs/augmentation_gain_report.json", "w"), indent=2)
print("\nwrote runs/augmentation_gain_report.json")
