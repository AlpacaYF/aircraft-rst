"""Audit the generated semantic knowledge base before it is used as supervision.

The point is to find what should be dropped or down-weighted by the
visual-consistency step of section 3.3.2: fields the teacher could not actually
see, and feature sentences that repeat across many classes and therefore carry
no discriminative signal.
"""

import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(ROOT, "knowledge", "aircraft_semantic_kb.json")
ATTR_KEYS = ["机翼布局", "发动机数量与布局", "尾翼形式", "进气道", "机鼻特征", "座舱与起落架"]
UNSEEN = re.compile(r"不可见|看不清|无法判断|无法确定|不明确|未见|无法观察")


def main():
    kb = json.load(open(KB, encoding="utf-8"))
    entries = [e for e in kb["entries"] if e.get("ok")]
    print("classes ok: %d / %d in vocab" % (len(entries), kb.get("n_classes_ok")))
    print("teacher: %s" % kb["meta"]["teacher_model"])

    lat = [e["latency_s"] for e in entries]
    ct = [e["completion_tokens"] for e in entries]
    print("latency  median %.1fs  max %.1fs" % (sorted(lat)[len(lat) // 2], max(lat)))
    print("tokens   median %d  total %d" % (sorted(ct)[len(ct) // 2], sum(ct)))

    print("\n--- attribute completeness (section 3.3.1 field Ac) ---")
    blind = defaultdict(list)
    for key in ATTR_KEYS:
        vals = [(e["class"], str(e["knowledge"].get("structural_attributes", {}).get(key, "")))
                for e in entries]
        miss = [c for c, v in vals if not v or UNSEEN.search(v)]
        for c, v in vals:
            if not v or UNSEEN.search(v):
                blind[key].append(c)
        print("%-18s %3d/%3d usable  (%.0f%% blind)"
              % (key, len(vals) - len(miss), len(vals), 100 * len(miss) / len(vals)))

    print("\n--- non-discriminative boilerplate ---")
    # a feature sentence shared by many classes cannot separate those classes
    feat_owner = defaultdict(set)
    for e in entries:
        for f in (e["knowledge"].get("discriminative_features") or []):
            key = re.sub(r"[\s，。、,\.]+", "", str(f)).lower()
            if len(key) >= 4:
                feat_owner[key].add(e["class"])
    shared = sorted(((v, k) for k, v in feat_owner.items() if len(v) >= 4), reverse=True)
    if not shared:
        print("none: no feature string recurs across >=4 classes")
    for owners, feat in shared[:15]:
        print("%2d classes | %s" % (len(owners), feat[:60]))

    print("\n--- feature list length ---")
    print(Counter(len(e["knowledge"].get("discriminative_features") or []) for e in entries))

    print("\n--- classes whose knowledge is thin (candidates for re-generation) ---")
    thin = []
    for e in entries:
        k = e["knowledge"]
        n_unseen = sum(1 for v in k.get("structural_attributes", {}).values() if UNSEEN.search(str(v)))
        n_feat = len(k.get("discriminative_features") or [])
        if n_unseen >= 4 or n_feat <= 2:
            thin.append((e["class"], n_unseen, n_feat))
    for c, u, f in sorted(thin, key=lambda t: (-t[1], t[2])):
        print("  %-14s blind_attrs=%d features=%d" % (c, u, f))
    if not thin:
        print("  none")

    print("\n--- validation issues carried over ---")
    flagged = [e["class"] for e in entries if e.get("validation_issues")]
    print("  %d %s" % (len(flagged), flagged))

    out = {
        "n_classes": len(entries),
        "blind_by_attribute": {k: sorted(v) for k, v in blind.items()},
        "boilerplate_features": [{"text": f, "classes": sorted(o)} for o, f in shared],
        "thin_classes": [{"class": c, "blind_attrs": u, "n_features": n} for c, u, n in thin],
        "validation_flagged": flagged,
    }
    path = os.path.join(ROOT, "knowledge", "kb_audit.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nwrote knowledge/kb_audit.json")


if __name__ == "__main__":
    main()
