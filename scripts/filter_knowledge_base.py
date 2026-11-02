"""Strip shortcut features from the semantic knowledge base.

The teacher model fills part of its "discriminative features" with livery,
insignia, tail numbers and carried stores. Those correlate with the label
(e.g. military grey vs airline titles) without describing the airframe, so a
model trained on them would appear to identify aircraft while actually reading
paint schemes. Clauses are evaluated individually so that a mixed sentence like
"T型尾翼结构，垂直尾翼上部为黑色" keeps its structural half.
"""

import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "knowledge", "aircraft_semantic_kb.json")
DST = os.path.join(ROOT, "knowledge", "aircraft_semantic_kb_filtered.json")
RPT = os.path.join(ROOT, "knowledge", "livery_filter_report.json")

# Paint schemes, titles, insignia, registration marks.
LIVERY = re.compile(
    r"涂装|喷涂|喷了|迷彩|字样|文字|字母|编号|注册号|尾号|序号|队名|机名"
    r"|印有|标有|写有|标注|题字|签名|涂鸦"
    r"|徽章|徽标|机徽|军徽|队徽|国籍|国旗|旗标|标志|标识|logo"
    r"|五角星|鲨鱼嘴|眼睛图案|色带|条纹|色块|色条|色线"
    r"|灰色|白色|蓝色|红色|绿色|黄色|橙色|紫色|银色|黑色|深色|浅色|米色|棕色|褐色|青色|双色|三色|相间"
    r"|背景|天空|地面|跑道|停机坪|云层|人群|车辆")

# Stores hung on the aircraft. Empty hardpoints and pylons themselves stay,
# which is what KEEP_OVERRIDE protects.
LOADOUT = re.compile(
    r"导弹|炸弹|火箭弹|鱼雷|水雷|机炮|机枪|航炮|荚舱|吊舱"
    r"|副油箱|外部油箱|油箱|弹药|载弹|挂弹"
    r"|挂载.{0,6}(枚|发|具)")

# Engine / rotor mounts are airframe structure, not stores.
KEEP_OVERRIDE = re.compile(
    r"发动机|引擎|旋翼|螺旋桨|整流罩|挂架|挂点|发射架|天线|雷达罩|舷窗|座舱盖|吊舱式")

SPLIT = re.compile(r"[，,、;；。]")
CONTINUATION = re.compile(r"^(呈|其|并|且|共|以及|其中|位于|安装|分布|形成|构成|呈现)")


def classify(clause):
    if not clause:
        return "empty"
    if LIVERY.search(clause):
        return "livery"
    if LOADOUT.search(clause) and not KEEP_OVERRIDE.search(clause):
        return "loadout"
    return "keep"


def clean_text(text):
    """Return (kept_text, [dropped clauses]) for a comma-separated sentence."""
    clauses = [c.strip() for c in SPLIT.split(str(text)) if c.strip()]
    kept, dropped = [], []
    prev_dropped = False
    for cl in clauses:
        tag = classify(cl)
        if tag == "keep":
            kept.append(cl)
            prev_dropped = False
        else:
            dropped.append({"text": cl, "rule": tag})
            prev_dropped = True
    if prev_dropped and len(kept) < len(clauses):
        # drop trailing fragments that only made sense for a removed store
        while kept and CONTINUATION.match(kept[-1]):
            dropped.append({"text": kept.pop(), "rule": "orphan_modifier"})
    return "，".join(kept), dropped


def main():
    kb = json.load(open(SRC, encoding="utf-8"))
    report = defaultdict(list)
    stats = Counter()
    out_entries = []

    for e in kb["entries"]:
        cls, k = e["class"], dict(e["knowledge"])
        rec = {"class": cls, "removed": []}

        feats, feat_drop = [], []
        for f in k.get("discriminative_features") or []:
            kept, dropped = clean_text(f)
            if dropped:
                feat_drop.append({"original": f, "kept": kept, "dropped": dropped})
            if kept:
                feats.append(kept)
        rec["removed"].extend({"field": "discriminative_features", **d} for d in feat_drop)
        k["discriminative_features"] = feats

        attrs, attr_drop = {}, []
        for key, val in (k.get("structural_attributes") or {}).items():
            kept, dropped = clean_text(val)
            if dropped:
                attr_drop.append({"attr": key, "original": val, "kept": kept, "dropped": dropped})
            attrs[key] = kept if kept else "不可见"
        k["structural_attributes"] = attrs
        rec["removed"].extend({"field": "structural_attributes", **d} for d in attr_drop)

        diffs, diff_drop = [], []
        for d in k.get("similar_class_differences") or []:
            head, _, tail = str(d).partition(":")
            kept, dropped = clean_text(tail)
            if dropped:
                diff_drop.append({"original": d, "kept": d, "dropped": dropped})
                if kept and len(kept) > 4:
                    diffs.append("%s: %s" % (head.strip(), kept))
                else:
                    pass
            else:
                diffs.append(str(d))
        k["similar_class_differences"] = diffs
        rec["removed"].extend({"field": "similar_class_differences", **d} for d in diff_drop)

        od, od_drop = clean_text(k.get("overall_description", ""))
        k["overall_description_original"] = k.get("overall_description", "")
        # If every clause was livery, keep nothing rather than silently falling
        # back to the contaminated text -- an empty description is skipped when
        # the semantic prototypes are built, and the other fields carry the rest.
        k["overall_description"] = od
        if od_drop:
            rec["removed"].append({"field": "overall_description",
                                   "original": k["overall_description_original"],
                                   "kept": od, "dropped": od_drop})
        k["_livery_free"] = not any(
            r["field"] != "overall_description" for r in rec["removed"])

        stats["classes_touched"] += 1 if rec["removed"] else 0
        stats["feat_dropped"] += sum(1 for x in rec["removed"] if x["field"] == "discriminative_features")
        stats["feat_kept"] += len(feats)
        if len(feats) < 3:
            stats["thin_classes"] += 1
        report[cls] = rec["removed"]
        out_entries.append({**e, "knowledge": k, "_filter_removed": len(rec["removed"])})

    out = {
        "meta": {**kb["meta"],
                 "filtered_from": os.path.basename(SRC),
                 "filter_rules": {
                     "livery": LIVERY.pattern,
                     "loadout": LOADOUT.pattern,
                     "keep_override": KEEP_OVERRIDE.pattern},
                 "note": "Livery / insignia / registration / carried-store clauses removed. "
                         "Original preserved in aircraft_semantic_kb.json."},
        "n_classes_ok": len(out_entries),
        "entries": out_entries,
    }
    with open(DST, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    with open(RPT, "w", encoding="utf-8") as fh:
        json.dump({"stats": dict(stats), "per_class": report}, fh, ensure_ascii=False, indent=2)

    print("classes touched by filter : %d/114" % stats["classes_touched"])
    print("discriminative features   : %d kept, %d entries had a clause dropped"
          % (stats["feat_kept"], stats["feat_dropped"]))
    print("classes left with <3 features: %d" % stats["thin_classes"])
    thin = sorted(((len(e["knowledge"]["discriminative_features"]), e["class"])
                   for e in out_entries))[:15]
    print("thinnest:", ", ".join("%s(%d)" % (c, n) for n, c in thin))
    print("\nwrote knowledge/aircraft_semantic_kb_filtered.json + livery_filter_report.json")


if __name__ == "__main__":
    main()
