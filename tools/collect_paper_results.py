#!/usr/bin/env python
"""Collect every number the thesis tables need, straight from the run artifacts.

Hand-copying results into a .docx is how wrong numbers end up in a thesis, so this
reads runs/*/eval_test.json and runs/baseline_*_zeroshot.json and emits exactly the
cell values `tools/fill_results_docx.py` writes, keyed by the tables' own header
text. A measurement that does not exist is simply absent from the output, which makes
the filler leave that cell as 待填 instead of guessing.

Each paper row is the MEAN over every repeat that exists for that configuration
(CONFIGS below), because one run is one draw from a distribution whose spread we
measured at 2.05 Macro-F1 points. The spread is reported alongside, so the prose can
say how many repeats stand behind a number instead of implying false precision.

  python -m tools.collect_paper_results --root . --out paper_results.json
"""

import argparse
import json
import os

# paper row -> run directories, seed 0 first then repeats. A configuration that only
# exists once is listed once; the repeat count is reported rather than padded.
CONFIGS = {
    "ours": ["align_full_kb", "align_full_kb_s1"],       # Region+Scene+Text, full model
    "ro": ["region_only", "region_only_s1"],             # Region only
    "rs": ["region_scene", "region_scene_s1"],           # Region + Scene
    "kn": ["no_hard"],                                   # + Text, without hard negatives
}
# 表4-2 rows use the 4-channel (RGB + GT-box mask) baselines, so every row in that
# table sees the same input as our Region. The 3-channel originals are still read, to
# report what the extra channel was worth on each side.
BASE = {"baseline_resnet50_mask": "resnet50", "baseline_vit_b_16_mask": "vit",
        "baseline_swin_b_mask": "swin"}
BASE_RGB = {"baseline_resnet50": "resnet50", "baseline_vit_b_16": "vit",
            "baseline_swin_b": "swin"}
ZERO = {"baseline_siglip2_zeroshot.json": "siglip_zeroshot",
        "baseline_clip-vit-large-patch14_zeroshot.json": "clip_zeroshot"}
SMALL = [("<64px", 181), ("64-128", 394)]        # 19.1% of the test entities

# entity-extraction rows: (cell key, detector run tag, classifier run, entity json).
# The tag is what detect_infer.py wrote <tag>_test_summary.json / _test_boxes.json
# under; the entity file is the one rst.entity_extract --out produced. A name holding a
# "/" is a path from the project root -- those rows are written under det_runs/ because
# their recogniser is a VLM, not one of our checkpoints.
#
# Every row is scored at the operating point its own detector can actually reach, and
# the resulting candidate density per image is exported as <row>_cand so the 表注 can
# show the four budgets are comparable (real density is 1.80 boxes/image):
#   YOLO / RT-DETR   0.5 confidence floor  (measured 1.60 / 1.74 candidates per image)
#   Grounding DINO   all of its top-3 boxes -- its scores peak at 0.58, so any floor
#                    above that leaves almost nothing and would rank calibration (3.00)
#   Qwen3-VL         all boxes it emitted -- it returns no score at all, so its output
#                    set IS its operating point (~1.5)
# GDINO is the row given the MOST candidates and still finishes last, so its gap cannot
# be a budget artifact; its F1 is 0.729 localisation recall x 0.416 recognition.
# A uniform top-3 budget was measured too (entity_*_top3.json: YOLO 69.29, RT-DETR
# 52.87) and is deliberately NOT the reported column, because capping a well-thresholded
# detector at 3 candidates throws away correct boxes and costs it ~13 points of F1.
DETECTORS = [
    ("det_yolo", "yolo11x_640_b16", "align_full_kb", "entity_yolo_s05.json"),
    ("det_rtdetr", "rtdetr-l_640", "align_full_kb", "entity_rtdetr_s05.json"),
    ("det_gdino", "gdino_tiny", "", "det_runs/entity_gdino_top3.json"),
    ("det_qwen", "qwen3vl", "", "det_runs/entity_qwen_all.json"),
    # 4.4.2 defines Ours as the full pipeline = the RT-DETR main detector + the RST
    # module, so its numbers are the RT-DETR row's by construction. The table note
    # says so explicitly; duplicating them is honest, inventing a difference is not.
    ("det_ours", "rtdetr-l_640", "align_full_kb", "entity_rtdetr_s05.json"),
]


def pct(x, nd=2):
    return "%.*f" % (nd, 100 * float(x))


def load(root, *parts):
    p = os.path.join(root, *parts)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def runs_of(root, key):
    out = []
    for name in CONFIGS[key]:
        art = load(root, "runs", name, "eval_test.json")
        if art and "with_semantic" in art:
            out.append((name, art))
    return out


def agg(arts, field, section="with_semantic"):
    """Mean over repeats -> (mean, n, spread); (None, 0, 0.0) if nothing measured."""
    vals = [a[section][field] for _, a in arts
            if a.get(section) and a[section].get(field) is not None]
    if not vals:
        return None, 0, 0.0
    return sum(vals) / len(vals), len(vals), max(vals) - min(vals)


def agg_bucket(arts, bucket, field):
    vals = [a["by_entity_size"][bucket][field] for _, a in arts
            if bucket in (a.get("by_entity_size") or {})]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper_results.json")
    ap.add_argument("--zeroshot-template", default="photo", choices=["raw", "photo"],
                    help="which prompt the zero-shot rows report; 'photo' is the usual "
                         "template protocol, 'raw' is the bare class name")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    cells, fmt, missing, spread = {}, {}, [], {}

    def want(name, value):
        if value is None:
            missing.append(name)
        else:
            fmt[name] = value

    # Fallback of last resort for rows whose artifact is temporarily gone (see the
    # _provenance field of each entry). A live run directory always wins.
    archive = load(root, "tools", "archived_results.json") or {}

    arts = {k: runs_of(root, k) for k in CONFIGS}
    for k, v in arts.items():
        if not v:
            missing.append("config:" + k)
        spread[k] = len(v)
        print("  %-6s %d repeat(s): %s" % (k, len(v), ", ".join(n for n, _ in v)))

    # ---- 表4-3: Ours / pure-vision baselines / zero-shot -------------------
    if arts["ours"]:
        row = {}
        for field, col in (("top1", "114类Top-1"), ("military_top1", "103类军机Top-1"),
                           ("macro_f1", "Macro-F1"), ("top5", "Top-5")):
            m, _, _ = agg(arts["ours"], field)
            if m is None:
                missing.append("ours:" + field)
            else:
                row[col] = pct(m)
        cells["ours"] = row

    for run, key in BASE.items():
        art = load(root, "runs", run, "eval_test.json")
        if art is None:
            if run in archive:
                cells[key] = {k: v for k, v in archive[run].items() if not k.startswith("_")}
                print("  ARCHIVE fallback for %s -- no live artifact, using the recorded "
                      "measurement (see tools/archived_results.json)" % run)
                continue
            missing.append(run)
            continue
        m = art["test"]
        cells[key] = {"114类Top-1": pct(m["top1"]), "103类军机Top-1": pct(m["military_top1"]),
                      "Macro-F1": pct(m["macro_f1"]), "Top-5": pct(m["top5"])}

    # what the mask channel is worth, per backbone, and what the baseline learning
    # rate is worth -- the two factors behind 表4-2's gap that are NOT our fusion
    for run, key in BASE_RGB.items():
        art = load(root, "runs", run, "eval_test.json")
        msk = load(root, "runs", run + "_mask", "eval_test.json")
        if art and msk:
            a1, m1 = art["test"], msk["test"]
            fmt["maskd_%s_top1" % key] = "%+.2f" % (100 * (m1["top1"] - a1["top1"]))
            fmt["maskd_%s_f1" % key] = "%+.2f" % (100 * (m1["macro_f1"] - a1["macro_f1"]))
            fmt["rgb_%s_top1" % key] = pct(a1["top1"])
            fmt["rgb_%s_f1" % key] = pct(a1["macro_f1"])
    lrv = load(root, "runs", "baseline_vit_b_16_lr1e4", "eval_test.json")
    if lrv:
        rgb = load(root, "runs", "baseline_vit_b_16", "eval_test.json")
        fmt["vit_lr1e4_top1"] = pct(lrv["test"]["top1"])
        fmt["vit_lr1e4_f1"] = pct(lrv["test"]["macro_f1"])
        if rgb:
            fmt["vit_lr_gain_top1"] = "%+.2f" % (100 * (lrv["test"]["top1"] - rgb["test"]["top1"]))
            fmt["vit_lr_gain_f1"] = "%+.2f" % (100 * (lrv["test"]["macro_f1"] - rgb["test"]["macro_f1"]))
    else:
        missing.append("baseline_vit_b_16_lr1e4")

    for fname, key in ZERO.items():
        art = load(root, "runs", fname)
        if art is None:
            missing.append(fname)
            continue
        m = art[args.zeroshot_template]
        row = {"114类Top-1": pct(m["top1"]), "103类军机Top-1": pct(m["military_top1"]),
               "Macro-F1": pct(m["macro_f1"])}
        if "top5" in m:
            row["Top-5"] = pct(m["top5"])
        else:
            missing.append(fname + ":top5")
        cells[key] = row

    # ---- 表4-6: Region only / +Scene / +Text ------------------------------
    for key, row_name in (("ro", "region_only"), ("rs", "region_scene"),
                          ("ours", "region_scene_text")):
        if not arts[key]:
            continue
        t1, _, _ = agg(arts[key], "top1")
        f1, _, _ = agg(arts[key], "macro_f1")
        cells[row_name] = {"Top-1": pct(t1), "Macro-F1": pct(f1)}

    # ---- 表4-7: Baseline / +MLLM Knowledge / Full Model -------------------
    for key, row_name in (("rs", "abl_baseline"), ("kn", "abl_knowledge"),
                          ("ours", "abl_full")):
        if not arts[key]:
            continue
        t1, _, _ = agg(arts[key], "top1")
        row = {"Top-1": pct(t1)}
        hp, _, _ = agg(arts[key], "hard_pair_acc")
        if hp is None:
            missing.append(row_name + ":hard_pair_acc")
        else:
            row["Hard Pair Accuracy"] = pct(hp)
        cells[row_name] = row

    # ---- prose placeholders ----------------------------------------------
    o, r, s, k = arts["ours"], arts["ro"], arts["rs"], arts["kn"]
    if o:
        for field, name in (("top1", "ours_top1"), ("top5", "ours_top5"),
                            ("macro_f1", "ours_f1"), ("military_top1", "ours_mil")):
            m, n, sp = agg(o, field)
            want(name, pct(m))
            spread[name] = (n, 100 * sp)
        fmt["n_repeat"] = str(spread["ours"])
    sw = load(root, "runs", "baseline_swin_b", "eval_test.json")
    if sw and o:
        t1, _, _ = agg(o, "top1")
        f1, _, _ = agg(o, "macro_f1")
        fmt["swin_top1"] = pct(sw["test"]["top1"])
        fmt["swin_f1"] = pct(sw["test"]["macro_f1"])
        fmt["gap_top1"] = "%+.2f" % (100 * (t1 - sw["test"]["top1"]))
        fmt["gap_f1"] = "%+.2f" % (100 * (f1 - sw["test"]["macro_f1"]))
    for fname, key in ZERO.items():
        art = load(root, "runs", fname)
        if art:
            t = args.zeroshot_template
            fmt[key.split("_")[0] + "_top1"] = pct(art[t]["top1"])
            # the evidence that the SigLIP 2 row is a measurement and not a bug:
            # a coarse control that must work if the alignment is sound, plus how far
            # apart the class-name embeddings actually are
            p = key.split("_")[0]
            ctl = art.get("controls") or {}
            if "military_vs_civilian_acc" in ctl:
                fmt[p + "_ctl"] = "%.2f" % ctl["military_vs_civilian_acc"]
            for src, dst in (("text_pairwise_cos_mean", "_textcos"),
                             ("mean_image_text_cos", "_imgtxt"),
                             ("distinct_predicted_classes", "_distinct")):
                if src in art[t]:
                    v = art[t][src]
                    fmt[p + dst] = ("%.4f" % v) if isinstance(v, float) else str(v)

    if o and r and s:
        for key, tag in (("ro", "ro"), ("rs", "rs"), ("ours", "rst")):
            t1, n1, sp1 = agg(arts[key], "top1")
            f1, n2, sp2 = agg(arts[key], "macro_f1")
            want(tag + "_top1", pct(t1))
            want(tag + "_f1", pct(f1))
            if n1 > 1:
                fmt["sp_%s_top1" % tag] = "%.2f" % (100 * sp1)
                fmt["sp_%s_f1" % tag] = "%.2f" % (100 * sp2)
        rs_t1, _, _ = agg(s, "top1")
        ro_t1, _, _ = agg(r, "top1")
        want("agg_top1", "%+.2f" % (100 * (rs_t1 - ro_t1)))
        rs_f1, _, _ = agg(s, "macro_f1")
        ro_f1, _, _ = agg(r, "macro_f1")
        o_t1b, _, _ = agg(o, "top1")
        o_f1b, _, _ = agg(o, "macro_f1")
        want("scene_f1", "%+.2f" % (100 * (rs_f1 - ro_f1)))
        want("text_top1", "%+.2f" % (100 * (o_t1b - rs_t1)))
        want("text_f1", "%+.2f" % (100 * (o_f1b - rs_f1)))
        n_tot = sum(n for _, n in SMALL)
        for field, tag in (("top1", "small_top1"), ("macro_f1", "small_f1")):
            d = sum(n * ((agg_bucket(s, b, field) or 0) - (agg_bucket(r, b, field) or 0))
                    for b, n in SMALL) / n_tot
            want(tag, "%+.2f" % (100 * d))
        # how much each side of that comparison moves between repeats: if the spread
        # is the same size as the gain, the gain is not established yet
        for b, _ in SMALL:
            for key, tag in (("rs", "rs"), ("ro", "ro")):
                vals = [a["by_entity_size"][b]["top1"] for _, a in arts[key]
                        if b in (a.get("by_entity_size") or {})]
                if len(vals) > 1:
                    fmt["spread_%s_%s" % (tag, b.replace("-", "_").replace("<", "lt"))] = \
                        "%.2f" % (100 * (max(vals) - min(vals)))

    if k and o:
        kn_t1, _, _ = agg(k, "top1")
        o_t1, _, _ = agg(o, "top1")
        want("kn_top1", pct(kn_t1))
        want("full_top1", pct(o_t1))
        want("hard_delta", "%+.2f" % (100 * (o_t1 - kn_t1)))
        for key, tag in (("rs", "base"), ("kn", "kn"), ("ours", "full")):
            hp, _, _ = agg(arts[key], "hard_pair_acc")
            want(tag + "_hp", pct(hp) if hp is not None else None)
        bhp, _, _ = agg(s, "hard_pair_acc")
        khp, _, _ = agg(k, "hard_pair_acc")
        fhp, _, _ = agg(o, "hard_pair_acc")
        if None not in (bhp, khp, fhp):
            fmt["kn_drop"] = "%.2f" % (100 * (bhp - khp))
            fmt["hard_gain"] = "%+.2f" % (100 * (fhp - khp))
            fmt["hard_vs_base"] = "%+.2f" % (100 * (fhp - bhp))
        sem, nsem = [], []
        for _, a in o:
            if a.get("without_semantic"):
                nsem.append(a["without_semantic"]["top1"])
            if (a.get("diagnostics") or {}).get("sem_only_top1_pct") is not None:
                sem.append(a["diagnostics"]["sem_only_top1_pct"])
        if sem and nsem:
            fmt["sem_only"] = "%.2f" % (sum(sem) / len(sem))
            fmt["sem_gap"] = "%+.2f" % (sum(sem) / len(sem) - 100 * o_t1)
            fmt["sem_delta"] = "%+.2f" % (100 * (o_t1 - sum(nsem) / len(nsem)))
        else:
            missing.append("ours:semantic diagnostics")

    # ---- 掩码通道的代价：region_only 与其去掉第 4 通道的对照 -----------------
    nm = load(root, "runs", "region_only_nomask", "eval_test.json")
    if nm and r:
        ro_t1, _, _ = agg(r, "top1")
        ro_f1, _, _ = agg(r, "macro_f1")
        nmt = nm["with_semantic"]
        fmt["nomask_top1"] = pct(nmt["top1"])
        fmt["nomask_f1"] = pct(nmt["macro_f1"])
        fmt["mask_gain_top1"] = "%+.2f" % (100 * (ro_t1 - nmt["top1"]))
        fmt["mask_gain_f1"] = "%+.2f" % (100 * (ro_f1 - nmt["macro_f1"]))
    else:
        missing.append("region_only_nomask")

    # ---- 表4-2: MLLM 直接识别行 -------------------------------------------
    MLLM = {"runs/mllm_qwen3-vl-8b-instruct_region.json": ("qwen_region", "qwen_region"),
            "runs/mllm_qwen3-vl-8b-instruct_region_scene.json": ("qwen_region_scene", "qwen_rs")}
    for rel, (key, tag) in MLLM.items():
        art = load(root, *rel.split("/"))
        if art is None:
            missing.append(rel)
            continue
        s = art["summary"]
        cells[key] = {"114类Top-1": pct(s["top1"]), "Macro-F1": pct(s["macro_f1"]),
                      "Top-5": pct(s["top5"]), "103类军机Top-1": pct(s["military_top1"])}
        want(tag + "_top1", pct(s["top1"]))
        want(tag + "_top5", pct(s["top5"]))
        want(tag + "_f1", pct(s["macro_f1"]))
        want(tag + "_unparsed", str(s["unparsed"]))
        fmt[tag + "_unparsed_pct"] = "%.1f" % (100 * s["unparsed"] / max(1, s["n"]))
    if "qwen_region_top1" in fmt and "ours_top1" in fmt:
        fmt["qwen_gap"] = "%.2f" % (float(fmt["ours_top1"]) - float(fmt["qwen_rs_top1"]))

    # ---- 表4-3: 检测器 + 分类器的实体抽取行 --------------------------------
    for row_key, tag, clf, ent_name in DETECTORS:
        # detect_infer.py names the summary after the boxes file it wrote
        summ = (load(root, "det_runs", "%s_test_boxes_summary.json" % tag)
                or load(root, "det_runs", "%s_test_summary.json" % tag))
        if summ is None:
            missing.append("detector:" + tag)
            continue
        row = {"AP50": pct(summ["AP50"]), "mAP@50:95": pct(summ["mAP50_95"])}
        ent = (load(root, ent_name) if "/" in ent_name
               else load(root, "runs", clf, ent_name))
        if ent is None:
            missing.append("entity metrics:" + tag)
        else:
            row["Entity Precision"] = pct(ent["entity_precision"])
            row["Entity Recall"] = pct(ent["entity_recall"])
            row["Entity F1"] = pct(ent["entity_f1"])
            fmt[tag + "_det_f1"] = pct(ent["detection_f1"])
            fmt[tag + "_det_r"] = pct(ent["detection_recall"])
            fmt[tag + "_recog"] = pct(ent["recognition_top1_given_match"])
            # the evidence that the four rows are comparable: what each operating point
            # costs in candidates per image, against the real density of 1.80
            fmt[row_key.replace("det_", "") + "_cand"] = "%.2f" % (
                ent["n_detections"] / float(ent["n_images"]))
        cells[row_key] = row
        short = row_key.replace("det_", "")
        for src, dst in (("AP50", "_ap50"), ("mAP@50:95", "_map"),
                         ("Entity F1", "_ent_f1"), ("Entity Recall", "_ent_r"),
                         ("Entity Precision", "_ent_p")):
            if src in row:
                fmt[short + dst] = row[src]
        # the error decomposition every detector row's prose is built from
        for src, dst in (("_det_f1", "_det_f1"), ("_recog", "_recog"),
                         ("_det_r", "_det_r")):
            if tag + src in fmt:
                fmt[short + dst] = fmt[tag + src]
        if short in ("yolo", "rtdetr"):
            # the same rows forced onto a uniform top-3 candidate budget. Reported in
            # the 表注: it is the sensitivity check that lets a reviewer see the two
            # VLM rows are not being flattered or crippled by their operating point.
            alt = load(root, "runs", clf, "entity_%s_top3.json" % short)
            if alt:
                fmt[short + "_ent_f1_budget"] = pct(alt["entity_f1"])
                fmt[short + "_ent_r_budget"] = pct(alt["entity_recall"])
            else:
                missing.append("entity_%s_top3" % short)
        if short == "yolo":
            # the same metrics at a lower confidence floor, so the threshold the table
            # reports is visible and its sensitivity is quantified
            alt = load(root, "runs", clf, "entity_yolo_s025.json")
            if alt:
                fmt["yolo_ent_f1_lo"] = pct(alt["entity_f1"])
                fmt["yolo_ent_r_lo"] = pct(alt["entity_recall"])
            else:
                missing.append("entity_yolo_s025")

    out = {"cells": cells, "fmt": fmt, "missing": missing,
           "repeats": {kk: [n for n, _ in v] for kk, v in arts.items()},
           "spread_pct": {kk: v for kk, v in spread.items() if isinstance(v, tuple)}}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\nwrote %s | %d table rows, %d prose values, %d missing"
          % (args.out, len(cells), len(fmt), len(missing)))
    for m in missing:
        print("  MISSING %s" % m)
    for name, (n, sp) in sorted(out["spread_pct"].items()):
        print("  spread  %-16s n=%d  max-min %.2f points" % (name, n, sp))


if __name__ == "__main__":
    main()
